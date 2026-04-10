Base.@kwdef mutable struct FiniteIntervalLogitRandomWalk{T,S} <: AbstractMHProposal
    lower::T
    upper::T
    transform::S
    step_size::Float64 = 0.6
    mh_steps::Int = 10
end

function FiniteIntervalLogitRandomWalk(dist; step_size=1.0, mh_steps=5)
    lower = minimum(dist)
    upper = maximum(dist)
    _check_finite_interval(lower, upper)
    transform = TransformVariables.as(Real, lower, upper)
    FiniteIntervalLogitRandomWalk(
        lower=lower,
        upper=upper,
        transform=transform,
        step_size=step_size,
        mh_steps=mh_steps,
    )
end

function _check_finite_interval(lower, upper)
    isfinite(lower) || throw(ArgumentError("proposal requires finite lower support bound"))
    isfinite(upper) || throw(ArgumentError("proposal requires finite upper support bound"))
    lower < upper || throw(ArgumentError("proposal requires lower < upper"))
    nothing
end

function draw_interior(dist; max_tries=10_000)
    lower = minimum(dist)
    upper = maximum(dist)
    _check_finite_interval(lower, upper)

    for _ in 1:max_tries
        value = rand(dist)
        if lower < value < upper
            return value
        end
    end

    throw(ArgumentError("failed to draw an interior point from the proposal support"))
end

function to_unconstrained(proposal::FiniteIntervalLogitRandomWalk, value)
    (; lower, upper) = proposal
    lower < value < upper || throw(ArgumentError("value must lie strictly inside the proposal support"))
    TransformVariables.inverse(proposal.transform, value)
end

from_unconstrained(proposal::FiniteIntervalLogitRandomWalk, η) =
    TransformVariables.transform(proposal.transform, η)

function logabsdetjac(proposal::FiniteIntervalLogitRandomWalk, η)
    _, logjac = TransformVariables.transform_and_logjac(proposal.transform, η)
    logjac
end

mutable struct NealAlgorithm2GaussianConvolution{
    D,
    T,
    U,
    P,
    S,
    W <: AbstractWrappedEBSample,
} <: AbstractNealAlgorithm
    prior::D
    α_dist::T
    A_dist::U
    A_proposal::P
    logα::Float64
    A::Float64
    A_accepts::Int
    A_proposals::Int
    last_A_accepts::Int
    last_A_proposals::Int
    components::Vector{W}
    empties::Vector{Int}
    assignments::Vector{Int}
    data::Vector{S}
end

track_parameters(::NealAlgorithm2GaussianConvolution) = true

function acceptance_rate(accepts::Integer, proposals::Integer)
    proposals == 0 && return NaN
    accepts / proposals
end

A_acceptance_rate(gc::NealAlgorithm2GaussianConvolution) =
    acceptance_rate(gc.A_accepts, gc.A_proposals)

last_A_acceptance_rate(gc::NealAlgorithm2GaussianConvolution) =
    acceptance_rate(gc.last_A_accepts, gc.last_A_proposals)

function NealAlgorithm2GaussianConvolution(
    Ss::AbstractVector{<:Empirikos.AbstractNormalSample};
    α_dist=Gamma(0.001, 100.0),
    prior,
    A_dist,
    A_init=draw_interior(A_dist),
    A_proposal=FiniteIntervalLogitRandomWalk(A_dist),
)
    A_proposal.lower < A_init < A_proposal.upper ||
        throw(ArgumentError("A_init must lie strictly inside the proposal support"))

    data = collect(Ss)
    all_Ss = wrap(_inflate_samples(data, A_init))
    empty_Ss = empty(all_Ss)

    NealAlgorithm2GaussianConvolution(
        prior,
        α_dist,
        A_dist,
        A_proposal,
        1.0,
        A_init,
        0,
        0,
        0,
        0,
        [all_Ss, empty_Ss],
        [2],
        ones(Int, length(data)),
        data,
    )
end

inflate_sample(sample::Empirikos.AbstractNormalSample, A) = NormalSample(response(sample), sqrt(var(sample) + A))

function _inflate_samples(Ss::AbstractVector{<:Empirikos.AbstractNormalSample}, A)
    inflate_sample.(Ss, Ref(A))
end

_A_observation_distribution(sample::Empirikos.AbstractNormalSample, θ, A) =
    Normal(θ, sqrt(var(sample) + A))

function _rebuild_components!(gc::NealAlgorithm2GaussianConvolution)
    inflated = _inflate_samples(gc.data, gc.A)
    grouped_indices = Dict{Int, Vector{Int}}()

    for i in eachindex(gc.assignments)
        push!(get!(() -> Int[], grouped_indices, gc.assignments[i]), i)
    end

    used_components = sort!(collect(keys(grouped_indices)))
    new_components = Vector{eltype(gc.components)}(undef, length(used_components) + 1)
    new_assignments = similar(gc.assignments)

    for (new_k, old_k) in enumerate(used_components)
        idxs = grouped_indices[old_k]
        merged = merge_samples(inflated[idxs])
        param = gc.components[old_k].param
        new_components[new_k] = WrappedEBSample(merged, length(idxs), param)
        new_assignments[idxs] .= new_k
    end

    new_components[end] = empty(new_components[1])
    gc.components = new_components
    gc.assignments .= new_assignments
    gc.empties = [length(new_components)]
    gc
end

function StatsBase.sample!(gc::NealAlgorithm2GaussianConvolution, i::Int)
    prior = gc.prior
    x = inflate_sample(gc.data[i], gc.A)
    old_comp = sub(gc.components[gc.assignments[i]], x)
    gc.components[gc.assignments[i]] = old_comp

    isempty(old_comp) && push!(gc.empties, gc.assignments[i])

    log_probs = [
        isempty(comp) ? -Inf : loglikelihood(x, comp.param) + log(comp.n) for
        comp in gc.components
    ]

    log_probs[first(gc.empties)] = logpdf(prior, x) + gc.logα

    new_k = sample(Weights(exp.(log_probs)))
    gc.components[new_k] = add(gc.components[new_k], x)
    gc.assignments[i] = new_k

    if new_k == first(gc.empties)
        popfirst!(gc.empties)
        if isempty(gc.empties)
            empty_Ss = empty(gc.components[1])
            push!(gc.components, empty_Ss)
            push!(gc.empties, length(gc.components))
        end
    end

    gc
end

function _log_A_likelihood(gc::NealAlgorithm2GaussianConvolution, A)
    sum(eachindex(gc.data)) do i
        θ = gc.components[gc.assignments[i]].param
        logpdf(_A_observation_distribution(gc.data[i], θ, A), response(gc.data[i]))
    end
end

function _log_A_posterior(gc::NealAlgorithm2GaussianConvolution, A)
    logpdf(gc.A_dist, A) + _log_A_likelihood(gc, A)
end

function _log_A_posterior(
    gc::NealAlgorithm2GaussianConvolution,
    proposal::FiniteIntervalLogitRandomWalk,
    η,
)
    A = from_unconstrained(proposal, η)
    _log_A_posterior(gc, A) + logabsdetjac(proposal, η)
end

function sample_A!(gc::NealAlgorithm2GaussianConvolution)
    proposal = gc.A_proposal
    η = to_unconstrained(proposal, gc.A)
    log_target = _log_A_posterior(gc, proposal, η)
    A = gc.A
    accepts = 0
    proposals = 0

    for _ in Base.OneTo(proposal.mh_steps)
        proposals += 1
        candidate_η = η + proposal.step_size * randn()
        candidate_log_target = _log_A_posterior(gc, proposal, candidate_η)

        if -Random.randexp() < candidate_log_target - log_target
            accepts += 1
            η = candidate_η
            log_target = candidate_log_target
            A = from_unconstrained(proposal, η)
        end
    end

    gc.A_accepts += accepts
    gc.A_proposals += proposals
    gc.last_A_accepts = accepts
    gc.last_A_proposals = proposals

    if !isequal(A, gc.A)
        gc.A = A
        _rebuild_components!(gc)
    end

    gc
end

function StatsBase.sample!(gc::NealAlgorithm2GaussianConvolution)
    isempty(gc.data) &&
        throw(ArgumentError("can't sample from Gibbs sampler with empty .data field"))

    for i in eachindex(gc.data)
        sample!(gc, i)
    end

    if track_parameters(gc)
        for i in eachindex(gc.components)
            !isempty(gc.components[i]) && sample_component_param!(gc, i)
        end
    end

    sample_A!(gc)
    sample_α!(gc)

    if length(gc.empties) > 100
        cleanup_components!(gc)
    end

    gc
end

mutable struct NealAlgorithmGaussianConvolutionSamples{N, S}
    gc::N
    assignments::Matrix{Int}
    components::Vector{Vector{S}}
    log_αs::Vector{Float64}
    As::Vector{Float64}
end

assignments(gf::NealAlgorithmGaussianConvolutionSamples) = gf.assignments

function StatsBase.fit!(gc::NealAlgorithm2GaussianConvolution, progress=true; samples=5000, burnin=samples ÷ 10)
    assignments = Matrix{Int}(undef, length(gc.data), samples)
    components = Vector{typeof(gc.components)}()
    log_αs = Vector{Float64}(undef, samples)
    As = Vector{Float64}(undef, samples)

    @showprogress (progress ? 1 : Inf) "Gibbs sampling burnin" for _ in 1:burnin
        sample!(gc)
    end

    @showprogress (progress ? 1 : Inf) "Gibbs sampling" for i in 1:samples
        sample!(gc)
        assignments[:, i] .= gc.assignments
        push!(components, copy(gc.components))
        log_αs[i] = gc.logα
        As[i] = gc.A
    end

    NealAlgorithmGaussianConvolutionSamples(gc, assignments, components, log_αs, As)
end

function _posterior_means(samples::NealAlgorithmGaussianConvolutionSamples)
    n_units, n_runs = size(samples.assignments)
    postmeans_mat = Matrix{Float64}(undef, n_units, n_runs)
    params = Vector{Float64}(undef, n_units)

    for j in 1:n_runs
        sqrt_A = sqrt(samples.As[j])
        params .= getproperty.(samples.components[j][samples.assignments[:, j]], :param)
        postmeans_mat[:, j] .= [
            PosteriorMean(samples.gc.data[i])(Normal(params[i], sqrt_A)) for i in 1:n_units
        ]
    end

    vec(mean(postmeans_mat, dims=2))
end

function _posterior_cis(samples::NealAlgorithmGaussianConvolutionSamples)
    n_units, n_runs = size(samples.assignments)
    lower_vec = Vector{Float64}(undef, n_units)
    upper_vec = Vector{Float64}(undef, n_units)
    weights = fill(1 / n_runs, n_runs)

    for i in 1:n_units
        comps = [samples.components[j][samples.assignments[i, j]] for j in 1:n_runs]
        params = getproperty.(comps, :param)
        prior_dbn = MixtureModel(Normal.(params, sqrt.(samples.As)), weights)
        post_dbn = Empirikos.posterior(samples.gc.data[i], prior_dbn)
        lower_vec[i] = quantile(post_dbn, 0.025)
        upper_vec[i] = quantile(post_dbn, 0.975)
    end

    lower_vec, upper_vec
end
