
abstract type AbstractNealAlgorithm end


function StatsBase.sample!(gc::AbstractNealAlgorithm)
    isempty(gc.data) &&
        throw(ArgumentError("can't sample from Gibbs sampler with empty .data field"))
    for i in 1:length(gc.data)
        sample!(gc, i) # update cluster assignment
    end
    if track_parameters(gc)
        for i in 1:length(gc.components)
            !isempty(gc.components[i]) && sample_component_param!(gc, i) # update parameter assignment
        end 
    end
    sample_α!(gc)   # update concentration parameter α 
    # remark: for the Polya Tree algorithm, also need to update the Polya Tree & Zs.
    if length(gc.empties) > 100
        cleanup_components!(gc)
    end
    return gc
end

#---------------------------------------
# Algorithm 2 
#---------------------------------------
mutable struct NealAlgorithm2{D, T,  S, W <: AbstractWrappedEBSample{S}}  <: AbstractNealAlgorithm
    prior::D
    α_dist::T
    logα::Float64
    components::Vector{W}
    empties::Vector{Int}
    assignments::Vector{Int}
    data::Vector{S}
end

track_parameters(::NealAlgorithm2) = true 

# basically: 
# recode
function cleanup_components!(gc)
    idx_nonempty = findall(!isempty, gc.components)
    idx_empty = findfirst(isempty, gc.components)
    empty_comp = gc.components[idx_empty]
    recode_dict = Dict(idx_nonempty .=> 1:length(idx_nonempty))
    gc.assignments .= getindex.(Ref(recode_dict), gc.assignments)
    gc.components = gc.components[idx_nonempty]
    push!(gc.components, empty_comp)
    gc.empties = [length(gc.components)]
end

function NealAlgorithm2(
    Ss::AbstractVector;
    α_dist =  Gamma(0.001,100.0),
    prior = _default_prior(Ss)
    ) 
    all_Ss = wrap(Ss)
    empty_Ss = empty(all_Ss)
    NealAlgorithm2(prior,
             α_dist,
             1.0,
             [all_Ss, empty_Ss],
             [2],
             ones(Int, length(Ss)),
             Ss)
end




mutable struct NealAlgorithmSamples{N, S} 
    gc::N
    assignments::Matrix{Int}
    components::Vector{Vector{S}}
    log_αs::Vector{Float64}
end

function _pval_fun(samples::NealAlgorithmSamples, mu_hats)
    pval_mat = Matrix{Float64}(undef, size(samples.assignments)...)
    for j in 1:size(samples.assignments, 2)
        pval_mat[:,j] .= 2 .* ccdf.(Normal.(0, sqrt.(getproperty.(samples.components[j][samples.assignments[:,j]], :param)) ), abs.(mu_hats))
    end
    mean(pval_mat, dims=2) |> vec
end 


function _posterior_means(samples::NealAlgorithmSamples)
    postmeans_mat = Matrix{Float64}(undef, size(samples.assignments)...)
    for j in 1:size(samples.assignments, 2)
        postmeans_mat[:,j] .= getproperty.(samples.components[j][samples.assignments[:,j]], :param)
    end
    mean(postmeans_mat, dims=2) |> vec
end 

assignments(gf::NealAlgorithmSamples) = gf.assignments


function StatsBase.fit!(gc::AbstractNealAlgorithm, progress=true;
              samples=5000, burnin=samples÷10)

    assignments = Matrix{Int}(undef, length(gc.data), samples)
    components = Vector{typeof(gc.components)}()
    log_αs = Vector{Float64}(undef, samples)

    @showprogress (progress ? 1 : Inf) "Gibbs sampling burnin" for _ in 1:burnin
        sample!(gc)
    end

    @showprogress (progress ? 1 : Inf) "Gibbs sampling" for i in 1:samples
        sample!(gc)
        assignments[:,i] .= gc.assignments
        push!(components, copy(gc.components))
        log_αs[i] = gc.logα
    end
    NealAlgorithmSamples(gc, assignments, components, log_αs)
end



function sample_component_param!(gc, i)
    post = Empirikos.posterior(gc.components[i].sample, gc.prior)
    old_comp = gc.components[i]
    new_comp = @set old_comp.param = rand(post)
    gc.components[i] = new_comp
end

function sample_α!(gc)
    # sample α
    n = length(gc.data)
    k = length(gc.components) - length(gc.empties)
    α = exp(gc.logα)
    a = shape(gc.α_dist)
    b = scale(gc.α_dist)
    η = rand(Beta(α+1, n ))

    b_α_star = 1/(1/b - log(η))
    p_a = (a + k - 1) / (a + k -1 + n / b_α_star)
    α = rand(MixtureModel([Gamma(a+k, b_α_star); Gamma(a+k-1, b_α_star)], [p_a, 1-p_a]))

    gc.logα = log(α)
end


function StatsBase.sample!(gc::NealAlgorithm2, i::Int)
    prior = gc.prior
    x = gc.data[i]
    old_comp = sub(gc.components[gc.assignments[i]], x)
    gc.components[gc.assignments[i]] = old_comp

    # recycle if empty
    isempty(old_comp) && push!(gc.empties, gc.assignments[i])

    log_probs = 
        [isempty(comp) ?
           -Inf :
           loglikelihood(x, comp.param) + log(comp.n)
         for comp
         in gc.components]

    log_probs[gc.empties[1]] = logpdf(prior, x) + gc.logα

    new_k = sample(Weights(exp.(log_probs)))
    gc.components[new_k] = add(gc.components[new_k], x)
    gc.assignments[i] = new_k

    # clean up empties
    if new_k == gc.empties[1]
        popfirst!(gc.empties)
        if isempty(gc.empties)
            empty_Ss = empty(gc.components[1])
            push!(gc.components, empty_Ss)
            push!(gc.empties, length(gc.components))
        end
    end

    return gc
end


#---------------------------------------
# Algorithm 8
#---------------------------------------

mutable struct NealAlgorithm8{D, T, S, W <: AbstractWrappedEBSample{S},F}  <: AbstractNealAlgorithm
    prior::D
    α_dist::T
    logα::Float64
    components::Vector{W}
    empties::Vector{Int}
    assignments::Vector{Int}    
    data::Vector{S}
    m::Int64 
    param_cache::Vector{F}
end

track_parameters(::NealAlgorithm8) = true 


function NealAlgorithm8(
    Ss::AbstractVector;
    m = 10,
    α_dist =  Gamma(0.001,100.0),
    prior = _default_prior(Ss)
    ) 
    all_Ss = wrap(Ss)
    empty_Ss = empty(all_Ss)
    NealAlgorithm8(prior,
             α_dist,
             1.0,
             [all_Ss, empty_Ss],
             [2],
             ones(Int, length(Ss)),
             Ss,
             m, 
             [1.0 for _ in 1:m]
             )
end


function StatsBase.sample!(gc::NealAlgorithm8, i::Int)
    m = gc.m
    logm = log(m)
    prior = gc.prior
    x = gc.data[i]
    old_comp = sub(gc.components[gc.assignments[i]], x)
    gc.components[gc.assignments[i]] = old_comp
  
    # following means that we are in case 2 of Algorithm 8
    if isempty(old_comp) 
        gc.param_cache[1] = old_comp.param
        push!(gc.empties, gc.assignments[i])
    else 
        gc.param_cache[1] = rand(prior)
    end 
    gc.param_cache[2:end] = rand(prior, m-1)

    log_probs = vcat(
        loglikelihood.(Ref(x), gc.param_cache) .+ gc.logα .- logm,
        [isempty(comp) ?
           -Inf :
           loglikelihood(x, comp.param) + log(comp.n)
         for comp
         in gc.components
         ]
     )

    sample_k = sample(Weights(exp.(log_probs)))

    # create new component
    if sample_k <= m
        new_k = gc.empties[1]
        new_comp = WrappedEBSample(x, 1, gc.param_cache[sample_k])      
    else 
        new_k = sample_k - m
        new_comp = add(gc.components[new_k], x)
    end 

    gc.components[new_k] = new_comp
    gc.assignments[i] = new_k


    # clean up empties
    if new_k == gc.empties[1]
        popfirst!(gc.empties)
        if isempty(gc.empties)
            empty_Ss = empty(gc.components[1])
            push!(gc.components, empty_Ss)
            push!(gc.empties, length(gc.components))
        end
    end

    return gc
end

