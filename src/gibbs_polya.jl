
mutable struct NealAlgorithm8Polya{D,T,S,W<:AbstractWrappedEBSample,F,V,K} <:
               AbstractNealAlgorithm
    prior::D
    α_dist::T
    logα::Float64
    components::Vector{W}
    empties::Vector{Int}
    assignments::Vector{Int}
    data::Vector{S}
    m::Int64 # for MH refresh
    param_cache::Vector{F}
    vp::V
    scratch::K
end

function NealAlgorithm8Polya(
    config_samples::AbstractVector;
    base_polya,
    neal_cp::NealAlgorithm2,
    m = 10,
    ) 

    σ²_prior = neal_cp.prior
    vp = VariancePolyaSampler(neal_cp.data; base_polya, σ²_prior=σ²_prior)
    
    scratch = deepcopy(config_samples)

    NealAlgorithm8Polya(σ²_prior,
             neal_cp.α_dist,
             neal_cp.logα,
             neal_cp.components,
             neal_cp.empties,
             neal_cp.assignments,
             config_samples,
             m, 
             [1.0 for _ in 1:m],
             vp,
             scratch
            )
end


#function NealAlgorithm8Polya(
#    Ss::AbstractVector;
#    base_polya,
#    m = 10,
#    α_dist =  Gamma(0.001,100.0),
#    prior = _default_prior(Ss)
#    ) 
#    all_Ss = wrap(Ss)
#    empty_Ss = empty(all_Ss)
#    vp = VariancePolyaSampler(Ss; base_polya, σ²_prior=prior)

#    NealAlgorithm8Polya(prior,
#             α_dist,
#             1.0,
#             [all_Ss, empty_Ss],
#             [2],
#             ones(Int, length(Ss)),
#             Ss,
#             m, 
#             [1.0 for _ in 1:m],
#             vp
#             )
#end



function StatsBase.sample!(gc::NealAlgorithm8Polya, i::Int)
    m = gc.m
    logm = log(m)
    prior = gc.prior
    x = gc.data[i]
    old_comp = sub(gc.components[gc.assignments[i]], x.S²) # edit compared to Neal8
    gc.components[gc.assignments[i]] = old_comp

    # following means that we are in case 2 of Algorithm 8
    if isempty(old_comp)
        gc.param_cache[1] = old_comp.param
        push!(gc.empties, gc.assignments[i])
    else
        gc.param_cache[1] = rand(prior)
    end
    gc.param_cache[2:end] = rand(prior, m - 1)

    xlik = VarianceIIDSample(x, gc.vp.realized_pt)

    log_probs = vcat(
        loglikelihood.(Ref(xlik), gc.param_cache) .+ gc.logα .- logm,
        [
            isempty(comp) ? -Inf : loglikelihood(xlik, comp.param) + log(comp.n) for
            comp in gc.components
        ],
    )

    sample_k = sample(Weights(exp.(log_probs)))

    # create new component
    if sample_k <= m
        new_k = gc.empties[1]
        new_comp = WrappedEBSample(x.S², 1, gc.param_cache[sample_k])
    else
        new_k = sample_k - m
        new_comp = add(gc.components[new_k], x.S²)
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



function StatsBase.sample!(neal8polya::NealAlgorithm8Polya)
    vp = neal8polya.vp
    realized_pt = rand(vp.base_polya)
    vp.realized_pt = realized_pt / std(realized_pt)
    zero_offsets!(vp.base_polya)  # reset current posterior profile 


    for i = 1:length(neal8polya.data)
        sample!(neal8polya, i) # update cluster assignment
    end



    for (comp_idx, comp) in enumerate(neal8polya.components)
        if isempty(comp)
            continue
        end
        vp.realized_pt = vp.realized_pt / std(vp.realized_pt)
        # setup scratch for component

        empty!(neal8polya.scratch)
        for i in Base.OneTo(length(neal8polya.data))
            if neal8polya.assignments[i] == comp_idx
                push!(neal8polya.scratch, neal8polya.data[i])
            end
        end

        # three things to do
        # 1. update the variance for this component
        # 2. update the zbar for each data point
        # 3. start updating the posterior polya tree

        neal8polya.vp.σ² = comp.param
        variance_mh = neal8polya.vp.variance_mh
        variance_mh = @set variance_mh.default_dist =
            Empirikos.posterior(comp.sample, neal8polya.prior)
        neal8polya.vp.variance_mh = variance_mh
        σ² = sample_variance!(neal8polya.vp, neal8polya.scratch)
        σ = sqrt(σ²)
        σ_inv = 1.0 / σ
        comp = @set comp.param = σ²
        neal8polya.components[comp_idx] = comp

        vp.realized_pt = vp.realized_pt * σ
        for sample in neal8polya.scratch
            impute_zbar!(neal8polya.vp, sample)
            posterior!(sample, neal8polya.vp.base_polya, σ_inv)
        end
    end
    sample_α!(neal8polya)   # update concentration parameter α 

    vp.realized_pt = vp.realized_pt / std(vp.realized_pt)

    if length(neal8polya.empties) > 20
        cleanup_components!(neal8polya)
    end
    return neal8polya
end

function StatsBase.fit!(gc::NealAlgorithm8Polya, progress=true;
              samples=1000, burnin=samples÷10)

    assignments = Matrix{Int}(undef, length(gc.data), samples)
    components = Vector{typeof(gc.components)}()
    realized_pts = Vector{typeof(gc.vp.realized_pt)}()
    log_αs = Vector{Float64}(undef, samples)
    Zs_mat = Matrix{Float64}(undef, length(gc.data), samples) 

    @showprogress (progress ? 1 : Inf) "Gibbs sampling burnin" for _ in 1:burnin
        sample!(gc)
    end

    @showprogress (progress ? 1 : Inf) "Gibbs sampling" for i in 1:samples
        sample!(gc)
        assignments[:,i] .= gc.assignments
        push!(components, copy(gc.components))
        log_αs[i] = gc.logα
        Zs_mat[:,i] .= getproperty.(gc.data, :Z̄)
        push!(realized_pts, deepcopy(gc.vp.realized_pt))
    end
    (gc=gc, assignments=assignments, components=components, log_αs=log_αs,
    realized_pts=realized_pts, Zs_mat=Zs_mat
    )
end
