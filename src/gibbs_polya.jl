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

mutable struct NealAlgorithm8PolyaSamples{N, A, S, P, Z}
    gc::N
    assignments::A
    components::S
    log_αs::Vector{Float64}
    realized_pts::P
    Zs_mat::Matrix{Z}
end

function StatsBase.fit!(gc::NealAlgorithm8Polya, progress=true;
              samples=1000, burnin=samples÷10, lightweight=false)

    if lightweight
        assignments = nothing
        components = nothing
        realized_pts = nothing
    else 
        assignments = Matrix{Int}(undef, length(gc.data), samples)
        components = Vector{typeof(gc.components)}()
        realized_pts = Vector{typeof(gc.vp.realized_pt)}()
    end 

    log_αs = Vector{Float64}(undef, samples)
    Zs_mat = Matrix{Float64}(undef, length(gc.data), samples) 

    @showprogress (progress ? 1 : Inf) "Gibbs sampling burnin" for _ in 1:burnin
        sample!(gc)
    end

    @showprogress (progress ? 1 : Inf) "Gibbs sampling" for i in 1:samples
        sample!(gc)
        log_αs[i] = gc.logα
        Zs_mat[:,i] .= getproperty.(gc.data, :Z̄)
        if !lightweight
            assignments[:,i] .= gc.assignments
            push!(components, copy(gc.components))
            push!(realized_pts, deepcopy(gc.vp.realized_pt))
        end
    end
    
    NealAlgorithm8PolyaSamples(gc, assignments, components, log_αs, realized_pts, Zs_mat)
end






# pvalue function

"""
   _pval_fun(samples::NealAlgorithm8PolyaSamples, mu_hats; 
             method=:adaptive, refinement_threshold=0.015, rtol=0.01, show_progress=true)

Compute p-values for the Polya tree model.

# Arguments
- `samples`: The samples from the Polya tree MCMC
- `mu_hats`: The observed values to compute p-values for  
- `method`: :monte_carlo, :integration, or :adaptive (default)
- `refinement_threshold`: threshold below which to refine p-values (for adaptive method)
- `rtol`: relative tolerance for quadrature integration
- `show_progress`: whether to show progress bars
"""
function _pval_fun(samples::NealAlgorithm8PolyaSamples, mu_hats; 
                  method=:adaptive, refinement_threshold=0.015, rtol=0.01, show_progress=true)
    
    
    if method == :monte_carlo
        # Direct Monte Carlo using Z values
        n_samples = size(samples.Zs_mat, 2)
        return vec((1 .+ sum(abs.(samples.Zs_mat) .>= abs.(mu_hats), dims=2)) ./ (n_samples + 1))
    
    elseif method == :integration
        # Full integration for all p-values
        return _compute_pvals_integration(samples, mu_hats, 1:length(mu_hats); rtol, show_progress)
    
    elseif method == :adaptive
        # First get Monte Carlo p-values
        pvals_mc = _pval_fun(samples, mu_hats; method=:monte_carlo)
        
        # Find which need refinement
        needs_refinement = pvals_mc .<= refinement_threshold
        refine_indices = findall(needs_refinement)
        
        if isempty(refine_indices)
            return pvals_mc
        end
        
        # Refine small p-values
        pvals_refined = _compute_pvals_integration(samples, mu_hats, refine_indices; rtol, show_progress)
        
        # Combine results
        pvals = copy(pvals_mc)
        pvals[refine_indices] = pvals_refined
        return pvals
    
    else
        error("Unknown method: $method. Use :monte_carlo, :integration, or :adaptive")
    end
end

function _pval_custom(config_sample, z, σ, realized_pt; rtol=0.01)
    # Scale the Polya tree by σ
    scaled_pt = realized_pt * σ  #/ std(realized_pt)=1
    
    # Define density function
    t_density(t) = exp(ULogarithmic, logpdf(scaled_pt, config_sample, t))
    
    # Normalize
    norm_const = QuadGK.quadgk(t_density, -Inf, Inf; rtol=rtol)[1]
    
    # Normalized density
    normalized_density(t) = t_density(t) / norm_const
    
    # Two-tailed p-value
    p_upper = QuadGK.quadgk(normalized_density, abs(z), Inf; rtol=rtol)[1]
    p_lower = QuadGK.quadgk(normalized_density, -Inf, -abs(z); rtol=rtol)[1]
    
    min(Float64(p_upper + p_lower), 1.0)
end

function _compute_pvals_integration(samples, mu_hats, indices; 
                                  rtol=0.01, show_progress=true)
    n_indices = length(indices)
    n_samples = length(samples.components)
    pvals = zeros(n_indices)
    
    # Get config_samples from the stored gc.data
    config_samples = samples.gc.data
    
    # Progress bar setup
    desc = n_indices == length(mu_hats) ? "Computing all p-values" : "Refining $(n_indices) p-values"
    progress = Progress(n_indices; enabled=show_progress, desc=desc)
    
    pval_vec = ones(n_samples)

    for (idx, i) in enumerate(indices)        
        # Average over MCMC samples
        for j in 1:n_samples
            # Extract σ
            σ = sqrt(samples.components[j][samples.assignments[i,j]].param)
            
            # Compute p-value
            pval_vec[j] = _pval_custom(config_samples[i], mu_hats[i], σ, samples.realized_pts[j]; rtol)
        end
        
        pvals[idx] = mean(pval_vec)
        fill!(pval_vec, 1.0)
        next!(progress)
    end
    
    return pvals
end

function _merge_samples(samples::NealAlgorithm8PolyaSamples...)
    if isempty(samples)
        throw(ArgumentError("Cannot merge empty list of samples"))
    end
    
    if length(samples) == 1
        return samples[1]
    end
    
    # Use the first sample's gc (they should all be the same structure)
    merged_gc = samples[1].gc
    
    # Check if any of the samples were created with lightweight=true
    has_lightweight = any(s -> s.assignments === nothing, samples)
    
    if has_lightweight
        # If any sample is lightweight, merge only the common fields
        merged_assignments = nothing
        merged_components = nothing
        merged_realized_pts = nothing
    else
        # Merge assignments matrices (concatenate along sample dimension)
        merged_assignments = hcat((s.assignments for s in samples)...)
        
        # Merge components vectors
        merged_components = vcat((s.components for s in samples)...)
        
        # Merge realized_pts vectors  
        merged_realized_pts = vcat((s.realized_pts for s in samples)...)
    end
    
    # Merge log_αs vectors
    merged_log_αs = vcat((s.log_αs for s in samples)...)
    
    # Merge Zs_mat matrices (concatenate along sample dimension)
    merged_Zs_mat = hcat((s.Zs_mat for s in samples)...)
    
    return NealAlgorithm8PolyaSamples(
        merged_gc,
        merged_assignments,
        merged_components, 
        merged_log_αs,
        merged_realized_pts,
        merged_Zs_mat
    )
end