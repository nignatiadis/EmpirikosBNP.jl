abstract type AbstractMHProposal end 


Base.@kwdef struct AverageImputationProposal{T}  <: AbstractMHProposal
    mh_steps::Int = 3 
    base::T = Normal() 
end

Base.@kwdef struct VarianceProposal{D, T}  <: AbstractMHProposal
    mh_steps::Int = 3 
    default_dist::D 
    dof::T = Inf
end

mutable struct VariancePolyaSampler{D,T, R, I, V, S}
    σ²_prior::D
    base_polya::T
    imputation_mh::I
    variance_mh::V
    realized_pt::R
    σ²::Float64
    data::Vector{S} # config_samples 
end

function VariancePolyaSampler(data; base_polya)
    # assuming data is a vector of ConfigurationSample
    Ss = var.(EmpirikosBNP.iid_samples.(data))
    σ²_prior = EmpirikosBNP.quantiles_to_invχ²(extrema(Ss)...)
    realized_pt = rand(base_polya)
    realized_pt = realized_pt / std(realized_pt)

    
    VariancePolyaSampler(
        σ²_prior,
        base_polya,
        AverageImputationProposal(),
        VarianceProposal(3, σ²_prior, Inf),
        realized_pt,
        σ²_prior,
        data
    )
end


function proposal(prop::AverageImputationProposal, config_sample)
    K = nobs(config_sample)
    d_base = isinf(prop.dof) ? Normal() : TDis 

    proposal_d = Normal() * sqrt(var_pt) / sqrt(K) 
end



function mh_step(pt, config_sample, curr_z; steps=5)


    for _ in Base.OneTo(steps)
        prop_z = rand(proposal_d)
        prop_ratio = (conditional_pdf(pt, config_sample, prop_z) *  pdf(proposal_d, curr_z)) /
                 (conditional_pdf(pt, config_sample, curr_z) * pdf(proposal_d, prop_z))
        if rand() < prop_ratio
            curr_z = prop_z
        end
    end  
    curr_z

    logα = logdensity(model, candidate) - logdensity(model, transition_prev) +
        logratio_proposal_density(sampler, transition_prev, candidate)

    # Decide whether to return the previous params or the new one.
    transition = if -Random.randexp(rng) < logα
    end 
end










function StatsBase.sample!(vp::VariancePolyaSampler)
    sample_posterior_polya_tree!(vp) # normalizes to variance 1?
    sample_variance!(vp)
    impute_zbar!(vp)
end


function sample_posterior_polya_tree!(vp::VariancePolyaSampler)
   # TODO: rescale suitably
   zero_offsets!(vp.base_polya)
   posterior!(vp.data, vp.base_polya)
   realized_pt = rand(vp.base_polya)
   vp.realized_pt = realized_pt / std(realized_pt)
end

#function StatsBase.fit!(gc::AbstractNealAlgorithm, progress=true;
#              samples=5000, burnin=samples÷10)

#    assignments = Matrix{Int}(undef, length(gc.data), samples)
#    components = Vector{typeof(gc.components)}()
#    log_αs = Vector{Float64}(undef, samples)

#    @showprogress (progress ? 1 : Inf) "Gibbs sampling burnin" for _ in 1:burnin
#        sample!(gc)
#    end

#    @showprogress (progress ? 1 : Inf) "Gibbs sampling" for i in 1:samples
#        sample!(gc)
#        assignments[:,i] .= gc.assignments
#        push!(components, copy(gc.components))
#        log_αs[i] = gc.logα
##    end
#    NealAlgorithmSamples(gc, assignments, components, log_αs)
#end

#mutable struct VariancePolyaSamples
#    σ²::Vector{Float64}
#    base_polya_σ²::Vector{Float64}
#    Z̄_mat::Matrix{Float64}
#end