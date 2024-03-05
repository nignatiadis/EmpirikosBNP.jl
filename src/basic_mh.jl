abstract type AbstractMHProposal end 


Base.@kwdef struct AverageImputationProposal{T}  <: AbstractMHProposal
    mh_steps::Int = 3 
    base::T = Normal() 
    var::Float64 = 1.0
end

function conditional_pdf(d::Distribution, config_sample::ConfigurationSample, z̄)
    pdf(d, IIDSample(config_sample, z̄)) 
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








mutable struct VariancePolyaSampler{D,T, R, I, V, S}
    σ²_prior::D
    base_polya::T
    imputation_mh::I
    variance_mh::V
    realized_pt::R
    σ²::Float64
    data::Vector{S} # config_samples 
end

function StatsBase.sample!(vp::VariancePolyaSampler)
    sample_posterior_polya_tree!(vp) # normalizes to variance 1?
    sample_variance!(vp)
    impute_zbar!(vp)
end

function sample_posterior_polya_tree!(vp::VariancePolyaSampler)
   zero_offsets!(vp.base_polya)
   posterior!(vp.data, vp.base_polya)
   realized_pt = rand(vp.base_polya)
   vp.realized_pt = realized_pt / std(realized_pt)
end

#mutable struct VariancePolyaSamples
#    σ²::Vector{Float64}
#    base_polya_σ²::Vector{Float64}
#    Z̄_mat::Matrix{Float64}
#end