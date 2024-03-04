abstract type AbstractMHProposal end 





Base.@kwdef struct AverageImputationProposal{T}  <: AbstractMHProposal
    mh_steps::Int = 3 
    dof::T = Inf 
end

function conditional_pdf(d::Distribution, config_sample::ConfigurationSample2, z̄)
    pdf(d, IIDSample( (@set config_sample.Z̄ = z̄) ) )
end

function mh_step(pt, config_sample, curr_z; var_pt=1, steps=5)
    K = nobs(config_sample)
    proposal_d = Normal() * sqrt(var_pt) / sqrt(K) 

    for _ in Base.OneTo(steps)
        prop_z = rand(proposal_d)
        prop_ratio = (conditional_pdf(pt, config_sample, prop_z) *  pdf(proposal_d, curr_z)) /
                 (conditional_pdf(pt, config_sample, curr_z) * pdf(proposal_d, prop_z))
        if rand() < prop_ratio
            curr_z = prop_z
        end
    end  
    curr_z
end