abstract type AbstractMHProposal end 


Base.@kwdef struct AverageImputationProposal{T}  <: AbstractMHProposal
    mh_steps::Int = 3 
    base::T = Normal() 
end

function proposal_dist(avg::AverageImputationProposal, config_sample, σ²)
    K = nobs(config_sample)
    avg.base * sqrt(σ²) / sqrt(K) 
end

Base.@kwdef struct VarianceProposal{D, T}  <: AbstractMHProposal
    mh_steps::Int = 3 
    default_dist::D 
    dof::T = 100
end

function proposal_dist(varprop::VarianceProposal)
    marginalize(ScaledChiSquareSample(nothing, varprop.dof), varprop.default_dist)
end



mutable struct VariancePolyaSampler{D, T, R, I, V, S, U}
    σ²_prior::D
    base_polya::T
    imputation_mh::I
    variance_mh::V
    realized_pt::R
    σ²::U # could be vector of σ^2
    data::Vector{S} # config_samples 
end

function VariancePolyaSampler(data; base_polya, σ²_prior=_default_prior(data))
    # assuming data is a vector of ConfigurationSample    
    realized_pt = rand(base_polya)
    realized_pt = realized_pt / std(realized_pt)

    Ss_merge = merge_samples(data)

    imputation_proposal = AverageImputationProposal(
        base = TDist(5) #/ std(TDist(5))
    )
    variance_proposal = VarianceProposal(
        default_dist = Empirikos.posterior(Ss_merge, σ²_prior),
        dof = 5,
    )

    VariancePolyaSampler(
        σ²_prior,
        base_polya,
        imputation_proposal,
        variance_proposal,
        realized_pt,
        response(Ss_merge),
        data
    )
end




function StatsBase.sample!(vp::VariancePolyaSampler)
    sample_posterior_polya_tree!(vp) # normalizes to variance 1?
    sample_variance!(vp)
    vp.realized_pt = vp.realized_pt * sqrt(vp.σ²)
    for z in vp.data
        impute_zbar!(vp, z)
    end
end


function sample_posterior_polya_tree!(vp::VariancePolyaSampler, σ² = vp.σ²)
   σinv = 1 ./ sqrt.(σ²)
   zero_offsets!(vp.base_polya)
   # revisit the following line ..
   posterior!(vp.data, vp.base_polya, σinv)
   realized_pt = rand(vp.base_polya)
   vp.realized_pt = realized_pt / std(realized_pt)
end

function sample_variance!(vp, data=vp.data)
    transition_prev = vp.σ²
    proposal_d = proposal_dist(vp.variance_mh)
    steps = vp.variance_mh.mh_steps

    var_iid = VarianceIIDSample(IIDSample(data), vp.realized_pt)

    for _ in Base.OneTo(steps)
        candidate = rand(proposal_d)

        logα = logpdf(proposal_d, transition_prev) -  
               logpdf(proposal_d, candidate) +
               loglikelihood(var_iid, candidate) -
               loglikelihood(var_iid, transition_prev) + 
               logpdf(vp.σ²_prior, candidate) -
               logpdf(vp.σ²_prior, transition_prev)
        
        if -Random.randexp() < logα
            transition_prev = candidate 
        end 
    end 
    vp.σ² = transition_prev
    transition_prev
end

function impute_zbar!(vp, config_sample::ConfigurationSample)
    transition_prev = config_sample.Z̄
    realized_pt = vp.realized_pt

    proposal_d = proposal_dist(vp.imputation_mh, config_sample, vp.σ²) 
    steps = vp.imputation_mh.mh_steps

    #var_iid = VarianceIIDSample(IIDSample(vp.data), vp.realized_pt)

    for _ in Base.OneTo(steps)
        candidate = rand(proposal_d)

        logα = logpdf(proposal_d, transition_prev) -  
               logpdf(proposal_d, candidate) +
               logpdf(realized_pt, config_sample, candidate) -
               logpdf(realized_pt, config_sample, transition_prev)

        if -Random.randexp() < logα
            transition_prev = candidate 
        end 
    end 
    config_sample.Z̄ = transition_prev
    transition_prev
end 

mutable struct VariancePolyaSamples
    σ²::Vector{Float64}
    Z̄_mat::Matrix{Float64}
end

function StatsBase.fit!(vp::VariancePolyaSampler, progress=true;
            samples=5000, burnin=samples÷10)

    Z̄_mat = Matrix{Float64}(undef, length(vp.data), samples)
    σ²_vec = Vector{Float64}(undef, samples)

    @showprogress (progress ? 1 : Inf) "Gibbs sampling burnin" for _ in 1:burnin
        sample!(vp)
    end

    @showprogress (progress ? 1 : Inf) "Gibbs sampling" for i in 1:samples
        sample!(vp)
        Z̄_mat[:,i] .= getproperty.(vp.data, :Z̄)
        σ²_vec[i] = vp.σ²
    end
    VariancePolyaSamples(σ²_vec, Z̄_mat)
end

