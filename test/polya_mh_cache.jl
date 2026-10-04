module PolyaMHReference
using EmpirikosBNP, Distributions, Random
const E = EmpirikosBNP

function variance!(vp, data)
    current = vp.σ²
    proposal = E.proposal_dist(vp.variance_mh)
    sample = E.VarianceIIDSample(E.IIDSample(data), vp.realized_pt)
    for _ in Base.OneTo(vp.variance_mh.mh_steps)
        candidate = rand(proposal)
        log_acceptance = logpdf(proposal, current) - logpdf(proposal, candidate) +
            loglikelihood(sample, candidate) - loglikelihood(sample, current) +
            logpdf(vp.σ²_prior, candidate) - logpdf(vp.σ²_prior, current)
        if -randexp() < log_acceptance
            current = candidate
        end
    end
    vp.σ² = current
    current
end

function impute!(vp, sample)
    current = sample.Z̄
    proposal = E.proposal_dist(vp.imputation_mh, sample, vp.σ²)
    for _ in Base.OneTo(vp.imputation_mh.mh_steps)
        candidate = rand(proposal)
        log_acceptance = logpdf(proposal, current) - logpdf(proposal, candidate) +
            logpdf(vp.realized_pt, sample, candidate) - logpdf(vp.realized_pt, sample, current)
        if -randexp() < log_acceptance
            current = candidate
        end
    end
    sample.Z̄ = current
    current
end
end

@testset "MH current likelihood reuse" begin
    E = EmpirikosBNP
    for steps in (0, 1, 3, 10), symmetrized in (false, true), seed in 1:10
        Random.seed!(96000 + seed)
        data = [E.ConfigurationSample(E.IIDSample(s * randn(5))) for s in (0.5, 1.0, 3.0)]
        base = symmetrized ? Empirikos.fold(TDist(8)) : TDist(8)
        prior = PolyaTreeDistribution(; base, J = 3, symmetrized,
            median_centered = !symmetrized)
        gc = E.NealAlgorithm8Polya(data; base_polya = prior,
            neal_cp = E.NealAlgorithm2(ScaledChiSquareSample.(data)))
        gc = EmpirikosBNP.@set gc.vp.variance_mh.mh_steps = steps
        gc = EmpirikosBNP.@set gc.vp.imputation_mh.mh_steps = steps
        actual = gc.vp
        expected = deepcopy(actual)
        Random.seed!(96100 + seed)
        reference = PolyaMHReference.variance!(expected, data)
        next_random = rand(10)
        Random.seed!(96100 + seed)
        @test E.sample_variance!(actual, data) == reference
        @test actual.σ² == expected.σ²
        @test rand(10) == next_random
        actual.realized_pt *= sqrt(actual.σ²)
        expected.realized_pt *= sqrt(expected.σ²)
        for sample in data
            expected_sample = deepcopy(sample)
            Random.seed!(96200 + seed)
            reference = PolyaMHReference.impute!(expected, expected_sample)
            next_random = rand(10)
            Random.seed!(96200 + seed)
            @test E.impute_zbar!(actual, sample) == reference
            @test sample.Z̄ == expected_sample.Z̄
            @test rand(10) == next_random
        end
    end
end
