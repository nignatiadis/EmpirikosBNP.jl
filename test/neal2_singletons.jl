include("neal2_partition_reference.jl")

@testset "Singleton variance posterior parameterization" begin
    prior = Empirikos.InverseScaledChiSquare(1.7, 5.0)
    @test params(InverseGamma(prior)) == (2.5, 4.25)
    for d in (1.0, 4.0, 11.0), s2 in (0.2, 3.0)
        post = Empirikos.posterior(ScaledChiSquareSample(s2, d), prior)
        @test post.ν == 5.0 + d
        @test post.σ² ≈ (5.0 * 1.7 + d * s2) / (5.0 + d)
        @test shape(InverseGamma(post)) == (5.0 + d) / 2
        @test scale(InverseGamma(post)) ≈ (5.0 * 1.7 + d * s2) / 2
    end
end

@testset "Immediate singleton draws, including recycled slots" begin
    E = EmpirikosBNP
    data = ScaledChiSquareSample.([0.25, 1.0, 4.0], 4)
    normals = NormalSample.([-0.5, 0.25, 1.0], 1.0)
    samplers = (
        E.NealAlgorithm2(data; prior = Empirikos.InverseScaledChiSquare(1.0, 5.0)),
        E.NealAlgorithm2(normals; prior = Normal(0, 2)),
        E.NealAlgorithm2DPGM(normals; prior = Normal(0, 2), Ap1_dist = Gamma(2, 1)),
        E.NealAlgorithm2GaussianConvolution(normals; prior = Normal(0, 2),
            A_dist = Uniform(0, 2), A_init = 0.5),
    )
    for gc in samplers
        gc.logα = 1000.0
        opened = Int[]
        for (step, i) in enumerate((1, 2, 1, 2))
            k = first(gc.empties)
            old = gc.components[k]
            gc.components[k] = E.WrappedEBSample(old.sample, 0, -12345.0)
            x = gc isa E.NealAlgorithm2GaussianConvolution ?
                E.inflate_sample(gc.data[i], gc.A) : gc.data[i]
            post = Empirikos.posterior(x, gc.prior)
            forced_weights = zeros(length(gc.components))
            forced_weights[k] = 1.0
            Random.seed!(810 + step)
            sample(Weights(forced_weights))
            expected = rand(post)
            next_random = rand()
            Random.seed!(810 + step)
            sample!(gc, i)
            @test gc.assignments[i] == k
            @test gc.components[k].n == 1
            @test gc.components[k].param ≈ expected
            @test rand() == next_random
            @test all(c.n == count(==(j), gc.assignments) for (j, c) in enumerate(gc.components))
            push!(opened, k)
        end
        @test opened == [2, 3, 4, 2]
    end
end

@testset "Next assignment uses refreshed singleton" begin
    E = EmpirikosBNP
    data = ScaledChiSquareSample.([0.25, 1.0, 4.0], 4)
    gc = E.NealAlgorithm2(data; prior = Empirikos.InverseScaledChiSquare(1.0, 5.0))
    gc.logα = 1000.0
    Random.seed!(991)
    sample!(gc, 1)
    singleton_param = gc.components[2].param
    gc.logα = 0.0
    x = data[2]
    # Removing observation 2 leaves one member in each occupied cluster.
    logs = [loglikelihood(x, gc.components[1].param),
        loglikelihood(x, singleton_param), logpdf(gc.prior, x)]
    selected = Int[]
    for seed in 1:100
        Random.seed!(seed)
        expected = sample(E._weights_from_logprobs(logs))
        copy_gc = deepcopy(gc)
        Random.seed!(seed)
        sample!(copy_gc, 2)
        @test copy_gc.assignments[2] == expected
        if expected == 2
            @test copy_gc.components[2].param == singleton_param
        end
        push!(selected, expected)
    end
    @test 2 in selected
end

@testset "Algorithm 2 preserves enumerated partition posterior" begin
    # Equal residual degrees of freedom isolate this correction from merging.
    data = ScaledChiSquareSample.([0.15, 0.3, 2.0, 5.0], 4)
    prior = Empirikos.InverseScaledChiSquare(1.0, 4.0)
    result = Neal2PartitionReference.invariance_check(data, prior, 1.2)
    @test length(result.states) == 15
    for (p, observed) in zip(result.probs, result.observed)
        # Independent posterior starts give binomial rather than MCMC error.
        @test abs(observed - p) <= 7 * sqrt(p * (1 - p) / result.draws) + 1 / result.draws
    end
end
