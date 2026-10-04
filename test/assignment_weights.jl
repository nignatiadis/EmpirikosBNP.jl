@testset "Stable categorical log weights" begin
    normalized(log_probs) = let w = EmpirikosBNP._weights_from_logprobs(log_probs)
        collect(w) / sum(w)
    end

    for T in (Float32, Float64, BigFloat)
        log_probs = T[-2, -1, 0, -Inf]
        original = copy(log_probs)
        expected = exp.(log_probs) / sum(exp.(log_probs))
        for offset in T[0, -1000, 1000, -2^20, 2^20]
            @test normalized(log_probs .+ offset) ≈ expected
        end
        @test isequal(log_probs, original)
        @test normalized(T[-Inf, -10000, -Inf]) == T[0, 1, 0]
        @test normalized(T[10000, 10000]) == T[0.5, 0.5]
        for invalid in (T[], T[-Inf], T[-Inf, -Inf], T[NaN, 0],
            T[0, NaN], T[Inf, 0], T[0, Inf], T[Inf, Inf])
            @test_throws ArgumentError EmpirikosBNP._weights_from_logprobs(invalid)
        end
    end

    # High-precision reference exponentiates the unshifted values directly.
    setprecision(BigFloat, 256) do
        for log_probs in ([-10000.0, -10001.0, -10002.0, -Inf],
            [10000.0, 9999.0, 9998.0, -Inf], [-700.0, 0.0, -Inf])
            reference = exp.(BigFloat.(log_probs))
            reference ./= sum(reference)
            @test normalized(log_probs) ≈ Float64.(reference) rtol = 1e-14
            @test all(isfinite, normalized(log_probs))
            @test sum(normalized(log_probs)) ≈ 1
        end
    end
    @test normalized([floatmax(Float64), -floatmax(Float64)]) == [1.0, 0.0]
    @test normalized([-floatmax(Float64), -floatmax(Float64)]) == [0.5, 0.5]
    @test normalized([0.0, -1000.0]) == [1.0, 0.0]

    # For exactly representable shifts, the actual seeded categorical draws agree.
    log_probs = [-2.0, -1.0, 0.0, -Inf]
    w = EmpirikosBNP._weights_from_logprobs(log_probs)
    shifted = EmpirikosBNP._weights_from_logprobs(log_probs .+ 2.0^40)
    rng, shifted_rng = MersenneTwister(41), MersenneTwister(41)
    @test [sample(rng, w) for _ in 1:1000] ==
        [sample(shifted_rng, shifted) for _ in 1:1000]
    rng = MersenneTwister(42)
    w = EmpirikosBNP._weights_from_logprobs([-Inf, -10000.0, -Inf])
    @test all(==(2), [sample(rng, w) for _ in 1:1000])
end

function assignment_test_samplers()
    data = NormalSample.([-0.5, 0.2, 1.1], 1.0)
    prior = Normal(0, 2)
    configs = [EmpirikosBNP.ConfigurationSample(EmpirikosBNP.IIDSample(row))
        for row in ([-1.0, 0.0, 1.0], [-0.4, 0.1, 0.3], [-2.0, 0.0, 2.0])]
    neal2 = EmpirikosBNP.NealAlgorithm2(ScaledChiSquareSample.(configs);
        prior = Empirikos.InverseScaledChiSquare(1.0, 5.0))
    tree_prior = PolyaTreeDistribution(base = Empirikos.fold(Normal()),
        J = 3, symmetrized = true, median_centered = false)
    (
        EmpirikosBNP.NealAlgorithm2(data; prior),
        EmpirikosBNP.NealAlgorithm8(data; prior),
        EmpirikosBNP.NealAlgorithm2DPGM(data; prior, Ap1_dist = Gamma(2, 1)),
        EmpirikosBNP.NealAlgorithm2GaussianConvolution(data; prior,
            A_dist = Uniform(0, 2), A_init = 0.5),
        EmpirikosBNP.NealAlgorithm8Polya(configs; base_polya = tree_prior, neal_cp = neal2),
    )
end

function check_assignment_counts(gc)
    @test sum(c.n for c in gc.components) == length(gc.data)
    @test all(c.n == count(==(k), gc.assignments) for (k, c) in enumerate(gc.components))
    @test sort(gc.empties) == findall(isempty, gc.components)
    @test all(k -> 1 <= k <= length(gc.components), gc.assignments)
end

@testset "Stable weights at every assignment site" begin
    Random.seed!(712)
    for gc in assignment_test_samplers()
        # A valid but extreme concentration makes raw exponentiation overflow.
        gc.logα = 1000.0
        sample!(gc, 1)
        @test gc.assignments[1] != gc.assignments[2]
        check_assignment_counts(gc)
        gc.logα = 0.0
        for _ in 1:5, i in eachindex(gc.data)
            sample!(gc, i)
            check_assignment_counts(gc)
        end
    end
    for gc in assignment_test_samplers()
        gc.logα = NaN
        @test_throws ArgumentError sample!(gc, 1)
    end

    # All unshifted Normal-DP weights underflow, although their ratios are valid.
    data = NormalSample.([1000.0, 0.0], 1.0)
    gc = EmpirikosBNP.NealAlgorithm2(data; prior = Normal(0, 2))
    raw_logs = [loglikelihood(data[1], gc.components[1].param),
        logpdf(gc.prior, data[1]) + gc.logα]
    @test all(isfinite, raw_logs)
    @test all(iszero, exp.(raw_logs))
    sample!(gc, 1)
    @test gc.assignments[1] != gc.assignments[2]
    check_assignment_counts(gc)
end
