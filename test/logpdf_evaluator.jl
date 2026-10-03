@testset "Prepared log densities" begin
    for T in (Float32, Float64, BigFloat)
        bases = (
            Normal(T(0), T(1)), Normal(T(1), T(2)), Normal(T(1), T(0)),
            TDist(T(0.5)), TDist(T(5)), TDist(T(8)), TDist(T(Inf)),
            T(1) + T(2) * TDist(T(8)), T(-2) * TDist(T(5)),
            Exponential(T(2)),
        )
        for base in bases
            evaluate = EmpirikosBNP._logpdf_evaluator(base)
            # Mixed observation types must retain Distributions' precision rules.
            for S in (Float32, Float64, BigFloat)
                for x in S.((-Inf, -1e20, -4, -0.0, 0, 0.3, 1, 4, 1e20, Inf, NaN))
                    expected = logpdf(base, x)
                    actual = evaluate(x)
                    @test typeof(actual) === typeof(expected)
                    @test isequal(actual, expected) || isapprox(actual, expected;
                        rtol = 16 * eps(float(one(expected))), atol = 0)
                end
            end
        end
    end

    for base in (Normal(), TDist(8), TDist(8) / std(TDist(8)), Exponential())
        evaluate = EmpirikosBNP._logpdf_evaluator(base)
        @test (@inferred evaluate(0.3)) ≈ logpdf(base, 0.3)
    end
end

@testset "Prepared folded densities" begin
    for base in (
        Empirikos.fold(Normal()), Empirikos.fold(Normal(0, 2)),
        Empirikos.fold(TDist(0.5)), Empirikos.fold(TDist(8)),
        Empirikos.fold(TDist(8) / std(TDist(8))),
        Empirikos.fold(2 * TDist(0.5)), Empirikos.fold(TDist(Inf)),
    )
        evaluate = EmpirikosBNP._symmetrized_base_logpdf_evaluator(base)
        for x in (0.0, 0.3, 2.0, 1e20, Inf)
            expected = logpdf(base, x) - log(2)
            actual = evaluate(x)
            @test isequal(actual, expected) || isapprox(actual, expected; atol = 1e-12)
        end
    end
end

@testset "Unsupported symmetrized bases" begin
    for base in (
        Normal(), TDist(8), Exponential(), Empirikos.fold(Logistic()),
        Empirikos.fold(Normal(1, 2)), Empirikos.fold(1 + 2 * TDist(8)),
    )
        @test_throws ArgumentError EmpirikosBNP._symmetrized_base_logpdf_evaluator(base)
        # Empty layers suffice to exercise construction without needing quantiles.
        prior = PolyaTreeDistribution(; base, symmetrized = true, J = 0,
            median_centered = false, offsets = Vector{Int}[], stored_splits = Vector{Float64}[])
        @test_throws ArgumentError EmpirikosBNP.PolyaTree(prior, Vector{Float64}[])
    end
end

@testset "Polya logpdf agrees with uncached density" begin
    rng = MersenneTwister(781)
    for (base, symmetrized) in (
        (Normal(), false), (Normal(1, 2), false), (TDist(8), false),
        (Exponential(), false), (Empirikos.fold(Normal()), true),
        (Empirikos.fold(Normal(0, 2)), true),
        (Empirikos.fold(TDist(8) / std(TDist(8))), true),
    )
        prior = PolyaTreeDistribution(; base, symmetrized, J = 4, median_centered = false)
        tree = rand(rng, prior)
        splits = vcat(prior.stored_splits...)
        points = vcat(randn(rng, 100), splits, prevfloat.(splits), nextfloat.(splits))
        symmetrized && append!(points, -points)
        for x in points
            y = symmetrized ? abs(x) : x
            expected = EmpirikosBNP._log_prob(tree, y) + prior.J * log(2) +
                (logpdf(base, y) - (symmetrized ? log(2) : 0))
            @test logpdf(tree, x) ≈ expected atol = 1e-12
        end
        @test (@inferred logpdf(tree, 0.3)) isa Real
    end
end
