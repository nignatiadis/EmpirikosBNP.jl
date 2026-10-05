@testset "Quantile split search matches Base ordering" begin
    E = EmpirikosBNP
    rng = MersenneTwister(96300)
    for base in (Normal(), Normal(1, 2), TDist(8),
        Empirikos.fold(Normal()), Empirikos.fold(TDist(8) / std(TDist(8))))
        prior = PolyaTreeDistribution(; base, J = 8, median_centered = false)
        for j in 1:prior.J
            splits = prior.stored_splits[j]
            @test all(!isnan, splits)
            points = vcat(randn(rng, 30), splits, prevfloat.(splits),
                nextfloat.(splits), [-Inf, -0.0, 0.0, Inf, NaN, -NaN])
            for T in (Float32, Float64, BigFloat), x in T.(points)
                @test E.kfun(prior, x, j) == searchsortedlast(splits, x) + 1
            end
            @test (@inferred E.kfun(prior, 0.1, j)) isa Int
        end
    end
end

@testset "Finest-level split lookup matches searchsortedlast" begin
    E = EmpirikosBNP
    rng = MersenneTwister(96301)
    for base in (Normal(), Normal(1, 2), TDist(8),
        Empirikos.fold(Normal()), Empirikos.fold(TDist(8) / std(TDist(8)))), J in (1, 2, 8, 10)
        prior = PolyaTreeDistribution(; base, J, median_centered = false)
        splits = prior.stored_splits[J]
        @test (prior.split_lookup isa E.SplitLookup) == (J > 1)
        points = vcat(randn(rng, 10^5) .* 3, rand(rng, 10^4) .* 1e-3,
            splits, prevfloat.(splits), nextfloat.(splits),
            [-floatmax(), floatmax(), -Inf, Inf, -nextfloat(0.0), nextfloat(0.0)])
        @test all(x -> E.kfun(prior, x, J) == searchsortedlast(splits, x) + 1, points)
    end
end
