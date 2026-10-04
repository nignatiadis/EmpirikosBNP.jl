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
