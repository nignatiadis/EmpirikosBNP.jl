@testset "Reuse raw standard deviation while rescaling" begin
    rng = MersenneTwister(96400)
    for symmetrized in (false, true)
        base = symmetrized ? Empirikos.fold(TDist(8)) : TDist(8)
        prior = PolyaTreeDistribution(; base, J = 4, symmetrized,
            median_centered = !symmetrized)
        raw = rand(rng, prior)
        s = std(raw)
        current = raw / s
        for scale in (0.1, 3.0, 0.001, 100.0, 1.2)
            scaled = current * scale
            @test std(scaled) == abs(scaled.σ) * s
            expected = scaled / std(scaled)
            current = scaled / (abs(scaled.σ) * s)
            @test current.σ == expected.σ
            @test current.ρ === raw
            @test logpdf.(Ref(current), [-2.0, 0.0, 1.0]) ==
                logpdf.(Ref(expected), [-2.0, 0.0, 1.0])
        end
    end
end
