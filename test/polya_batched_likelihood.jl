@testset "Batched configuration likelihoods match one scale at a time" begin
    E = EmpirikosBNP
    rng = MersenneTwister(96400)
    for (base, symmetrized) in (
        (Empirikos.fold(TDist(8) / std(TDist(8))), true),
        (Empirikos.fold(Normal()), true),
        (TDist(8), false),
        (TDist(8) / std(TDist(8)), false),
    )
        tree = rand(rng, PolyaTreeDistribution(; base, symmetrized,
            J = 8, median_centered = !symmetrized))
        buffers = E.ScaledLogpdfBuffers()
        for d in (tree / std(tree), 0.7 + 1.2 * tree), K in (1, 5, 42, 3, 42)
            config = E.ConfigurationSample(configuration = randn(rng, 12),
                S² = ScaledChiSquareSample(1.0, 11), Z̄ = randn(rng))
            # Reuse some scales between calls, as the assignment sweep does,
            # and include scales whose Student-t product overflows.
            σ²s = vcat(rand(rng, K) .* 4, 1.0, 1e-300)
            reference = [logpdf(sqrt(σ²) * d, config) for σ² in σ²s]
            for _ in 1:2
                @test E._scaled_logpdfs!(similar(σ²s), d, config, σ²s, buffers) == reference
            end
            config.Z̄ = NaN
            @test isequal(E._scaled_logpdfs!(similar(σ²s), d, config, σ²s, buffers),
                [logpdf(sqrt(σ²) * d, config) for σ² in σ²s])
        end
    end
end
