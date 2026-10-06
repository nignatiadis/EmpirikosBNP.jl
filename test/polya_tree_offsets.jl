@testset "Finest-bin tree offsets match per-response increments" begin
    E = EmpirikosBNP
    rng = MersenneTwister(96401)
    for (base, symmetrized) in ((Empirikos.fold(TDist(8) / std(TDist(8))), true),
        (Normal(), false)), J in (0, 1, 3, 8)
        prior = PolyaTreeDistribution(; base, symmetrized, J, median_centered = false)
        samples = [E.ConfigurationSample(configuration = randn(rng, 12),
            S² = ScaledChiSquareSample(1.0, 11), Z̄ = randn(rng)) for _ in 1:200]
        σs = rand(rng, 200) .+ 0.5
        reference = deepcopy(prior)
        E.zero_offsets!(reference)
        foreach((sample, σ) -> E.posterior!(sample, reference, σ), samples, σs)
        model = deepcopy(prior)
        foreach(offset -> fill!(offset, 7), model.offsets)
        E._posterior_offsets!(model, samples, σs)
        @test model.offsets == reference.offsets
    end
end
