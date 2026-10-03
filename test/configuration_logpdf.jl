@testset "Polya configuration likelihood without temporary arrays" begin
    rng = MersenneTwister(912)
    for (base, symmetrized) in (
        (TDist(8), false),
        (Empirikos.fold(Normal()), true),
        (Empirikos.fold(TDist(8) / std(TDist(8))), true),
    )
        tree = rand(rng, PolyaTreeDistribution(; base, symmetrized,
            J = 5, median_centered = false))
        for d in (tree, tree / std(tree), 0.7 + 1.2 * tree, -2.0 * tree)
            for n in (0, 1, 3, 8, 31, 129, 1024)
                values = randn(rng, n)
                config = EmpirikosBNP.ConfigurationSample(
                    configuration = values,
                    S² = ScaledChiSquareSample(1.0, max(n - 1, 1)),
                    Z̄ = 0.3,
                )
                # Preserve the old broadcast-and-sum path as the reference.
                reference(d, zbar) = sum(logpdf.(Ref(d), values .+ zbar))
                for zbar in (-1.1, 0.0, 0.3, 2.0)
                    @test logpdf(d, config, zbar) ≈ reference(d, zbar) atol = 1e-10
                end
                @test logpdf(d, config) ≈ reference(d, config.Z̄) atol = 1e-10
                @test (@inferred logpdf(d, config)) isa Float64
                for variance in (0.01, 0.7, 2.0, 100.0)
                    sample = EmpirikosBNP.VarianceIIDSample(config, d)
                    @test loglikelihood(sample, variance) ≈
                        reference(sqrt(variance) * d, config.Z̄) atol = 1e-10
                end
            end

            configs = [EmpirikosBNP.ConfigurationSample(EmpirikosBNP.IIDSample(randn(rng, n)))
                for n in (3, 5, 8)]
            sample = EmpirikosBNP.VarianceIIDSample(EmpirikosBNP.IIDSample(configs), d)
            scaled = sqrt(1.7) * d
            expected = sum(sum(logpdf.(Ref(scaled), c.configuration .+ c.Z̄)) for c in configs)
            @test loglikelihood(sample, 1.7) ≈ expected atol = 1e-10
        end
    end

    config = EmpirikosBNP.ConfigurationSample(EmpirikosBNP.IIDSample([0.1, 0.7, -0.3]))
    for d in (Normal(), TDist(8), Exponential())
        @test logpdf(d, config, 1.0) == sum(logpdf.(Ref(d), config.configuration .+ 1.0))
    end
end
