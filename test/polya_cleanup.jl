include("polya_cleanup_reference.jl")

@testset "Polya per-sweep cleanup" begin
    E = EmpirikosBNP
    for extra_slots in (0, 3, 25), m in (1, 10)
        Random.seed!(8400)
        data = [E.ConfigurationSample(E.IIDSample(s .* randn(6)))
            for s in (0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 3.0, 4.0)]
        neal2 = E.NealAlgorithm2(ScaledChiSquareSample.(data))
        base_polya = PolyaTreeDistribution(
            base = Empirikos.fold(TDist(8) / std(TDist(8))),
            J = 3, symmetrized = true, median_centered = false,
        )
        actual = E.NealAlgorithm8Polya(data; base_polya, neal_cp = neal2, m)
        for _ in 1:extra_slots
            push!(actual.components, empty(first(actual.components)))
            push!(actual.empties, length(actual.components))
        end
        @test E._should_cleanup(actual)
        expected = deepcopy(actual)
        for step in 1:10
            Random.seed!(8500 + step)
            PolyaCleanupReference.sweep!(expected)
            assigned_components = expected.components[expected.assignments]
            E.cleanup_components!(expected)
            next_random = rand()

            Random.seed!(8500 + step)
            @test sample!(actual) === actual
            @test rand() == next_random
            @test actual.assignments == expected.assignments
            @test actual.components == expected.components
            @test actual.logα == expected.logα
            @test actual.empties == expected.empties == [length(actual.components)]
            @test count(isempty, actual.components) == 1
            @test actual.components[actual.assignments] == assigned_components
            @test all(c.n == count(==(k), actual.assignments)
                for (k, c) in enumerate(actual.components))
            @test getproperty.(actual.data, :Z̄) == getproperty.(expected.data, :Z̄)
            @test actual.param_cache == expected.param_cache
            @test actual.vp.σ² == expected.vp.σ²
            @test actual.vp.base_polya.offsets == expected.vp.base_polya.offsets
            @test logpdf.(Ref(actual.vp.realized_pt), [-2.0, 0.0, 1.0]) ==
                logpdf.(Ref(expected.vp.realized_pt), [-2.0, 0.0, 1.0])
        end
    end
end
