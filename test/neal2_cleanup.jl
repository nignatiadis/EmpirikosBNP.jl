# Reference sweep with the previous cleanup schedule.
function sweep_before_cleanup_change!(gc)
    for i in eachindex(gc.data)
        sample!(gc, i)
    end
    for k in eachindex(gc.components)
        isempty(gc.components[k]) || EmpirikosBNP.sample_component_param!(gc, k)
    end
    EmpirikosBNP.sample_α!(gc)
    length(gc.empties) > 100 && EmpirikosBNP.cleanup_components!(gc)
    gc
end

@testset "Algorithm 2 per-sweep cleanup" begin
    E = EmpirikosBNP
    fixtures = (
        (ScaledChiSquareSample.([0.15, 0.3, 1.0, 2.0, 5.0], 4),
            Empirikos.InverseScaledChiSquare(1.0, 4.0)),
        (NormalSample.([-2.0, -0.5, 0.0, 0.5, 2.0], 1.0), Normal(0, 2)),
    )
    for (data, prior) in fixtures, extra_slots in (2, 105)
        actual = E.NealAlgorithm2(data; prior)
        for _ in 1:extra_slots
            push!(actual.components, empty(first(actual.components)))
            push!(actual.empties, length(actual.components))
        end
        expected = deepcopy(actual)
        for step in 1:10
            Random.seed!(7200 + step)
            sweep_before_cleanup_change!(expected)
            assigned_components = expected.components[expected.assignments]
            E.cleanup_components!(expected)
            next_random = rand()

            Random.seed!(7200 + step)
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
        end
    end
end

@testset "Algorithm 8 cleanup schedule unchanged" begin
    E = EmpirikosBNP
    data = ScaledChiSquareSample.([0.15, 0.3, 1.0, 2.0, 5.0], 4)
    for extra_slots in (2, 105)
        actual = E.NealAlgorithm8(data; prior = Empirikos.InverseScaledChiSquare(1.0, 4.0))
        for _ in 1:extra_slots
            push!(actual.components, empty(first(actual.components)))
            push!(actual.empties, length(actual.components))
        end
        @test E._should_cleanup(actual) == (length(actual.empties) > 100)
        expected = deepcopy(actual)
        for step in 1:10
            Random.seed!(7300 + step)
            sweep_before_cleanup_change!(expected)
            next_random = rand()
            Random.seed!(7300 + step)
            sample!(actual)
            @test rand() == next_random
            @test actual.assignments == expected.assignments
            @test actual.components == expected.components
            @test actual.logα == expected.logα
            @test actual.empties == expected.empties
        end
    end
end
