include("neal2_uncached_reference.jl")

@testset "Algorithm 2 prior-predictive cache" begin
    E = EmpirikosBNP
    fixtures = (
        (ScaledChiSquareSample.([0.15, 0.3, 1.0, 2.0, 5.0], 4),
            Empirikos.InverseScaledChiSquare(1.0, 4.0),
            Empirikos.InverseScaledChiSquare(2.0, 6.0)),
        (NormalSample.([-2.0, -0.5, 0.0, 0.5, 2.0], 1.0), Normal(0, 2), Normal(1, 3)),
    )
    for (data, prior, next_prior) in fixtures
        gc = E.NealAlgorithm2(data; prior)
        @test gc.prior_logpdfs == logpdf.(Ref(prior), data)
        cache = gc.prior_logpdfs
        for stage in 1:3
            if stage == 2
                gc.prior = next_prior
            elseif stage == 3
                gc.data .= reverse(gc.data)
                for k in eachindex(gc.components)
                    old = gc.components[k]
                    isempty(old) && continue
                    ids = findall(==(k), gc.assignments)
                    gc.components[k] = E.WrappedEBSample(E.merge_samples(gc.data[ids]), old.n, old.param)
                end
            end
            if stage > 1
                @test gc.prior_logpdfs != logpdf.(Ref(gc.prior), gc.data)
                @test E.refresh_prior_logpdfs!(gc) === gc
            end
            @test gc.prior_logpdfs === cache
            @test gc.prior_logpdfs == logpdf.(Ref(gc.prior), gc.data)
            expected = deepcopy(gc)
            for step in 1:30
                Random.seed!(7600 + step)
                Neal2UncachedReference.sweep!(expected)
                next_random = rand()
                Random.seed!(7600 + step)
                sample!(gc)
                @test rand() == next_random
                @test gc.assignments == expected.assignments
                @test gc.components == expected.components
                @test gc.empties == expected.empties
                @test gc.logα == expected.logα
                @test gc.prior_logpdfs == logpdf.(Ref(gc.prior), gc.data)
            end
        end
    end
    # Keep the density's precision rather than forcing cache entries to Float64.
    gc = E.NealAlgorithm2(NormalSample.([0.1, 2.0], 1.0);
        prior = Normal(big"0.0", big"2.0"))
    @test eltype(gc.prior_logpdfs) == BigFloat
    @test gc.prior_logpdfs == logpdf.(Ref(gc.prior), gc.data)
end
