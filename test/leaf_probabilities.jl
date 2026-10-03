# The pre-cache implementation, including its summation order.
function uncached_log_prob(tree, x)
    J = length(tree.θs)
    log_prob = 0.0
    J == 0 && return log_prob
    index = EmpirikosBNP.kfun(tree.pt, x, J)
    for j in J:-1:1
        log_prob += log(tree.θs[j][index])
        index = cld(index, 2)
    end
    log_prob
end

@testset "Cached leaf probabilities" begin
    rng = MersenneTwister(581)
    for J in (1, 3, 8), (base, symmetrized) in (
        (Normal(), false), (TDist(8), false),
        (Empirikos.fold(Normal()), true),
        (Empirikos.fold(TDist(8) / std(TDist(8))), true),
    )
        prior = PolyaTreeDistribution(; base, J, symmetrized, median_centered = false)
        tree = rand(rng, prior)
        splits = prior.stored_splits[end]
        xs = vcat(randn(rng, 100), splits, prevfloat.(splits), nextfloat.(splits),
            -splits, [-Inf, -0.0, 0.0, Inf, NaN])
        @test length(tree.log_leaf_probs) == 2^J
        for x in xs
            @test isequal(EmpirikosBNP._log_prob(tree, x), uncached_log_prob(tree, x))
            y = symmetrized ? abs(x) : x
            expected = (uncached_log_prob(tree, y) + J * log(2)) + tree.base_logpdf(y)
            @test isequal(logpdf(tree, x), expected)
        end
        @test (@inferred EmpirikosBNP._log_prob(tree, 0.3)) isa Float64
        copied = deepcopy(tree)
        @test copied.log_leaf_probs == tree.log_leaf_probs
        @test copied.log_leaf_probs !== tree.log_leaf_probs
    end

    prior = PolyaTreeDistribution(base = Normal(), J = 0, median_centered = false)
    tree = EmpirikosBNP.PolyaTree(prior, Vector{Float64}[])
    @test tree.log_leaf_probs == [0.0]
    for x in (-Inf, 0.0, Inf, NaN)
        @test isequal(EmpirikosBNP._log_prob(tree, x), uncached_log_prob(tree, x))
    end

    # Do not narrow user-supplied weights to Float64; zero weights retain -Inf.
    for T in (Float64, BigFloat)
        prior = PolyaTreeDistribution(base = Normal(), J = 1, median_centered = false)
        tree = EmpirikosBNP.PolyaTree(prior, [T[0, 1]])
        for x in (-1.0, 0.0, 1.0)
            @test isequal(EmpirikosBNP._log_prob(tree, x), uncached_log_prob(tree, x))
        end
    end
end
