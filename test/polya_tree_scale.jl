function tree_scale_fixture(; J = 2, extra_slots = 0)
    E = EmpirikosBNP
    Random.seed!(92100)
    data = [E.ConfigurationSample(E.IIDSample(z))
        for z in ([-0.3, 0.1, 0.2], [-1.2, 0.0, 1.2], [-0.7, 0.1, 0.6])]
    prior = Empirikos.InverseScaledChiSquare(1.0, 6.0)
    neal = E.NealAlgorithm2(ScaledChiSquareSample.(data); prior)
    neal.assignments .= [1, 2, 1]
    neal.components = [
        E.WrappedEBSample(E.merge_samples([data[1].S², data[3].S²]), 2, 0.8),
        E.WrappedEBSample(data[2].S², 1, 1.7),
    ]
    for _ in 1:(1 + extra_slots)
        push!(neal.components, empty(first(neal.components)))
    end
    neal.empties = collect(3:length(neal.components))
    base_polya = PolyaTreeDistribution(base = Empirikos.fold(Normal()), J = J,
        α = 1.0, symmetrized = true, median_centered = false)
    E.NealAlgorithm8Polya(data; base_polya, neal_cp = neal)
end

function tree_branch_logpdf(tree, proposal; with_counts)
    sum(1:proposal.J) do j
        sum(1:2^(j - 1)) do k
            a = proposal.α * proposal.ρ(j, k)
            left = with_counts ? proposal.offsets[j][2k - 1] : 0
            right = with_counts ? proposal.offsets[j][2k] : 0
            logpdf(Beta(a + left, a + right), Float64(tree.θs[j][2k - 1]))
        end
    end
end

@testset "Joint tree/scale MH acceptance and rejection" begin
    E = EmpirikosBNP
    accepted_count = 0
    for extra_slots in (0, 25), seed in 1:40
        gc = tree_scale_fixture(; extra_slots)
        raw = gc.vp.realized_pt.ρ
        original = deepcopy(gc)
        proposal = deepcopy(gc.vp.base_polya)
        # Independent counts, using the base CDF rather than posterior!.
        for datum in eachindex(gc.data)
            variance = gc.components[gc.assignments[datum]].param
            for z in E.iid_samples(gc.data[datum]), j in 1:proposal.J
                u = cdf(proposal.base, abs(z) * sqrt(var(raw) / variance))
                index = min(floor(Int, 2^j * u) + 1, 2^j)
                proposal.offsets[j][index] += 1
            end
        end
        Random.seed!(92200 + seed)
        proposed = rand(proposal)
        ratio = (std(proposed) / std(raw))^2
        oldvars = [c.param for c in gc.components if !isempty(c)]
        newvars = ratio .* oldvars
        function target(tree, variances)
            tree_branch_logpdf(tree, proposal; with_counts = false) +
                sum(logpdf.(Ref(gc.prior), variances)) +
                sum(eachindex(gc.data)) do i
                    density = tree * (sqrt(variances[gc.assignments[i]]) / std(tree))
                    logpdf(density, gc.data[i])
                end
        end
        full_ratio = target(proposed, newvars) - target(raw, oldvars) +
            tree_branch_logpdf(raw, proposal; with_counts = true) -
            tree_branch_logpdf(proposed, proposal; with_counts = true) +
            length(oldvars) * log(ratio)
        simplified = sum(log(ratio) + logpdf(gc.prior, v * ratio) - logpdf(gc.prior, v) for v in oldvars)
        @test full_ratio ≈ simplified atol = 1e-10
        expected_accept = -randexp() < full_ratio
        next_random = rand()

        # Stale proposal counts must not affect the move.
        foreach(offset -> fill!(offset, 999), gc.vp.base_polya.offsets)
        Random.seed!(92200 + seed)
        accepted = E.sample_tree_and_scales!(gc)
        accepted_count += accepted
        @test accepted == expected_accept
        @test rand() == next_random
        @test gc.vp.base_polya.offsets == proposal.offsets
        @test gc.assignments == original.assignments
        @test gc.logα == original.logα
        @test gc.empties == original.empties
        @test gc.components[gc.empties] == original.components[original.empties]
        @test [c.sample for c in gc.components] == [c.sample for c in original.components]
        @test [c.param for c in gc.components if !isempty(c)] == (accepted ? newvars : oldvars)
        expected_tree = accepted ? proposed : raw
        @test gc.vp.realized_pt.ρ.θs == expected_tree.θs
        @test std(gc.vp.realized_pt) ≈ 1
        @test isapprox(var(gc.vp.realized_pt.ρ) / first(gc.components).param,
            var(raw) / first(original.components).param; rtol = 1e-12)
    end
    @test 0 < accepted_count < 80
end

@testset "One-level tree posterior against quadrature" begin
    E = EmpirikosBNP
    # Two fixed observations at zero. This posterior factorizes into
    # p(root) proportional to p^2 * var(raw_tree), and v ~ InverseGamma(4, 3).
    prior = Empirikos.InverseScaledChiSquare(1.0, 6.0)
    data = [E.ConfigurationSample(E.IIDSample([0.0, 0.0]))]
    base_polya = PolyaTreeDistribution(base = Empirikos.fold(Normal()), J = 1,
        α = 1.0, symmetrized = true, median_centered = false)
    Random.seed!(92300)
    neal = E.NealAlgorithm2(ScaledChiSquareSample.(data); prior)
    gc = E.NealAlgorithm8Polya(data; base_polya, neal_cp = neal)
    boundary = quantile(Normal(), 0.75)
    left_second = 4quadgk(x -> x^2 * pdf(Normal(), x), 0, boundary)[1]
    rawvar(p) = p * left_second + (1 - p) * (2 - left_second)
    weight(p) = p^2 * rawvar(p)
    norm = quadgk(weight, 0, 1)[1]
    root_mean = quadgk(p -> p * weight(p), 0, 1)[1] / norm
    # Tail of the standardized shape at 1.5, independently from its two cells.
    function shape_tail(p)
        t = 1.5sqrt(rawvar(p))
        left = max(0.0, (cdf(Normal(), boundary) - cdf(Normal(), t)) * 4)
        right = min(1.0, 4ccdf(Normal(), t))
        p * left + (1 - p) * right
    end
    tail_mean = quadgk(p -> shape_tail(p) * weight(p), 0, 1; rtol = 1e-9)[1] / norm
    values = zeros(20000, 3)
    for iteration in 1:22000
        E.sample_tree_and_scales!(gc)
        # Exact conditional variance refresh isolates the new joint move.
        comp = first(gc.components)
        gc.components[1] = E.WrappedEBSample(comp.sample, comp.n, rand(InverseGamma(4, 3)))
        if iteration > 2000
            p = Float64(gc.vp.realized_pt.ρ.θs[1][1])
            values[iteration - 2000, :] .= (p, gc.components[1].param, shape_tail(p))
        end
    end
    exact = [root_mean, 1.0, tail_mean]
    for j in 1:3
        batches = vec(mean(reshape(values[:, j], 200, 100); dims = 1))
        se = std(batches) / sqrt(length(batches))
        @test abs(mean(values[:, j]) - exact[j]) < 6se
    end
end
