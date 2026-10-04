module Neal2PartitionReference
using Distributions, Empirikos, EmpirikosBNP, Random, StatsBase

function partitions(n)
    result = Vector{Vector{Int}}()
    labels = ones(Int, n)
    function visit(i, largest)
        if i > n
            push!(result, copy(labels))
            return
        end
        for k in 1:(largest + 1)
            labels[i] = k
            visit(i + 1, max(largest, k))
        end
    end
    visit(2, 1)
    result
end

canonical(labels) = Tuple(indexin(labels, unique(labels)))

# Independent inverse-gamma integration, omitting observation constants common
# to every partition. logpdf at 1 supplies a*log(b) - loggamma(a) - b.
function block_posterior(data, indices, prior)
    a = (prior.ν + sum(data[i].ν for i in indices)) / 2
    b = (prior.ν * prior.σ² + sum(data[i].ν * response(data[i]) for i in indices)) / 2
    InverseGamma(a, b)
end

function exact_posterior(data, prior, alpha)
    states = partitions(length(data))
    base = InverseGamma(prior.ν / 2, prior.ν * prior.σ² / 2)
    logweights = map(states) do labels
        result = maximum(labels) * log(alpha)
        for k in 1:maximum(labels)
            indices = findall(==(k), labels)
            post = block_posterior(data, indices, prior)
            result += sum(log, 1:(length(indices) - 1); init = 0.0)
            result += (logpdf(base, 1.0) + scale(base)) -
                (logpdf(post, 1.0) + scale(post))
        end
        result
    end
    probs = exp.(logweights .- maximum(logweights))
    probs ./= sum(probs)
    (; states, probs)
end

function draw_state(data, prior, alpha, reference)
    labels = reference.states[sample(Weights(reference.probs))]
    gc = EmpirikosBNP.NealAlgorithm2(data; prior)
    gc.logα = log(alpha)
    gc.assignments .= labels
    gc.components = map(1:maximum(labels)) do k
        indices = findall(==(k), labels)
        summary = EmpirikosBNP.merge_samples(data[indices])
        EmpirikosBNP.WrappedEBSample(summary, length(indices),
            rand(block_posterior(data, indices, prior)))
    end
    push!(gc.components, empty(gc.components[1]))
    gc.empties = [length(gc.components)]
    gc
end

function invariance_check(data, prior, alpha; draws = 20000, seed = 551)
    reference = exact_posterior(data, prior, alpha)
    lookup = Dict(Tuple(labels) => i for (i, labels) in enumerate(reference.states))
    counts = zeros(Int, length(reference.states))
    Random.seed!(seed)
    for _ in 1:draws
        # Independent exact posterior starts avoid an MCMC mixing assumption.
        gc = draw_state(data, prior, alpha, reference)
        for i in eachindex(data)
            sample!(gc, i)
        end
        counts[lookup[canonical(gc.assignments)]] += 1
    end
    (; reference..., observed = counts / draws, draws)
end
end
