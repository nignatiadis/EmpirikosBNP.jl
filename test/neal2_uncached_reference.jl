module Neal2UncachedReference

using Distributions, Empirikos, EmpirikosBNP, StatsBase
const E = EmpirikosBNP

# Algorithm 2 assignment from commit 9f2055d, before prior-predictive caching.
function assignment!(gc, i)
    x = gc.data[i]
    old_comp = E.sub(gc.components[gc.assignments[i]], x)
    gc.components[gc.assignments[i]] = old_comp
    isempty(old_comp) && push!(gc.empties, gc.assignments[i])
    log_probs = [isempty(comp) ? -Inf : loglikelihood(x, comp.param) + log(comp.n)
        for comp in gc.components]
    log_probs[first(gc.empties)] = logpdf(gc.prior, x) + gc.logα
    new_k = sample(E._weights_from_logprobs(log_probs))
    gc.components[new_k] = E.add(gc.components[new_k], x)
    gc.assignments[i] = new_k
    if new_k == first(gc.empties)
        E.sample_component_param!(gc, new_k)
        popfirst!(gc.empties)
        if isempty(gc.empties)
            push!(gc.components, empty(first(gc.components)))
            push!(gc.empties, length(gc.components))
        end
    end
    gc
end

function sweep!(gc)
    for i in eachindex(gc.data)
        assignment!(gc, i)
    end
    for k in eachindex(gc.components)
        isempty(gc.components[k]) || E.sample_component_param!(gc, k)
    end
    E.sample_α!(gc)
    E.cleanup_components!(gc)
    gc
end

end
