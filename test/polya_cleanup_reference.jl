module PolyaCleanupReference
using EmpirikosBNP, Empirikos, Random, StatsBase
using EmpirikosBNP: @set
const E = EmpirikosBNP

# Frozen sweep from b08b586, retaining its cleanup threshold of 20.
function sweep!(gc)
    vp = gc.vp
    realized_pt = rand(vp.base_polya)
    norm_constant = std(realized_pt)
    vp.realized_pt = realized_pt / norm_constant
    E.zero_offsets!(vp.base_polya)
    for i in eachindex(gc.data)
        sample!(gc, i)
    end
    for (comp_idx, comp) in enumerate(gc.components)
        isempty(comp) && continue
        empty!(gc.scratch)
        for i in eachindex(gc.data)
            if gc.assignments[i] == comp_idx
                push!(gc.scratch, gc.data[i])
            end
        end
        vp.σ² = comp.param
        variance_mh = vp.variance_mh
        variance_mh = @set variance_mh.default_dist = Empirikos.posterior(comp.sample, gc.prior)
        vp.variance_mh = variance_mh
        σ² = E.sample_variance!(vp, gc.scratch)
        σ = sqrt(σ²)
        τ_inv = norm_constant / σ
        comp = @set comp.param = σ²
        gc.components[comp_idx] = comp
        vp.realized_pt = vp.realized_pt * σ
        for observation in gc.scratch
            E.impute_zbar!(vp, observation)
            E.posterior!(observation, vp.base_polya, τ_inv)
        end
        vp.realized_pt = vp.realized_pt / std(vp.realized_pt)
    end
    E.sample_α!(gc)
    length(gc.empties) > 20 && E.cleanup_components!(gc)
    gc
end
end
