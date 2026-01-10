# Scaled Chi Square Samples 

function merge_samples(Ss::AbstractVector{<:ScaledChiSquareSample})
    total_dof = sum(nuisance_parameter.(Ss))
    total_Ssq = mean(response.(Ss))
    ScaledChiSquareSample(total_Ssq, total_dof)
end

function sub(orig::ScaledChiSquareSample, rm::ScaledChiSquareSample)
    ν = orig.ν - rm.ν
    Ss = ν == 0 ? 0.0 : (orig.ν * response(orig) - rm.ν * response(rm))/ν
    ScaledChiSquareSample(Ss, ν)
end

function add(orig::ScaledChiSquareSample, rm::ScaledChiSquareSample)
    ν = orig.ν + rm.ν
    Ss = (orig.ν * response(orig) + rm.ν * response(rm))/ν
    ScaledChiSquareSample(Ss, ν)
end

Base.empty(::ScaledChiSquareSample) = ScaledChiSquareSample(0.0, 0)

# Normal Samples

function merge_samples(Ss::AbstractVector{<:NormalSample})
    precisions = 1 ./ var.(Ss)
    weighted_mean = mean(response.(Ss), weights(precisions))
    combined_σ = sqrt(1 / sum(precisions))
    NormalSample(weighted_mean, combined_σ)
end

function sub(orig::NormalSample, rm::NormalSample)
    orig_precision = 1 / var(orig)
    rm_precision = 1 / var(rm)
    new_precision = orig_precision - rm_precision
    
    if new_precision <= 0
        return NormalSample(0.0, Inf)
    end
    
    new_Z = (orig_precision * response(orig) - rm_precision * response(rm)) / new_precision
    new_σ = sqrt(1 / new_precision)
    NormalSample(new_Z, new_σ)
end

function add(orig::NormalSample, rm::NormalSample)
    orig_precision = 1 / var(orig)
    rm_precision = 1 / var(rm)
    new_precision = orig_precision + rm_precision
    
    new_Z = (orig_precision * response(orig) + rm_precision * response(rm)) / new_precision
    new_σ = sqrt(1 / new_precision)
    NormalSample(new_Z, new_σ)
end

Base.empty(::NormalSample) = NormalSample(0.0, Inf)

# Wrappers




abstract type AbstractWrappedEBSample{T} end

struct WrappedEBSample{T,P} <: AbstractWrappedEBSample{T}
    sample::T
    n::Int
    param::P #debatable
end

function wrap(samples::AbstractVector)
    n = length(samples)
    ebz = merge_samples(samples)
    WrappedEBSample(ebz, n, 1.0)
end

function sub(orig::WrappedEBSample, rm)
    ebz = sub(orig.sample, rm)
    WrappedEBSample(ebz, orig.n - 1, orig.param)
end

function add(orig::WrappedEBSample, rm)
    ebz = add(orig.sample, rm)
    WrappedEBSample(ebz, orig.n + 1, orig.param)
end

Base.isempty(Ss::WrappedEBSample) = Ss.n == 0


function Base.empty(S::WrappedEBSample)
    S = @set S.sample = empty(S.sample)
    S = @set S.n = 0
    S
end

## AbstractIIDSample merging

function merge_samples(Ss::AbstractVector{<:AbstractIIDSample})
    IIDSample(Ss)
end



# actually sub!
function sub(orig::IIDSample{<:AbstractVector}, rm)
    idx_delete = findfirst(==(rm), orig.Z)
    deleteat!(orig.Z , idx_delete)
    orig
end

function add(orig::IIDSample{<:AbstractVector}, rm)
    push!(orig.Z, rm)
    orig
end

function Base.empty(orig::IIDSample{<:AbstractVector})
    IIDSample(empty(orig.Z))
end


