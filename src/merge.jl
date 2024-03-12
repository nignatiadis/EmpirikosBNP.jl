function merge_samples(Ss::AbstractVector{<:ScaledChiSquareSample})
    total_dof = sum(nuisance_parameter.(Ss))
    total_Ssq = mean(response.(Ss))
    ScaledChiSquareSample(total_Ssq, total_dof)
end

function sub(orig::ScaledChiSquareSample, rm::ScaledChiSquareSample)
    ν = orig.ν - rm.ν
    Ss = ν == 0 ? 0.0 : (orig.ν * response(orig) - response(rm)*rm.ν)/ν
    ScaledChiSquareSample(Ss, ν)
end

function add(orig::ScaledChiSquareSample, rm::ScaledChiSquareSample)
    ν = orig.ν + rm.ν
    Ss = (orig.ν * response(orig) + response(rm)*rm.ν)/ν
    ScaledChiSquareSample(Ss, ν)
end

Base.empty(::ScaledChiSquareSample) = ScaledChiSquareSample(0.0, 0)

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
    IIDSample(view(Ss, collect(1:length(Ss)))) 
end

# actually sub!
function sub(orig::IIDSample{<:SubArray}, i)
    idx = orig.Z.indices[1]
    deleteat!(idx , findfirst(==(i), idx))
    orig
end

function add(orig::IIDSample{<:SubArray}, i)
    idx = orig.Z.indices[1]
    push!(idx, i)
    orig
end

function Base.empty(orig::IIDSample{<:SubArray})
    newview = view(orig.Z.parent, empty(orig.Z.indices[1]))
    IIDSample(newview)
end


