function merge_samples(Ss::AbstractVector{<:ScaledChiSquareSample})
    total_dof = sum(nuisance_parameter.(Ss))
    total_Ssq = mean(response.(Ss))
    ScaledChiSquareSample(total_Ssq, total_dof)
end

function sub(orig::ScaledChiSquareSample, rm::ScaledChiSquareSample)
    ν = orig.sample.ν - rm.ν
    Ss = ν == 0 ? 0.0 : (orig.sample.ν * response(orig.sample) - response(rm)*rm.ν)/ν
    ScaledChiSquareSample(Ss, ν)
end

function add(orig::ScaledChiSquareSample, rm::ScaledChiSquareSample)
    ν = orig.sample.ν + rm.ν
    Ss = (orig.sample.ν * response(orig.sample) + response(rm)*rm.ν)/ν
    ScaledChiSquareSample(Ss, ν)
end

Base.empty(::ScaledChiSquareSample) = ScaledChiSquareSample(0.0, 0)


#struct WrappedEBSample{T,P}
#    sample::T
#    n::Int
#    param::P #debatable
#end

#WrappedEBSample(sample, n) = WrappedEBSample(sample, n, 0.0)


#function merge_samples(Ss::AbstractVector{<:ScaledChiSquareSample})
#    total_dof = sum(nuisance_parameter.(Ss))
#    total_Ssq = mean(response.(Ss))
#    WrappedEBSample(ScaledChiSquareSample(total_Ssq, total_dof), length(Ss), total_Ssq)
#end




#function sub(orig::WrappedEBSample{<:ScaledChiSquareSample}, rm::ScaledChiSquareSample)
#   ν = orig.sample.ν - rm.ν
#  Ss = ν == 0 ? 0.0 : (orig.sample.ν * response(orig.sample) - response(rm)*rm.ν)/ν
#    WrappedEBSample(ScaledChiSquareSample(Ss, ν), orig.n - 1, orig.param)
#end

#function add(orig::WrappedEBSample{<:ScaledChiSquareSample}, rm::ScaledChiSquareSample)
#    ν = orig.sample.ν + rm.ν
#    Ss = (orig.sample.ν * response(orig.sample) + response(rm)*rm.ν)/ν
#    WrappedEBSample(ScaledChiSquareSample(Ss, ν), orig.n + 1, orig.param)
#end

#Base.isempty(Ss::WrappedEBSample) = Ss.n == 0


#function Base.empty(S::WrappedEBSample)
#    S = @set S.sample = empty(S.sample)
#    S = @set S.n = 0
#    S
#end





#function sub(orig::WrappedEBSample{<:ScaledChiSquareSample}, rm::ScaledChiSquareSample)
#    ν = orig.sample.ν - rm.ν
#    Ss = ν == 0 ? 0.0 : (orig.sample.ν * response(orig.sample) - response(rm)*rm.ν)/ν
#    WrappedEBSample(ScaledChiSquareSample(Ss, ν), orig.n - 1, orig.param)
#end

#function add(orig::WrappedEBSample{<:ScaledChiSquareSample}, rm::ScaledChiSquareSample)
#    ν = orig.sample.ν + rm.ν
#    Ss = (orig.sample.ν * response(orig.sample) + response(rm)*rm.ν)/ν
#    WrappedEBSample(ScaledChiSquareSample(Ss, ν), orig.n + 1, orig.param)
#end

#Base.isempty(Ss::WrappedEBSample) = Ss.n == 0


#function Base.empty(S::WrappedEBSample)
#    S = @set S.sample = empty(S.sample)
#    S = @set S.n = 0
#    S
#end
