abstract type DistributionVariate <: Distributions.VariateForm end 

kfun(base::Distribution, x::AbstractFloat, j::Int) = min(floor(Int, 2^j * cdf(base, x)) + 1, 2^j)
_ns(base::Distribution, J::Int, x::AbstractVector) = map(j -> counts(kfun.(base, x, j), 1:2^j), 1:J)

Base.@kwdef struct PolyaTreeDistribution{D,F,O,V} <: Distribution{DistributionVariate,Continuous}
    J::Int64 = 7
    base::D
    α::Float64 = 10.0
    ρ::F = (j,k) -> j^2
    offsets::O = _ns(base, J, Int[])
    median_centered::Bool = true
    symmetrized::Bool = false
    stored_splits::V =  map(j -> quantile.(Ref(base), (1:(2^j-1)) ./ 2^j), 1:J)
end

function Base.show(io::IO, d::PolyaTreeDistribution)
    print(io, "PolyaTreeDistribution(")
    print(io, "$(d.base), ")
    print(io, "J = $(d.J), ")
    print(io, "α = $(d.α), ")
    print(io, "median_centered = $(d.median_centered), ")
    print(io, "symmetrized = $(d.symmetrized)")
    print(io, ")")
end

Base.@kwdef struct PTFun
    inner_multiplier::Float64
    inner_power::Int64 
    boundary_multiplier::Float64 = inner_multiplier
    boundary_power::Int64 = inner_power
end

# TODO: Check why this is defined at the boundary only. Is it only true for *folded* base measure?
function (f::PTFun)(j,k)
    k == 2^(j-1) ? (f.boundary_multiplier * j^f.boundary_power) : (f.inner_multiplier * j^f.inner_power)
end

function _grid_points(polya::PolyaTreeDistribution) 
    quantile.(polya.base, (1:(2^polya.J-1)) ./ 2^polya.J)
end 

function kfun(polya::PolyaTreeDistribution, x::AbstractFloat, j::Int)
    searchsortedlast(polya.stored_splits[j], x) + 1
end 

function _ns(polya::PolyaTreeDistribution, x::AbstractVector) 
    map(j -> counts(kfun.(Ref(polya), x, j), 1:2^j), 1:polya.J)
end

struct PolyaTree{P,T} <: Distribution{Univariate, Continuous}
    pt::P
    θs::T
end

function _prob(pt::PolyaTree, x)
    θs = pt.θs 
    mapreduce( ((j, θ),) -> θ[kfun(pt.pt, x, j)], (a,b) -> a * b, enumerate(θs) )
end

function _log_prob_old(pt::PolyaTree, x)
    θs = pt.θs
    sum(log.(θ[kfun(pt.pt, x, j)]) for (j, θ) in enumerate(θs))
end

function _log_prob(pt::PolyaTree, x)
    θs = pt.θs
    log_prob = 0.0
    for (j, θ) in enumerate(θs)
        index = kfun(pt.pt, x, j)
        log_prob += log(θ[index])
    end
    log_prob
end

function Distributions.logpdf(pt::PolyaTree, x::Real)
    symmetrized = pt.pt.symmetrized
    J = pt.pt.J
    base = pt.pt.base
    x = symmetrized ? abs(x) : x
    log_f = _log_prob(pt, x) + Distributions.logpdf(base, x) + J * log(2)
    symmetrized ? log_f - log(2) : log_f
end

function Distributions.pdf(pt::PolyaTree, x::Real)
    symmetrized = pt.pt.symmetrized
    J = pt.pt.J
    base = pt.pt.base
    x = symmetrized ? abs(x) : x
    f = 2^J * _prob(pt, x) * Distributions.pdf(base, x)
    symmetrized ? f/2 : f
end

#function Distributions.logpdf(pt::PolyaTree, x::Real)
#   log(pdf(pt, x))
#end

function Base.rand(rng::AbstractRNG, d::PolyaTreeDistribution)
    n = d.offsets
    α = d.α
    ρ = d.ρ
    θs = map(layer -> begin
        j, nj = layer
        m = reshape(nj, 2, 2^(j-1))
        θl = map(i -> ULogarithmic(rand(rng, Beta((α * ρ(j,i) .+ m[:,i])...))), 1:2^(j-1))
        mapreduce(θ -> [θ, one(θ) - θ], vcat, θl)
        end, enumerate(n))
    if d.median_centered
        θs[1] = ULogarithmic.([0.5; 0.5])
    end
    PolyaTree(d, θs)   
end 

function Base.rand(rng::AbstractRNG, d::MixtureModel{DistributionVariate})
    rand(rng, component(d, rand(rng, d.prior)))
end

function ∫x²dP(d::Normal, a, b)
    μ, σ = params(d)
    if μ != 0 || σ != 1
        throw(ArgumentError("Only implemented for N(0,1) currently"))
    end
    if a >= 0 && isinf(a) 
        throw(ArgumentError("a cannot be +Inf"))
    end
    if b <= 0 && isinf(b) 
        throw(ArgumentError("b cannot be -Inf"))
    end
    # use interval dispatch to clean following code
    if isinf(a) && isinf(b)
        int = var(d)
    elseif isinf(b)
        int = ccdf(d, a) + a*pdf(d, a)
    elseif isinf(a)
        int = cdf(d, b) - b*pdf(d, b)
    else 
        int = cdf(d, b) - cdf(d, a) + (a*pdf(d, a) - b*pdf(d, b))
    end
    int 
end

function ∫x²dP(d::TDist, a, b)
    ν = d.ν
    if ν != 5 && ν != 8
        throw(ArgumentError("Only implemented for TDist(5) and TDist(8) currently"))
    end

    if a >= 0 && isinf(a) 
        throw(ArgumentError("a cannot be +Inf"))
    end
    if b <= 0 && isinf(b) 
        throw(ArgumentError("b cannot be -Inf"))
    end
    
    function primitive(t, ν)
        if ν == 5 
            if isinf(t)
                return t > 0 ? Float64(5/6) : Float64(-5/6)
            else 
                sqrt5 = sqrt(5.0)
                term1 = (5.0 * t * (t^2 - 5)) / ((t^2 + 5)^2)
                term2 = sqrt5 * atan(t / sqrt5)
                result = (sqrt5 * (term1 + term2)) / (3 * π)
                return result
            end 
        elseif ν == 8
            if isinf(t)
                return t > 0 ? Float64(2/3) : Float64(-2/3)
            else 
                return (2 * t^3 * (t^4 + 28 * t^2 + 280)) / (3 * (t^2 + 8)^(7/2))
            end
        else 
            throw(ArgumentError("Only implemented for TDist(5) and TDist(8) currently"))
        end
   
    end

    primitive(b, ν) - primitive(a, ν)
end

function ∫x²dP(d::Empirikos.Folded, a, b)
    if a < 0 || b < 0 || a > b
        throw(ArgumentError("Only implemented for 0≤a<b"))
    end 
    2 * ∫x²dP(Empirikos.unfold(d), a, b)
end

function ∫x²dP(d::Distributions.LocationScale, a, b)
    if d.μ != 0 
        throw(ArgumentError("Only implemented under zero centering currently"))
    end

    σ = d.σ 

    abs2(σ) * ∫x²dP(d.ρ, a/σ, b/σ)
end


function ∫xdP(d::Normal, a, b)
    μ, σ = params(d)
    if μ != 0 || σ != 1
        throw(ArgumentError("Only implemented for N(0,1) currently"))
    end
    if a >= 0 && isinf(a) 
        throw(ArgumentError("a cannot be +Inf"))
    end
    if b <= 0 && isinf(b) 
        throw(ArgumentError("b cannot be -Inf"))
    end
    # use interval dispatch to clean following code
    if isinf(a) && isinf(b)
        int = μ  # Expected value of N(0,1) is 0
    elseif isinf(b)
        int = pdf(d, a)
    elseif isinf(a)
        int = -pdf(d, b)
    else 
        int = pdf(d, a) - pdf(d, b)
    end
    Float64(int) 
end

function ∫xdP(d::TDist, a, b)
    ν = d.ν
    if ν != 5 && ν != 8
        throw(ArgumentError("Only implemented for TDist(5) and TDist(8) currently"))
    end

    if a >= 0 && isinf(a) 
        throw(ArgumentError("a cannot be +Inf"))
    end
    if b <= 0 && isinf(b) 
        throw(ArgumentError("b cannot be -Inf"))
    end
    
    function primitive(t, ν)
        if ν == 5 
            if isinf(t)
                return t > 0 ? Float64(0.0) : Float64(0.0)  # The limit is 0 as t approaches ±∞
            else 
                return -(50 * sqrt(5)) / (3 * π * (t^2 + 5)^2)
            end 
        elseif ν == 8
            if isinf(t)
                return t > 0 ? Float64(0.0) : Float64(0.0)  # The limit is 0 as t approaches ±∞
            else 
                return -640 / (t^2 + 8)^(7/2)
            end
        else 
            throw(ArgumentError("Only implemented for TDist(5) and TDist(8) currently"))
        end
    end

    primitive(b, ν) - primitive(a, ν)
end

function ∫xdP(d::Distributions.LocationScale, a, b)
    if d.μ != 0 
        throw(ArgumentError("Only implemented under zero centering currently"))
    end

    σ = d.σ 

    σ * ∫xdP(d.ρ, a/σ, b/σ)
end


function StatsBase.var(pt::PolyaTree)
    if !pt.pt.symmetrized
        throw(ArgumentError("Variance calculation is only valid for symmetric PolyaTree distributions"))
    end
    
    J = pt.pt.J
    base = pt.pt.base
    qs = quantile.(base, collect((0:(2^J))/ 2^J))
    qs_len = length(qs) - 1
    ps = zeros(Float64, qs_len)
    sq = zeros(Float64, qs_len)
    for i in Base.OneTo(qs_len)
        midpt = (qs[i] + qs[i+1])/2
        ps[i] = _prob(pt, midpt)
        sq[i] = ∫x²dP(base, qs[i], qs[i+1])
    end
    ps .*= 2^J
    sum(sq, weights(ps))
end

# Posterior Computations


abstract type AbstractIIDSample{V} <: Empirikos.EBayesSample{V} end

struct IIDSample{V} <: AbstractIIDSample{V}
    Z::V
end

iid_samples(samples::IIDSample) = samples.Z # rename to response?
StatsBase.nobs(IIDSample) = length(IIDSample.Z)

Base.@kwdef mutable struct ConfigurationSample{V, S, T} <: AbstractIIDSample{V}
    configuration::V
    S²::S = ScaledChiSquareSample(var(configuration), length(configuration) - 1)
    Z̄::T = zero(Float64)
end


function ConfigurationSample(iid_sample::IIDSample)
    Z̄ = mean(iid_sample.Z)
    configuration = iid_sample.Z .- Z̄
    ConfigurationSample(configuration=configuration)
end

ScaledChiSquareSample(config::ConfigurationSample) = config.S²

StatsBase.nobs(config::ConfigurationSample) = length(config.configuration)

iid_samples(config::ConfigurationSample, z̄) = config.configuration .+ z̄
iid_samples(config::ConfigurationSample) = iid_samples(config, config.Z̄)



function Distributions.logpdf(d::Distribution, iid_sample::AbstractIIDSample)
    sum(Distributions.logpdf.(d, iid_samples(iid_sample)))
end 

function Distributions.logpdf(d::Distribution, iid_sample::ConfigurationSample, z̄)
    sum(Distributions.logpdf.(d, iid_samples(iid_sample, z̄)))
end 



#function Distributions.pdf(d::Distribution, iid_sample::AbstractIIDSample)
#    prod(exp.(ULogarithmic, Distributions.logpdf.(d, iid_samples(iid_sample))))
#end 


function Empirikos.posterior(sample::AbstractIIDSample, model::PolyaTreeDistribution)
    resp = iid_samples(sample)
    offset_post = model.symmetrized ? _ns(model, abs.(resp)) : _ns(model, resp)
    offset_old = model.offsets
    post_model = @set model.offsets = offset_post .+ offset_old
    post_model
end

function zero_offsets!(model::PolyaTreeDistribution)
    model.offsets .= zero.(model.offsets)
    model
end

function posterior!(sample::AbstractIIDSample, model::PolyaTreeDistribution, σ = 1.0)
    resp = iid_samples(sample)
    offset_post = model.symmetrized ? _ns(model, abs.(resp) .* σ) : _ns(model, resp .* σ)
    model.offsets .+= offset_post
    model
end

function posterior!(samples::AbstractVector{<:AbstractIIDSample}, model::PolyaTreeDistribution, σ=1.0)
    for sample in samples
        posterior!(sample, model, σ)
    end
end

struct VarianceIIDSample{D, V} <: Empirikos.EBayesSample{V}
    iidsample::V
    base::D 
end

function Empirikos.likelihood_distribution(Z::VarianceIIDSample, param)
    sqrt(param) * Z.base
end
function StatsBase.response(Z::VarianceIIDSample)
    Z.iidsample
end

