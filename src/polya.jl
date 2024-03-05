abstract type DistributionVariate <: Distributions.VariateForm end 

kfun(base::Distribution, x::AbstractFloat, j::Int) = min(floor(Int, 2^j * cdf(base, x)) + 1, 2^j)
_ns(base::Distribution, J::Int, x::AbstractVector) = map(j -> counts(kfun.(base, x, j), 1:2^j), 1:J)

Base.@kwdef struct PolyaTreeDistribution{D,F,O} <: Distribution{DistributionVariate,Continuous}
    J::Int64 = 7
    base::D
    α::Float64 = 10.0
    ρ::F = (j) -> j^2
    offsets::O = _ns(base, J, Int[])
    median_centered::Bool = true
    symmetrized::Bool = false
end

function _grid_points(polya::PolyaTreeDistribution) 
    quantile.(polya.base, (1:(2^polya.J-1)) ./ 2^polya.J)
end 

kfun(polya::PolyaTreeDistribution, x::AbstractFloat, j::Int) = kfun(polya.base, x, j)
_ns(polya::PolyaTreeDistribution, x::AbstractVector) = _ns(polya.base, polya.J, x)

struct PolyaTree{P,T} <: Distribution{Univariate, Continuous}
    pt::P
    θs::T
end

function _prob(pt::PolyaTree, x)
    θs = pt.θs 
    mapreduce( ((j, θ),) -> θ[kfun(pt.pt, x, j)], (a,b) -> a * b, enumerate(θs) )
end

function Distributions.pdf(pt::PolyaTree, x::Real)
    symmetrized = pt.pt.symmetrized
    J = pt.pt.J
    base = pt.pt.base
    x = symmetrized ? abs(x) : x
    f = 2^J * _prob(pt, x) * Distributions.pdf(base, x)
    symmetrized ? f/2 : f
end

function Distributions.logpdf(pt::PolyaTree, x::Real)
   log(pdf(pt, x))
end

function Base.rand(rng::AbstractRNG, d::PolyaTreeDistribution)
    n = d.offsets
    α = d.α
    ρ = d.ρ
    θs = map(layer -> begin
        j, nj = layer
        m = reshape(nj, 2, 2^(j-1))
        θl = map(i -> ULogarithmic.(rand(rng, Beta(α * ρ(j) .+ m[:,i]...))), 1:2^(j-1))
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
    if ν != 5 
        throw(ArgumentError("Only implemented for TDist(5) currently"))
    end

    if a >= 0 && isinf(a) 
        throw(ArgumentError("a cannot be +Inf"))
    end
    if b <= 0 && isinf(b) 
        throw(ArgumentError("b cannot be -Inf"))
    end
    
    function primitive(t)
        sqrt5 = sqrt(5.0)
        term1 = (5.0 * t * (t^2 - 5)) / ((t^2 + 5)^2)
        term2 = sqrt5 * atan(t / sqrt5)
        result = (sqrt5 * (term1 + term2)) / (3 * π)
        return result
    end

    lhs = isinf(a) ? Float64(-5/6) : primitive(a)
    rhs = isinf(b) ? Float64(5/6) : primitive(b)
    rhs - lhs
end

function ∫x²dP(d::Empirikos.Folded, a, b)
    if a < 0 || b < 0 || a > b
        throw(ArgumentError("Only implemented for 0<a<b"))
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



function StatsBase.var(pt::PolyaTree)
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

iid_samples(samples::IIDSample) = samples.Z
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

StatsBase.nobs(config::ConfigurationSample) = length(config.configuration)
iid_samples(config::ConfigurationSample) = config.configuration .+ config.Z̄



function Distributions.logpdf(d::Distribution, iid_sample::AbstractIIDSample)
    sum(Distributions.logpdf.(d, iid_samples(iid_sample)))
end 


function Distributions.pdf(d::Distribution, iid_sample::AbstractIIDSample)
    prod(exp.(ULogarithmic, Distributions.logpdf.(d, iid_samples(iid_sample))))
end 


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

function posterior!(sample::AbstractIIDSample, model::PolyaTreeDistribution)
    resp = iid_samples(sample)
    offset_post = model.symmetrized ? _ns(model, abs.(resp)) : _ns(model, resp)
    model.offsets .+= offset_post
    model
end

function posterior!(samples::AbstractVector{<:AbstractIIDSample}, model::PolyaTreeDistribution)
    for sample in samples
        posterior!(sample, model)
    end
end

struct VarianceIIDSample{D, V<:AbstractIIDSample} <: Empirikos.EBayesSample{V}
    iidsample::V
    base::D 
end

function Empirikos.likelihood_distribution(Z::VarianceIIDSample, param)
    sqrt(param) * Z.base
end
function StatsBase.response(Z::VarianceIIDSample)
    Z.iidsample
end