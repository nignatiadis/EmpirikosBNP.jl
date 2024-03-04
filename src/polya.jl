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