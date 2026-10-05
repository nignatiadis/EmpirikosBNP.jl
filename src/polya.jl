abstract type DistributionVariate <: Distributions.VariateForm end 

kfun(base::Distribution, x::AbstractFloat, j::Int) = min(floor(Int, 2^j * cdf(base, x)) + 1, 2^j)
_ns(base::Distribution, J::Int, x::AbstractVector) = map(j -> counts(kfun.(base, x, j), 1:2^j), 1:J)

# Exact grid over the finest splits, only when every split is positive, as for
# folded bases; other trees use a binary search. Cells have width 1 / scale, a
# power of two, so for x >= 0 the product x * scale is exact and cell k is exactly
# [k / scale, (k + 1) / scale). Cells are narrower than the gaps between splits,
# so each holds at most one, and no check against the splits is needed.
struct SplitCell
    split::Float64 # the split in this cell, or NaN, which no x reaches
    below::Int     # number of splits below this cell
end

struct SplitLookup
    scale::Float64
    cells::Vector{SplitCell}
end

SplitLookup(splits) = nothing

function SplitLookup(splits::Vector{Float64})
    gap = minimum(diff(splits); init = Inf)
    (isfinite(gap) && gap > 0 && splits[1] > 0 && isfinite(splits[end])) || return nothing
    width = exp2(floor(log2(gap / 2)))
    # Two splits in one cell would be closer than its width.
    width < gap || return nothing
    scale = 1 / width
    # The last cell starts above every split and receives all larger x.
    ncells = floor(Int, splits[end] * scale) + 2
    ncells <= typemax(UInt16) || return nothing
    cells = map(0:(ncells - 1)) do k
        first = searchsortedfirst(splits, k * width)
        inside = first <= length(splits) && splits[first] < (k + 1) * width
        SplitCell(inside ? splits[first] : NaN, first - 1)
    end
    SplitLookup(scale, cells)
end

# Same result as searchsortedlast(splits, x; lt = <) for non-NaN x. Negative x
# land in the first cell, which no split precedes.
@inline function _searchsortedlast(lookup::SplitLookup, x::Float64)
    cells = lookup.cells
    last_cell = length(cells) - 1.0
    v = x * lookup.scale
    v = ifelse(v >= 0.0, v, 0.0)
    v = ifelse(v < last_cell, v, last_cell)
    c = @inbounds cells[unsafe_trunc(Int, v) + 1]
    c.below + (c.split <= x)
end

Base.@kwdef struct PolyaTreeDistribution{D,F,O,V,L} <: Distribution{DistributionVariate,Continuous}
    J::Int64 = 7
    base::D
    α::Float64 = 10.0
    ρ::F = (j,k) -> j^2
    offsets::O = _ns(base, J, Int[])
    median_centered::Bool = true
    symmetrized::Bool = false
    stored_splits::V =  map(j -> quantile.(Ref(base), (1:(2^j-1)) ./ 2^j), 1:J)
    split_lookup::L = J == 0 ? nothing : SplitLookup(stored_splits[J])
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
    symmetric::Bool = true
end

# TODO: Check why this is defined at the boundary only. Is it only true for *folded* base measure?
function (f::PTFun)(j,k)
    if f.symmetric
        fjk = k == 2^(j-1) ? (f.boundary_multiplier * j^f.boundary_power) : (f.inner_multiplier * j^f.inner_power)
    else 
        fjk = ((k == 2^(j-1)) || (k == 1) ) ? (f.boundary_multiplier * j^f.boundary_power) : (f.inner_multiplier * j^f.inner_power)
    end
    fjk
end
function _grid_points(polya::PolyaTreeDistribution) 
    quantile.(polya.base, (1:(2^polya.J-1)) ./ 2^polya.J)
end 

function kfun(polya::PolyaTreeDistribution, x::AbstractFloat, j::Int)
    # Quantile splits contain no NaNs. Preserve total ordering for NaNs and signed zeros.
    splits = polya.stored_splits[j]
    if isnan(x) || iszero(x)
        return searchsortedlast(splits, x) + 1
    end
    lookup = polya.split_lookup
    if j == polya.J && lookup isa SplitLookup && x isa Float64
        return _searchsortedlast(lookup, x) + 1
    end
    searchsortedlast(splits, x; lt = <) + 1
end

function _ns(polya::PolyaTreeDistribution, x::AbstractVector) 
    map(j -> counts(kfun.(Ref(polya), x, j), 1:2^j), 1:polya.J)
end

struct PolyaTree{P,T,F,L} <: Distribution{Univariate, Continuous}
    pt::P
    θs::T
    base_logpdf::F
    log_leaf_probs::L
end

function PolyaTree(pt, θs)
    base_logpdf = pt.symmetrized ?
        _symmetrized_base_logpdf_evaluator(pt.base) : _logpdf_evaluator(pt.base)
    PolyaTree(pt, θs, base_logpdf, _log_leaf_probs(θs))
end

function _prob(pt::PolyaTree, x)
    θs = pt.θs 
    mapreduce( ((j, θ),) -> θ[kfun(pt.pt, x, j)], (a,b) -> a * b, enumerate(θs) )
end

function _log_prob_old(pt::PolyaTree, x)
    θs = pt.θs
    sum(log.(θ[kfun(pt.pt, x, j)]) for (j, θ) in enumerate(θs))
end

function _log_leaf_probs(θs)
    J = length(θs)
    # Weights stay fixed for a realized tree; preserve the leaf-to-root sum order.
    map(1:2^J) do index
        log_prob = 0.0
        for j in J:-1:1
            log_prob += log(θs[j][index])
            index = cld(index, 2)
        end
        log_prob
    end
end

function _log_prob(pt::PolyaTree, x)
    lookup = pt.pt.split_lookup
    # As every split is positive, kfun's ordering of signed zeros changes nothing.
    if lookup isa SplitLookup && x isa Float64 && !isnan(x) && length(pt.θs) == pt.pt.J
        return @inbounds pt.log_leaf_probs[_searchsortedlast(lookup, x) + 1]
    end
    J = length(pt.θs)
    index = J == 0 ? 1 : kfun(pt.pt, x, J)
    pt.log_leaf_probs[index]
end

function Distributions.logpdf(pt::PolyaTree, x::Real)
    symmetrized = pt.pt.symmetrized
    J = pt.pt.J
    x = symmetrized ? abs(x) : x
    log_f = _log_prob(pt, x) + J * log(2)
    log_f + pt.base_logpdf(x)
end

# Prepare a normalized log density for repeated evaluation with fixed parameters.
_logpdf_evaluator(d) = Base.Fix1(logpdf, d)

function _logpdf_evaluator(d::Normal{Float64})
    μ, σ = params(d)
    iszero(σ) && return Base.Fix1(logpdf, d)
    standard = Normal(zero(μ), one(σ))
    logscale = log(σ)
    evaluate(x::Float64) = logpdf(standard, (x - μ) / σ) - logscale
    evaluate(x::Real) = logpdf(d, x)
    evaluate
end

function _logpdf_evaluator(d::TDist{Float64})
    ν = first(params(d))
    isinf(ν) && return Base.Fix1(logpdf, d)
    logconstant = logpdf(d, zero(ν))
    halfνp1 = (ν + 1) / 2
    # StatsFuns supplies the normalizer; only the Student-t kernel depends on x.
    evaluate(x::Float64) = logconstant - halfνp1 * log1p(x^2 / ν)
    evaluate(x::Real) = logpdf(d, x)
    evaluate
end

function _logpdf_evaluator(d::Distributions.LocationScale{<:Any,<:Any,<:Union{Normal,TDist,PolyaTree}})
    μ, σ, base = params(d)
    evaluate = _logpdf_evaluator(base)
    logscale = log(abs(σ))
    x -> evaluate((x - μ) / σ) - logscale
end

function _symmetrized_base_logpdf_evaluator(
    base::Empirikos.Folded{<:Union{Normal,TDist,Distributions.LocationScale{<:Any,<:Any,<:TDist}}},
)
    d = Empirikos.unfold(base)
    iszero(median(d)) || throw(ArgumentError("A symmetrized Polya tree requires a base centered at zero."))
    _logpdf_evaluator(d)
end

_symmetrized_base_logpdf_evaluator(base) =
    throw(ArgumentError("A symmetrized Polya tree requires a folded Normal or Student-t base."))

function Distributions.pdf(pt::PolyaTree, x::Real)
    symmetrized = pt.pt.symmetrized
    J = pt.pt.J
    base = pt.pt.base
    x = symmetrized ? abs(x) : x
    f = 2^J * _prob(pt, x) * Distributions.pdf(base, x)
    symmetrized ? f/2 : f
end



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


function StatsBase.mean(pt::PolyaTree)
    if !pt.pt.symmetrized
        J = pt.pt.J
        base = pt.pt.base
        qs = quantile.(base, collect((0:(2^J))/ 2^J))
        qs_len = length(qs) - 1
        ps = zeros(Float64, qs_len)
        sq = zeros(Float64, qs_len)
        for i in Base.OneTo(qs_len)
            midpt = (qs[i] + qs[i+1])/2
            ps[i] = _prob(pt, midpt)
            sq[i] = ∫xdP(base, qs[i], qs[i+1])
        end
        ps .*= 2^J
        m = sum(sq, weights(ps))
    else 
        m = zero(Float64)
    end 
    m
end

function StatsBase.var(pt::PolyaTree)
    # Compute second moment
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
    sm = sum(sq, weights(ps))

    # Compute first moment
    fm = Empirikos.mean(pt) 

    # Return variance
    sm - abs2(fm)
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

function Distributions.logpdf(
    d::Union{PolyaTree,Distributions.LocationScale{<:Any,<:Any,<:PolyaTree}},
    config::ConfigurationSample,
    z̄ = config.Z̄,
)
    _sum_logpdf(d, config, z̄)
end

function _sum_logpdf(d, config::ConfigurationSample, z̄)
    logdensity = _logpdf_evaluator(d)
    sum(z -> logdensity(z + z̄), config.configuration; init = 0.0)
end

# Location, scale and degrees of freedom of a Student-t tree base, or nothing.
_student_t_parameters(base::TDist{Float64}) = (0.0, 1.0, dof(base))
_student_t_parameters(base::Distributions.LocationScale{Float64,<:Any,TDist{Float64}}) =
    (base.μ, base.σ, dof(base.ρ))
_student_t_parameters(base) = nothing

function _student_t_parameters(tree::PolyaTreeDistribution)
    base = tree.base
    if tree.symmetrized && base isa Empirikos.Folded
        base = Empirikos.unfold(base)
    end
    _student_t_parameters(base)
end

# A Student-t base density is c - (ν + 1) / 2 * log1p(((x - μ₀) / σ₀)^2 / ν) for
# each observation, where c is its value at μ₀. The log1p terms are summed as one
# log of the product of the (1 + u): the same density, rounded differently.
# Other bases, and products that overflow, use the term-by-term sum.
function Distributions.logpdf(
    d::Distributions.LocationScale{<:Any,<:Any,<:PolyaTree},
    config::ConfigurationSample{<:AbstractVector{Float64}},
    z̄::Float64 = config.Z̄,
)
    μ, σ, pt = params(d)
    student_t = _student_t_parameters(pt.pt)
    (student_t === nothing || isinf(student_t[3])) && return _sum_logpdf(d, config, z̄)
    μ₀, σ₀, ν = student_t
    symmetrized = pt.pt.symmetrized
    leaf_sum = 0.0
    product = 1.0
    for z in config.configuration
        x = ((z + z̄) - μ) / σ
        x = symmetrized ? abs(x) : x
        leaf_sum += _log_prob(pt, x)
        product *= 1 + ((x - μ₀) / σ₀)^2 / ν
    end
    isfinite(product) || return _sum_logpdf(d, config, z̄)
    constant = pt.pt.J * log(2) + pt.base_logpdf(μ₀) - log(abs(σ))
    leaf_sum + length(config.configuration) * constant - (ν + 1) / 2 * log(product)
end



#function Distributions.pdf(d::Distribution, iid_sample::AbstractIIDSample)
#    prod(exp.(ULogarithmic, Distributions.logpdf.(d, iid_samples(iid_sample))))
#end 


function Empirikos.posterior(sample::AbstractIIDSample, model::PolyaTreeDistribution)
    resp = iid_samples(sample)
    offsets = copy.(model.offsets)
    _increment_offsets!(offsets, model, resp, 1.0)
    post_model = @set model.offsets = offsets
    post_model
end

function zero_offsets!(model::PolyaTreeDistribution)
    for offset in model.offsets
        fill!(offset, 0)
    end
    model
end

@inline function _increment_offsets!(offsets, model::PolyaTreeDistribution, x::Real)
    model.J == 0 && return offsets
    index = kfun(model, x, model.J)
    @inbounds for j in model.J:-1:1
        offsets[j][index] += 1
        index = cld(index, 2)
    end
    offsets
end

# σ is a multiplicative scale applied to each response before binning.
function _increment_offsets!(
    offsets,
    model::PolyaTreeDistribution,
    resp::AbstractVector,
    σ::Real,
)
    symmetrized = model.symmetrized
    @inbounds for r in resp
        x = r * σ
        _increment_offsets!(offsets, model, symmetrized ? abs(x) : x)
    end
    offsets
end

# σ is a multiplicative scale applied to each response before binning.
function posterior!(sample::AbstractIIDSample, model::PolyaTreeDistribution, σ::Real = 1.0)
    resp = iid_samples(sample)
    _increment_offsets!(model.offsets, model, resp, σ)
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
