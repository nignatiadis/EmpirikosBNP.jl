using EmpirikosBNP
using Test
using Empirikos
using QuadGK 
using Random
using Distributions


Random.seed!(1)


# Symmetrized Normal Base

pt = PolyaTreeDistribution(base=Empirikos.fold(Normal(0.0, 1.0)), 
    J=5, α=10.0, 
    symmetrized=true,
    median_centered=false)

polyatree = rand(pt)

@test Float64(quadgk(x -> pdf(polyatree, x), -Inf, Inf)[1]) ≈ 1.0 atol = 1e-6
@test Float64(quadgk(x -> x*pdf(polyatree, x), -Inf, Inf)[1]) ≈ 0.0 atol = 1e-10
@test Float64(quadgk(x -> x^2*pdf(polyatree, x), -Inf, Inf)[1]) ≈ var(polyatree) atol = 1e-6

# Check a couple things about the computation of the variance.

J = polyatree.pt.J
base = polyatree.pt.base
qs = quantile.(base, collect((0:(2^J))/ 2^J))

@test qs == [0.0; EmpirikosBNP._grid_points(polyatree.pt); Inf]
@test length(polyatree.θs) == J
@test length(qs) == 2^J + 1
function sum_polya_tree(probs)
    # Start with probability 1 at the root
    level_probs = [1.0]
    
    for level in probs
        new_probs = []
        for (i, p) in enumerate(level_probs)
            push!(new_probs, p * level[2i-1])  # left child
            push!(new_probs, p * level[2i])    # right child
        end
        level_probs = new_probs
    end
    
    return sum(level_probs)
end

@test sum_polya_tree(  polyatree.θs ) ≈ 1.0# should be 1.0

norm_tdist = Empirikos.Folded(TDist(8)/std(TDist(8)))
@test EmpirikosBNP.∫x²dP(norm_tdist, 0.2 , 1.0) ≈ quadgk(x -> x^2 * pdf(norm_tdist, x), 0.2, 1.0)[1]



ptt = PolyaTreeDistribution(base=Empirikos.fold(TDist(5)), 
    J=7, α=2.0, 
    symmetrized=true,
    median_centered=false)


polyatreet = rand(ptt)

@test Float64(quadgk(x -> pdf(polyatreet, x), -Inf, Inf)[1]) ≈ 1.0 atol = 1e-6
@test Float64(quadgk(x -> x*pdf(polyatreet, x), -Inf, Inf)[1]) ≈ 0.0 atol = 1e-10
@test Float64(quadgk(x -> x^2*pdf(polyatreet, x), -Inf, Inf)[1]) ≈ var(polyatreet) atol = 1e-6
@test std(polyatreet) == sqrt(var(polyatreet))


ptt_8 = PolyaTreeDistribution(base=Empirikos.fold(TDist(8)), 
    J=7, α=2.0, 
    symmetrized=true,
    median_centered=false)


polyatreet_8 = rand(ptt_8)

@test Float64(quadgk(x -> pdf(polyatreet_8, x), -Inf, Inf)[1]) ≈ 1.0 atol = 1e-4
@test Float64(quadgk(x -> x*pdf(polyatreet_8, x), -Inf, Inf)[1]) ≈ 0.0 atol = 1e-10
@test Float64(quadgk(x -> x^2*pdf(polyatreet_8, x), -Inf, Inf)[1]) ≈ var(polyatreet_8) atol = 1e-5
@test std(polyatreet_8) == sqrt(var(polyatreet_8))


ptt_scale = PolyaTreeDistribution(base=Empirikos.fold(TDist(5) / std(TDist(5))), 
    J=7, α=2.0, 
    symmetrized=true,
    median_centered=false)
polyatreet_scale = rand(ptt_scale)

@test Float64(quadgk(x -> pdf(polyatreet_scale, x), -Inf, Inf)[1]) ≈ 1.0 atol = 1e-6
@test Float64(quadgk(x -> x*pdf(polyatreet_scale, x), -Inf, Inf)[1]) ≈ 0.0 atol = 1e-10
@test Float64(quadgk(x -> x^2*pdf(polyatreet_scale, x), -Inf, Inf)[1]) ≈ var(polyatreet_scale) atol = 1e-6



    
## test for VarianceIIDSample

Random.seed!(1)

zs_rand = rand(Normal(0.0, 1.0), 100)

chisq = ScaledChiSquareSample(mean(abs2, zs_rand), length(zs_rand))

var_list = 0.85:0.01:1.15

logliks_chisq = loglikelihood.(chisq, var_list)

#plot(var_list, logliks_chisq)

var_iid_sample = EmpirikosBNP.VarianceIIDSample(EmpirikosBNP.IIDSample(zs_rand), Normal())
logliks_var_iid_sample = loglikelihood.(var_iid_sample, var_list)

@test argmax(logliks_var_iid_sample) == argmax(logliks_chisq)

diff_1 = logliks_chisq[1] - logliks_var_iid_sample[1]

@test all(logliks_chisq - logliks_var_iid_sample .≈ diff_1)
# More on VarianceIIDSample

single_var = 5.0

@test sum(logpdf.(Normal(0,sqrt(single_var)), zs_rand)) ≈ loglikelihood(var_iid_sample, single_var)

t_samples = EmpirikosBNP.VarianceIIDSample(EmpirikosBNP.IIDSample(zs_rand), TDist(8))
@test sum(logpdf.(TDist(8)*sqrt(single_var), zs_rand)) ≈ loglikelihood(t_samples, single_var)
@test prod(pdf.(TDist(8), zs_rand ./ sqrt(single_var))) / single_var^(length(zs_rand)/2) ≈ exp(loglikelihood(t_samples, single_var))

#kfun(base::Distribution, x::AbstractFloat, j::Int) = min(floor(Int, 2^j * cdf(base, x)) + 1, 2^j)
#_ns(base::Distribution, J::Int, x::AbstractVector) = map(j -> counts(kfun.(base, x, j), 1:2^j), 1:J)


#kfun(polya::PolyaTreeDistribution, x::AbstractFloat, j::Int) = kfun(polya.base, x, j)
#_ns(polya::PolyaTreeDistribution, x::AbstractVector) = _ns(polya.base, polya.J, x)

pt = PolyaTreeDistribution(base=Empirikos.fold(TDist(5)), 
    J=5, α=10.0, 
    symmetrized=true,
    median_centered=false)


@test EmpirikosBNP.kfun(pt.base, 0.0, 1) == 1#1
@test EmpirikosBNP.kfun(pt.base, 0.8, 1) == 2 #2
@test EmpirikosBNP.kfun(pt.base, quantile(pt.base, 0.5), 1) == 2
@test EmpirikosBNP.kfun(pt.base, Inf, 1) == 2


@test EmpirikosBNP.kfun(pt, 0.0, 1) == 1#1
@test EmpirikosBNP.kfun(pt, 0.8, 1) == 2 #2
@test EmpirikosBNP.kfun(pt, quantile(pt.base, 0.5), 1) == 2
@test EmpirikosBNP.kfun(pt, Inf, 1) == 2



@test EmpirikosBNP.kfun(pt.base, quantile(pt.base, 0.5), 2) == 3
@test EmpirikosBNP.kfun(pt.base, quantile(pt.base, 0.25), 2) == 2
@test EmpirikosBNP.kfun(pt.base, quantile(pt.base, 0.24), 2) == 1
@test EmpirikosBNP.kfun(pt.base, quantile(pt.base, 0.75), 2) == 4
@test EmpirikosBNP.kfun(pt.base, Inf, 2) == 4

@test EmpirikosBNP.kfun(pt, quantile(pt.base, 0.5), 2) == 3
@test EmpirikosBNP.kfun(pt, quantile(pt.base, 0.25), 2) == 2
@test EmpirikosBNP.kfun(pt, quantile(pt.base, 0.24), 2) == 1
@test EmpirikosBNP.kfun(pt, quantile(pt.base, 0.75), 2) == 4
@test EmpirikosBNP.kfun(pt, Inf, 2) == 4



Zs = randexp(1000)

ns1 = EmpirikosBNP._ns(pt.base, 5, Zs)
ns2 = EmpirikosBNP._ns(pt, Zs)

@test ns1 == ns2


#@btime EmpirikosBNP._ns($(pt.base), $(5), $(Zs))
#@btime EmpirikosBNP._ns($(pt), $(Zs))



# Test p-value computation 

my_config_sample = EmpirikosBNP.ConfigurationSample(EmpirikosBNP.IIDSample([12.0340292912603, 3.990445548408448, -3.6786150785477068, 14.289772526102656, -14.826286689345718, 7.246581950355288, 10.141984086148856, -14.761417934550533, 17.590823315956502, -8.30338731871067, -8.856670510285603, -14.867259186791806]))
my_mu_hat = -5.09987303940544

_p1 = EmpirikosBNP._pval_custom(my_config_sample, my_mu_hat, 1.0, Uniform(-20,20); rtol=0.0001)
_p2 = EmpirikosBNP._pval_custom(my_config_sample, my_mu_hat, std(Uniform(-20,20)), Uniform(-20,20) / std( Uniform(-20,20) ); rtol=0.0001)

@test _p1 ≈ _p2

# We can also compute the p-value directly here by noting that the conditional density of Z_bar given the configuration
# is just uniform on a grid that we can determine (below using brute force)
_grid = -10:0.001:10
_logpdfs = [logpdf(Uniform(-20,20), my_config_sample, u) for u in _grid]

_idx_all = findall(_logpdfs .> -Inf)

l=_grid[minimum(_idx_all)]
u=_grid[maximum(_idx_all)]

@test (my_mu_hat - l) /(u-l) ≈ _p1 atol=1e-4


# Oracle p-value computation: avoid p-values above 1 due to numerical error

my_config_sample = EmpirikosBNP.ConfigurationSample(EmpirikosBNP.IIDSample([0.1972321182792515, -0.3108964672024414, 0.4300037014906913, 0.1825837138583747, -0.49892306642587597]))
my_mu_hat = -1.9295690823362132e-5
@test EmpirikosBNP._pval_custom(my_config_sample, my_mu_hat, 1.0, PGeneralizedGaussian(0, 1, 3); rtol=0.01) <= 1.0



