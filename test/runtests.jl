using EmpirikosBNP
using Test
using Empirikos
using QuadGK 
using Random


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

plot(var_list, logliks_chisq)

var_iid_sample = EmpirikosBNP.VarianceIIDSample(EmpirikosBNP.IIDSample(zs_rand), Normal())
logliks_var_iid_sample = loglikelihood.(var_iid_sample, var_list)

@test argmax(logliks_var_iid_sample) == argmax(logliks_chisq)
