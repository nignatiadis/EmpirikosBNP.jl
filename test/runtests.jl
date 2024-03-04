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
