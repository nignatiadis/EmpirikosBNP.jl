module EmpirikosBNP

import Base:rand, empty, *, /

using Distributions
import Distributions:pdf, logpdf

using Empirikos
import Empirikos:posterior, likelihood_distribution, ScaledChiSquareSample
using LogarithmicNumbers


using ProgressMeter
using Random
using Roots
using Setfield
using StatsBase
import StatsBase: var, nobs, sample, sample!, fit!


include("defaults.jl")
include("merge.jl")
include("polya.jl")
include("basic_mh.jl")
include("gibbs_dp.jl")


export PolyaTreeDistribution

end
