module EmpirikosBNP

import Base:show, rand, empty, *, /

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
using TransformVariables
import StatsBase: var, nobs, sample, sample!, fit!

using QuadGK

include("polya.jl")
include("merge.jl")
include("defaults.jl")
include("basic_mh.jl")
include("gibbs_dp.jl")
include("gibbs_dp_convolution.jl")
include("gibbs_polya.jl")


export PolyaTreeDistribution

end
