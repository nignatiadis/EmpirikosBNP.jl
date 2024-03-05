module EmpirikosBNP

import Base:rand

using Distributions
import Distributions:pdf, logpdf

using Empirikos
import Empirikos:posterior, likelihood_distribution
using LogarithmicNumbers


using ProgressMeter
using Random
using Setfield
using StatsBase
import StatsBase: var, nobs, sample, sample!



include("polya.jl")
include("basic_mh.jl")

export PolyaTreeDistribution

end
