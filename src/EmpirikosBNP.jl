module EmpirikosBNP

import Base:rand

using Distributions
import Distributions:pdf, logpdf

using Empirikos
import Empirikos:posterior


using Random
using LogarithmicNumbers
using Setfield
using StatsBase
import StatsBase: var, nobs



include("polya.jl")
include("basic_mh.jl")

export PolyaTreeDistribution

end
