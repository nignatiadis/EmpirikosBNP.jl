module EmpirikosBNP

import Base:rand

using Distributions
import Distributions:pdf

using Empirikos
import Empirikos:posterior


using Random
using LogarithmicNumbers
using Setfield
using StatsBase



include("polya.jl")

export PolyaTreeDistribution

end
