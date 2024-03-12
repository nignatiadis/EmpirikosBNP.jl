function _default_prior(Ss::AbstractVector{<:ScaledChiSquareSample})
    _min, _max = extrema( response.(Ss))
    quantiles_to_invχ²(_min, _max)
end

function _default_prior(Ss::AbstractVector{<:ConfigurationSample})
    _default_prior(getproperty.(Ss, :S²))
end

function quantiles_to_invχ²(lower_quantile, upper_quantile)
    function f(σ²)
        myf(ν) = quantile(Empirikos.InverseScaledChiSquare(σ², ν), 0.99) - upper_quantile
        νstar = fzero( myf, 1e-6, 1e+6)
        q = quantile(Empirikos.InverseScaledChiSquare(σ², νstar), 0.01) 
        (νstar, q)
    end
    σ²star = fzero( σ² -> f(σ²)[2] - lower_quantile, lower_quantile * 1.1, upper_quantile * 0.9)
    νstar, _ = f(σ²star)
    Empirikos.InverseScaledChiSquare(σ²star, νstar)
end