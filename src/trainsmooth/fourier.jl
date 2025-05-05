#=
Spike train smoothing using an FFT
=#

"""
    fouriersmooth(V::AbstractVector{T}, σ, m) where {T}

Smooth an `AbstractVector` by applying a DFT to it and filtering the higher frequencies using an inverted gaussian filter.

The filter itself is calculated as:

```math
f(x) = 1 - e^{\\left (  \\frac{(x - x_{mean})^{2}}{2\\sigma^{2}}\\right )^m}
```

`σ` controls the standard deviation/width of the filter and `m` controls its "flatness".
"""
function fouriersmooth(V::AbstractVector{T}, σ, m) where {T}
    isodd(m) && throw(ArgumentError("`m` must be even."))
    normcentered = rangenormalize(V)
    premean = mean(normcentered)
    normcentered .-= mean(normcentered)
    normcentered = vcat(normcentered, reverse(normcentered))
    plan = plan_fft(normcentered)
    ft = plan * normcentered
    winf = gaussianfilter(length(ft) / 2, σ, m)
    for i in eachindex(ft)
        ft[i] *= (1 - winf(i))
    end
    rea = real.(ifft(ft))[begin:length(V)]
    rea .+= premean
    derangenormalize!(rea, minimum(V), maximum(V))
    return rea
end

function fouriersmooth(v::AbstractVector, cutoff::Integer)
    normcentered = rangenormalize(v)
    premean = mean(normcentered)
    prelength = length(v)
    normcentered .-= premean
    normcentered = vcat(normcentered, reverse(normcentered))
    plan = plan_rfft(normcentered)
    ft = plan * normcentered
    ft[cutoff:end] .= zero(eltype(ft))
    # @show length(v)
    # iplan = plan_irfft(ft, length(v))
    # rea = iplan * ft
    rea = irfft(ft, length(v) * 2)
    rea = rea[begin:prelength] 
    rea .+= premean
    derangenormalize!(rea, minimum(v), maximum(v))
    return rea
end
