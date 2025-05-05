#=
Filters for use in fourier smoothing.
=#

function gaussianfilter(mean, σ, m = 2)
    sigm = (1 / 2σ^2)
    x -> exp(-(((x - mean)^2) * sigm)^m)
end
