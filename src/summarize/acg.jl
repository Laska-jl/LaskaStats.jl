
function acg(v::AbstractVector, bins::Int, binsize)
    out = zeros(Int64, bins)
    for val in v
    end
end

function autocov(v::AbstractVector, h)
    autocov = 0
    m = mean(v)
    for i in Iterators.take(eachindex(v), h)
        autocov += (v[i] - m) * (v[i + h] - m)
    end
    var = 0
    for i in eachindex(v)
        var += (v[i] - m)^2
    end
    var /= length(v)
    autocov /= length(v)
    return autocov / var
end
