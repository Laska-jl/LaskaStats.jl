#=
Functions for generating functions
=#

# `m`

function generate_standard_m(size::Integer)
    size
    # f = function (x::AbstractVector{T}) where {T}
    #     d = [(1.0 + x[i]^2)^(-0.5) for i in 1:size]
    #     diagm(d)
    # end
    # return f
end
