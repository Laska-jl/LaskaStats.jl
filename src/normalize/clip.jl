# Normalize by clipping values

"""
    featureclip!(v::AbstractVector{T}, operator::Function, val::T) where T

Clip the `Vector` `v`, replacing all values for which `operator(v[i], val)` returns `true` with `val`.

# Examples

```julia-repl
# Clip all values below 5.0
julia> x = [-11.193910798661717, -4.435118347593609, 9.248557345053548, -16.05993315199367, 6.022329579245262, -12.183162249713567, 14.146889820135373, 9.131280599825487, 17.268599084940032, 7.049543783905182]
10-element Vector{Float64}:
 -11.193910798661717
  -4.435118347593609
   ⋮
  17.268599084940032
   7.049543783905182

julia> LaskaStats.featureclip!(x, <, 5.0)

julia> @show x
x = [5.0, 5.0, 9.248557345053548, 5.0, 6.022329579245262, 5.0, 14.146889820135373, 9.131280599825487, 17.268599084940032, 7.049543783905182]
10-element Vector{Float64}:
  5.0
  5.0
  ⋮
 17.268599084940032
  7.049543783905182

# Clip all values above 5.0
julia> x = [-11.193910798661717, -4.435118347593609, 9.248557345053548, -16.05993315199367, 6.022329579245262, -12.183162249713567, 14.146889820135373, 9.131280599825487, 17.268599084940032, 7.049543783905182]
10-element Vector{Float64}:
 -11.193910798661717
  -4.435118347593609
   ⋮
  17.268599084940032
   7.049543783905182

julia> LaskaStats.featureclip!(x, >, 5.0)

julia> @show x
x = [-11.193910798661717, -4.435118347593609, 5.0, -16.05993315199367, 5.0, -12.183162249713567, 5.0, 5.0, 5.0, 5.0]
10-element Vector{Float64}:
 -11.193910798661717
  -4.435118347593609
   ⋮
   5.0
   5.0
```
"""
function featureclip!(v::AbstractVector{T}, operator::Function, val::T) where T
    for i in eachindex(v)
        v[i] = operator(v[i], val) ? val : v[i]
    end
end
