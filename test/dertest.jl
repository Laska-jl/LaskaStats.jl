
f = Figure()
ax = Axis(f[1, 1])

sm, der = LaskaStats.fouriersmoothderiv(ps[1:(end - 1)], 80, 20, 1)

lines!(ax, sm)

slide = Slider(f[2, 1], range = 1:length(sm))

derline = lift(slide.value) do val
    point = sm[val]
    [
        ((val - 1), (point - der[val])),
        ((val + 1), (point + der[val])),
    ]
end

linesegments!(ax, derline, color = :red)
