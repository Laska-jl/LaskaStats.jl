function find_cp(R, lag)
    changepoints = [0]
    last_cp = 0
    for i in axes(R, 2)
        candidate = i - argmax(R[:, i])
        if candidate > last_cp + lag
            if !any(changepoints .== candidate)
                push!(changepoints, candidate)
                last_cp = maximum(changepoints)
            end
            if candidate < last_cp
                pop!(changepoints)
                last_cp = maximum(changepoints)
            end
        end
    end
    return changepoints
end
