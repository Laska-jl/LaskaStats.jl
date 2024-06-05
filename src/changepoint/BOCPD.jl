# https://arxiv.org/pdf/2302.04759
# https://github.com/maltamiranomontero/DSM-bocd/tree/main

"""
    bocpd(
        data::AbstractVector,
        hazard::AbstractHazard,
        m::Function,
        grad_m::Function,
        model::AbstractDSMModel,
        K::Integer = 50;
        verbose::Bool = false)

Function for detecting changepoints in `data`.

Uses Diffusion Score Matching Bayesian Online Changepoint Detection (DSM-BOCD) described in: https://proceedings.mlr.press/v202/altamirano23a.html.                 
Original Python implementation by the authors of algorithm can be found at: https://github.com/maltamiranomontero/DSM-bocd?tab=readme-ov-file

# Arguments

- `data`: Vector of data in which to find changepoints.
- `hazard`: Hazard function providing a changepoint prior. See [`LaskaStats.ConstantHazard`](@ref) for an example.
- `m`: 


"""
function bocpd(
        data::AbstractVector,
        hazard::AbstractHazard,
        m::Function,
        grad_m::Function,
        model::AbstractDSMModel,
        K::Integer = 50;
        verbose::Bool = false)
    dlen = length(data)
    log_R = fill(-Inf, (dlen + 1, dlen + 1))
    log_R[1, 1] = 0.0
    log_message = 0.0
    max_indices::Vector{Int64} = [1]

    for t in 2:dlen
        if verbose && t % 100 == 0
            println("processing observation #$(t)")
        end
        haz = hazard(collect(1:(min(t, K) - 1)))
        log_haz = log.(haz)
        log_1mhaz = log.(1 .- haz)

        # Predictive probability
        logpredprob = log_pred_prob(model, t, data, max_indices)

        # Growth probabilities
        log_growth_probs = logpredprob .+ log_message .+ log_1mhaz

        # Changepoint probabilities
        log_cp_prob = log(sum(exp.(logpredprob .+ log_message .+ log_haz)))

        # Calculate evidence
        new_log_joint = fill(-Inf, t)
        new_log_joint[1] = log_cp_prob
        new_log_joint[max_indices .+ 1] = log_growth_probs

        # Determine run length distribution
        max_indices = sortperm(-new_log_joint)[begin:min(length(new_log_joint), K - 1)]

        log_R[begin:(t), t] = new_log_joint
        log_R[begin:(t), t] .-= log(sum(exp.(new_log_joint)))

        # Update model
        update_params!(model, m, grad_m, t, data)

        # Pass message
        log_message = new_log_joint[max_indices]
    end
    return exp.(log_R)
end
