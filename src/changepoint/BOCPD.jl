# https://arxiv.org/pdf/2302.04759
# https://github.com/maltamiranomontero/DSM-bocd/tree/main

function bocpd(data, hazard, m, grad_m, model, K = 50; verbose = false)
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
