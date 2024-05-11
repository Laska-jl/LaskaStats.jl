# https://arxiv.org/pdf/2302.04759
# https://github.com/maltamiranomontero/DSM-bocd/tree/main

function bocpd(data, hazard, m, grad_m, model, K = 50)
    dlen = length(data)
    log_R = fill(Float64, -Inf, (dlen, dlen))
    log_R[1, 1] = 0.0
    log_message = 0.0
    max_indices = [1]

    for t in 2:dlen
        haz = hazard(collect(1:min(t, K)))
        log_haz = log.(haz)
        log_1mhaz = log.(1 .- haz)

        # Predictive probability
        logpredprob = log_pred_prob(model, t, data, max_indices)

        # Growth probabilities
        log_growth_probs = logpredprob .+ log_message .+ log_1mhaz

        # Changepoint probabilities
        log_cp_prob = log(sum(exp.(logpredprob .+ log_message .+ log_haz)))

        # Calculate evidence
        new_log_joint = fill(t + 1, -Inf)
        new_log_joint[1] = log_cp_prob
        new_log_joint[max_indices .+ 1] = log_growth_probs

        # Determine run length distribution
        max_indices = sortperm(-new_log_joint)[begin:K]

        @views log_R[t:(t + 1), t] = new_log_joint
        @views log_R[t:(t + 1), t] -= log(sum(exp.(new_log_joint)))

        # Update model
        update_params!(model, m, grad_m, t, data)

        # Pass message
        log_message = new_log_joint[max_indices]
    end
    return exp.(log_R)
end
