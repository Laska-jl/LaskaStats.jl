# Analysis of acgs

# Median firing bin
function acgmedian(acg::Vector)
    median(1:length(acg), StatsBase.weights(acg))
end

# Probability of firing within `bins` bins
function firingprob(acg::Vector, bins::Integer, nspikes::Integer)
    sum(@view(acg[begin:bins])) / nspikes
end

# Log probability of firing within `bins` bins
function logfiringprob(acg::Vector, bins::Integer, nspikes::Integer)
    log(sum(@view(acg[begin:bins]))) - log(nspikes)
end
