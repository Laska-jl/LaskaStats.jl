using LaskaStats
include("./../../paths.jl")

res = importphy(PATH_TO_PHYOUTPUT, PATH_TO_GLXMETA, PATH_TO_TRIGGERCHANNEL)

rel = relativespikes(res, Dict("US" => 0, "CS" => 300), 1000, 1500)

cr = getcluster(rel, 151)

spikes, time = LaskaStats.psth(cr.spiketimes, 500)

pop!(spikes)

spikes_scaled = LaskaStats.zstandardize(spikes)

# m and grad_m

mean_mu0 = 0.0
var_mu0 = 1.0

mean_Sigma0 = 10.0
var_Sigma0 = 1.0

mu0 = reshape([mean_mu0 / var_mu0, 1.0 / var_mu0], (2, 1))

Sigma0 = Matrix(1.0I, 2, 2)
Sigma0[1, 1] = mean_Sigma0 / var_Sigma0
Sigma0[2, 2] = 1 / var_Sigma0
