"""
Module containing functionality for calculating statistical measures of spike trains, clusters etc.
"""
module LaskaStats

using Reexport
using LaskaCore
using LaskaCore:
                 RelativeSpikeVector,
                 SpikeVector,
                 AbstractSpikeVector,
                 Cluster,
                 RelativeCluster,
                 AbstractCluster,
                 AbstractExperiment,
                 PhyOutput,
                 RelativeSpikes
using Statistics
using FFTW
using Unitful
using DataStructures
using DSP
using Distributions
using LinearAlgebra
import SpecialFunctions: loggamma
using Optimisers
using ForwardDiff
using Turing

# Summarizing statistics
include("summarize/cv2.jl")
include("summarize/mad.jl")
include("summarize/frequency.jl")
include("summarize/relativefrequency.jl")
include("summarize/isi.jl")
include("summarize/fanofactor.jl")
export cv2, cv2mean, mad, relativefrequency, frequency, isi

include("normalize/rangenormalize.jl")
include("normalize/standardize.jl")

include("responselatency/gaussianconv.jl")
include("summarize/spikedensity.jl")

include("trainsmooth/filters.jl")
include("trainsmooth/fourier.jl")

include("corr/acg.jl")

include("similarity/LF.jl")

include("changepoint/BOCPD.jl")
include("changepoint/DSMmodels.jl")
include("changepoint/hazard.jl")
include("changepoint/omegaestimator.jl")
end
