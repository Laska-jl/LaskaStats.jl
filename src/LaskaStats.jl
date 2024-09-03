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
using Optimisers
using Zygote
using Turing
import SpecialFunctions: loggamma, gamma
using TruncatedMVN
using Random
using StaticArrays
import StatsBase
using KissSmoothing
using SingularSpectrumAnalysis
import InvertedIndices: Not
import PDMats

# Utilities

# Distributions
include("distributions/invchisq.jl")

# Summarizing statistics
include("summarize/cv2.jl")
include("summarize/mad.jl")
include("summarize/frqrefactor.jl")
include("summarize/relativefrequency.jl")
include("summarize/isi.jl")
include("summarize/rhythmindex.jl")
include("summarize/acganalysis.jl")
export cv2, cv2mean, mad, relativefrequency, frequency, isi

include("normalize/rangenormalize.jl")
include("normalize/standardize.jl")

include("summarize/psth.jl")

include("trainsmooth/filters.jl")
include("trainsmooth/fourier.jl")

include("corr/acg.jl")

include("similarity/LF.jl")

include("changepoint/utils/functions.jl")
include("changepoint/hazard.jl")
include("changepoint/DSMmodels.jl")
include("changepoint/models/DSMExponentialGaussian.jl")
include("changepoint/models/DSMGaussian.jl")
include("changepoint/bocpdutils.jl")
include("changepoint/omegaestimator.jl")
include("changepoint/BOCPD.jl")

end
