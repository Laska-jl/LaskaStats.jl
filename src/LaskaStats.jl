module LaskaStats

using LaskaCore
using Statistics

# Summarizing statistics
include("summarize/cv2.jl")
include("summarize/mad.jl")
include("summarize/frequency.jl")
include("summarize/relativefrequency.jl")
include("summarize/isi.jl")
export cv2, cv2mean, mad, relativefrequency, frequency, isi

end
