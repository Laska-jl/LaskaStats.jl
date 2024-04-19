cr = getcluster(rel, 151)
cr2 = getcluster(rel, 158)

v1 = SpikeVector(cr.spiketimes[1], 30000)
v2 = SpikeVector(cr2.spiketimes[1], 30000)

LaskaStats.LF_burstsimilarity(
    v1, v2,
       20u"ms",
    3,
       1,
       20u"ms"
       )

LaskaStats.LF_silence_similarity(v1, v2, 300)

LaskaStats.LF_difference(v1, v2, 20u"ms", 3, 1, 20u"ms", 40u"ms")
