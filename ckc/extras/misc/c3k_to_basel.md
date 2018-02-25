# Steps for interpolating C3K to BaSeL

1. adjust temperatures to be like BaSeL

  - T=42000 -> T=42500
  - T=47000 -> T=47500
  - T=37000 -> T=37500


2. missing logg

  - logg -0.5 -> -0.7
  - logg 0.0 -> -0.29
  - copy over from 3.5->5.5 for 2800-3200; but not overwrite actual models.  Use next lowest logg.
  - copy over *all* 5.0 -> 5.5
  - for logg=0.28: use logg=0.5 unless you have both logg=0.0 and logg=0.5 at the correct T, in which case interpolate in logg
  - for logg=0.6: use logg=0.5 unless you have both logg=1.0 and logg=0.5 at the correct T, in which case interpolate in logg
  - for logg=0.87: use logg=1.0 unless you have both logg=1.0 and logg=0.5 at the correct T, in which case interpolate in logg
  - push the lowest logg a few points lower?


3. Missing logt

  - fill in T=3350 by interpolating
  - fill in holes at fixed logT (g0 - x - g1)
  - fill in holes at fixed logg (t0 - x - t1)
