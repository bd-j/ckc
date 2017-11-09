C3K utilities
===

- `make_full_h5.py` Create HDF5 files of the full resolution R=300,000 C3K .spec files located on odyssey, or of the lower resolution .flux files.  There will be one file for each *feh* and *afe* value.

- `downsample_h5.py`  Methods for downsampling the full resolution C3K spectra stored in HDF5 files produced by `make_full_h5.py` to other resolutions.  The output is a single file containing all spectra at all metallicities. This contains two different parallelization schemes: one parallelizes over metallicities and loops over spectra, the other (the preferred and default method) loops over metallicities and parallelizes over spectra in each metallicity, dynamically resizing the HDF5 output arrays to add the convolved spectra as they are produced at each metalllicity

 Smoothing or downsampling is defined by a dictionary of parameters:
```python
pars = {'min_wave_smooth':900.,
        'max_wave_smooth':25000,
        'dispersion':1.0,
        'oversample':2.0, # The number of pixels per resolution element
        'resolution':1e4, # The output resolution, in terms of lambda/fwhm
        'logarithmic':False}
```
