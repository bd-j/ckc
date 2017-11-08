C3K utilities
===

- `make_full_h5.py` Create HDF5 files of the full resolution R=300,000 C3K .spec files located on odyssey

- `downsample_h5.py`  Methods for downsampling the full resolution C3K spectra stored in HDF5 files produced by `make_full_h5.py` to other resolutions


Smoothing or downsampling is defined by a dictionary of parameters
```python
pars = {'min_wave_smooth':0.0,
        'max_wave_smooth':np.inf,
        'dispersion':1.0,
        'oversample':2.0,
        'resolution':1e4, # The output resolution, in terms of lambda/fwhm
        'logarithmic':False,
        }
```
