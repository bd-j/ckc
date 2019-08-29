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

- `make_sed.py` Methods for downsampling and combining the full resolution C3K .spec files with the flux files (and blackbodies at longer wavelengths) to provide full wavelength range SEDs as HDF5 files (for a given feh/afe combo).  

   The downsampling options are given as a list of tuples with each tuple describing a single wavelength segment.
   ```python
   segments = [(min_wave, max_wave, resolution_fwhm, use_fft),
               (910, 3400., 1000, False),
               (3400., 7000., 5000, True)]
   ```

- `sed_to_fsps.py`  Methods to interpolate SEDs produced by `make_sed.py` to the BaSeL grid in logt-logg used by FSPS and store in a new HDF5 file

- `make_binary.py` Convert the HDF5 output of `sed_to_fsps.py` to binary files and ancillary information files suitable for FSPS.

- `ckc_to_fsps.py` Methods that wrap the SED making and interpolation to FSPS methods, and runs them for a set of metalllcities with a given set of downsampling 
