import numpy as np


__all__ = ["sigma_to_fwhm", "intrinsic_params", "R10K", "R500"]

ckms = 2.998e5
sigma_to_fwhm = 2.355
Rckc = 2e5 # in FWHM?  This is the wavelength spacing
wlims_ckc = (900, 40e4) # anstroms

# Note that smoothspec expects resolution to be defined in terms of sigma, not FWHM

intrinsic_params = {'name': 'intrinsic', 'version': 'CKC v1.2',
                    'resolution': 1e5, 'res_units': '\lambda/FWHM_\lambda',
                    'min_wave_smooth': 900, 'max_wave_smooth':40e4}

# Account for intrinsic resolution
rfwhm = 1/np.sqrt(1/1e4**2 - 1/Rckc**2)
R10K = {'name': 'R10K',
        'resolution': rfwhm * sigma_to_fwhm, 'res_units': '\lambda/\sigma_\lambda',
        'logarithmic': True, 'oversample': 2.,
        'fftsmooth': True, 'smoothtype': 'R',
        'min_wave_smooth': 0.35e4, 'max_wave_smooth':3e4}

rfwhm = 1/np.sqrt(1.0/500**2 - 1.0/Rckc**2)
R500 = {'name': 'R500',
        'resolution': rfwhm * sigma_to_fwhm, 'res_units': '\lambda/\sigma_\lambda',
        'logarithmic': True, 'oversample': 2.,
        'fftsmooth': True, 'smoothtype': 'R',
        'min_wave_smooth': 910., 'max_wave_smooth':35e4}
