import numpy as np
from ykc_data import sigma_to_fwhm

__all__ = ["wfc3_g102", "wfc3_g141", "spherex", "R500"]

# Note that smoothspec expects resolution to be defined in terms of sigma, not FWHM
wfc3_g102 = {'name': 'wfc3_ir_g102',
             'resolution': 48.0 / sigma_to_fwhm, 'res_units': '\AA sigma',
             'dispersion': 24.5, 'disp_units': '\AA per pixel',
             'oversample': 4.,
             'fftsmooth': True, 'smoothtype': 'lambda',
             'min_wave_smooth': 0.5e4, 'max_wave_smooth':1.3e4}

wfc3_g141 = {'name': 'wfc3_ir_g141',
             'resolution': 93.0 / sigma_to_fwhm, 'res_units': '\AA sigma',
             'dispersion': 46.5, 'disp_units': '\AA per pixel',
             'oversample': 4.,
             'fftsmooth': True, 'smoothtype': 'lambda',
             'min_wave_smooth': 0.5e4, 'max_wave_smooth':2.0e4}

spherex = {'name': 'spherex',
             'resolution': 50.0 * sigma_to_fwhm, 'res_units': '\AA sigma',
             'logarithmic': True, 'oversample': 2.,
             'fftsmooth': True, 'smoothtype': 'R',
             'min_wave_smooth': 0.4e4, 'max_wave_smooth':2.5e4}

R500 = {'name': 'R500',
        'resolution': 500.0 * sigma_to_fwhm, 'res_units': '\AA sigma',
        'logarithmic': True, 'oversample': 2.,
        'fftsmooth': True, 'smoothtype': 'R',
        'min_wave_smooth': 0.35e4, 'max_wave_smooth':2.5e4}
