# Build a combined ckc v1.2 and dM model library at MILES + IRT resolutions
# This is done using c3k v1.2 and dM models already convolved to 
import sys, json, os
import numpy as np
import h5py

from ckc.downsample_h5 import downsample_one_h5
from ckc import libparams

sigma_to_fwhm = 2 * np.sqrt(2 * np.log(2))

# Here's the R=10K ckc lib
ckc_R10K_hname = '../lores/ckc_R10K.h5'

# build the dM lib at R=10K (FWHM)
dM_fullres_hname = '../h5/ckc14_dMall.full.h5'
dM_R10K_hname = '../lores/dMall_R10K.h5'
if not os.path.exists(dM_R10K_hname):
    wave, spectra, params = downsample_one_h5(dM_fullres_hname, **libparams.R10K)
    with h5py.File(dM_R10K_hname, "w") as f:
        wave = f.create_dataset('wavelengths', data=wave)
        spectra = f.create_dataset('spectra', data=spectra)
        par = f.create_dataset('parameters', data=params)
        for k, v in list(libparams.R10K.items()):
            f.attrs[k] = json.dumps(v)

#sys.exit()

# combine ckc and dM
# assumes wavelength scale is exactly the same
combined_hname = '../lores/ckc+dMall_R10K.h5'
if not os.path.exists(combined_hname):
    with h5py.File(ckc_R10K_hname,'r') as ckc, h5py.File(dM_R10K_hname,'r') as dm, h5py.File(combined_hname,'w') as combined:
        assert ckc['wavelengths'].shape == dm['wavelengths'].shape
        assert np.allclose(ckc['wavelengths'][:], dm['wavelengths'][:])

        wave = combined.create_dataset('wavelengths', data=ckc['wavelengths'][:])
        params = combined.create_dataset('parameters', data=np.concatenate([ckc['parameters'], dm['parameters']]))
        spectra = combined.create_dataset('spectra', data=np.vstack([ckc['spectra'], dm['spectra']]))

#sys.exit()
    
# smoothing parameters.
# for manga we use the quadrature difference between 2.355 \AA FWHM and 5500AA/10,000 = 0.55\AA
rtarg = 2.998e5 / (50.0 * sigma_to_fwhm)
delta_R = 1/(np.sqrt(1/rtarg)**2 - 1e-4**2) * sigma_to_fwhm
manga_pars = {'fftsmooth': True,
              'logarithmic': True,
              'max_wave_smooth': 10500.0,
              'min_wave_smooth': 3600.0,
              'name': 'R10K_to_MaNGA-sigma50kms',
              'oversample': 1.0,
              'res_units': '\\lambda/\\sigma_\\lambda',
              'resolution': delta_R,
              'smoothtype': 'R'}

# for UV we use the quadrature difference between R=10,000 and R=300
delta_R = 1./np.sqrt((1/300.)**2 - 1e-4**2) * sigma_to_fwhm
uv_pars = {'fftsmooth': True,
           'logarithmic': True,
           'min_wave_smooth': 91,
           'max_wave_smooth': manga_pars['min_wave_smooth'] - 1.0,
           'name': 'R10K_to_UV-R300K',
           'oversample': 1.0,
           'res_units': '\\lambda/\\sigma_\\lambda',
           'resolution': delta_R,
           'smoothtype': 'R'}

# for IR we use the quadrature difference between R=10,000 and R=300
delta_R = 1./np.sqrt((1/300.)**2 - 1e-4**2) * sigma_to_fwhm
ir_pars = {'fftsmooth': True,
           'logarithmic': True,
           'min_wave_smooth': manga_pars['max_wave_smooth'] + 1.0,
           'max_wave_smooth': 1e7,
           'name': 'R10K_to_IR-R300K',
           'oversample': 1.0,
           'res_units': '\\lambda/\\sigma_\\lambda',
           'resolution': delta_R,
           'smoothtype': 'R'}


# Downsample ckc+dM/manga
mwave, mspec, params = downsample_one_h5(combined_hname, **manga_pars)
# Downsample ckc+dM/UV
uwave, uspec, params = downsample_one_h5(combined_hname, **uv_pars)
# Downsample ckc+dM/IR
iwave, ispec, params = downsample_one_h5(combined_hname, **ir_pars)

# combine and write
combined_lores_hname = '../lores/manga/c3k+dMall_manga-sigma50kms.h5'
wave = np.concatenate([uwave, mwave, iwave])

out = h5py.File(combined_lores_hname, 'w')
w = out.create_dataset('wavelengths', data=wave)
cspec = out.create_dataset('spectra', data=np.hstack([uspec, mspec, ispec]))
cpars = out.create_dataset('parameters', data=params)
out.attrs['MaNGA_info'] = json.dumps(manga_pars)
out.attrs['IR_info'] = json.dumps(ir_pars)
out.attrs['UV_info'] = json.dumps(uv_pars)
out.close()

