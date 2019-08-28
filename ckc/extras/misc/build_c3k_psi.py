# Build a combined ckc v1.2 and dM model library at MILES + IRT resolutions
# This is done using c3k v1.2 and dM models already convolved to 
import sys, json, os
import numpy as np
import h5py

from downsample_h5 import downsample_one_h5
import libparams

# Here's the R=10K ckc lib
ckc_R10K_hname = 'lores/ckc_R10K.h5'

# build the dM lib at R=10K (FWHM)
dM_fullres_hname = 'h5/ckc14_dMall.full.h5'
dM_R10K_hname = 'lores/dMall_R10K.h5'
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
combined_hname = 'lores/ckc+dMall_R10K.h5'
if True: #not os.path.exists(combined_hname):
    with h5py.File(ckc_R10K_hname,'r') as ckc, h5py.File(dM_R10K_hname,'r') as dm, h5py.File(combined_hname,'w') as combined:
        assert ckc['wavelengths'].shape == dm['wavelengths'].shape
        assert np.allclose(ckc['wavelengths'][:], dm['wavelengths'][:])

        wave = combined.create_dataset('wavelengths', data=ckc['wavelengths'][:])
        params = combined.create_dataset('parameters', data=np.concatenate([ckc['parameters'], dm['parameters']]))
        spectra = combined.create_dataset('spectra', data=np.vstack([ckc['spectra'], dm['spectra']]))

#sys.exit()
    
# smoothing parameters.
# for miles we use the quadrature difference between 2.54 \AA FWHM and 5500AA/10,000 = 0.55\AA
delta_sigma = np.sqrt(2.55**2 - 0.55**2) / 2.35
miles_pars = {'fftsmooth': True,
              'logarithmic': False,
              'max_wave_smooth': 7500.0,
              'min_wave_smooth': 3500.0,
              'name': 'R10K_to_MILES-2.55AAFWHM',
              'oversample': 2.0,
              'res_units': '\\sigma_\\lambda',
              'resolution': delta_sigma,
              'smoothtype': 'lambda'}

# for irtf we use the quadrature difference between R=10,000 and R=2,000
delta_R = 1./np.sqrt(5e-4**2 - 1e-4**2) * 2.35
irtf_pars = {'fftsmooth': True,
              'logarithmic': True,
              'max_wave_smooth': 3e4,
              'min_wave_smooth': 7501,
              'name': 'R10K_to_IRTF-R2K',
              'oversample': 2.0,
              'res_units': '\\lambda/\\sigma_\\lambda',
              'resolution': delta_R,
              'smoothtype': 'R'}

# Downsample ckc+dM/miles
mwave, mspec, params = downsample_one_h5(combined_hname, **miles_pars)
# Downsample ckc+dM/irtf
iwave, ispec, params = downsample_one_h5(combined_hname, **irtf_pars)

# combine and write
combined_lores_hname = 'lores/irtf/ckc+dMall_miles+irtf.h5'
wave = np.concatenate([mwave, iwave])

out = h5py.File(combined_lores_hname, 'w')
w = out.create_dataset('wavelengths', data=wave)
cspec = out.create_dataset('spectra', data=np.hstack([mspec, ispec]))
cpars = out.create_dataset('parameters', data=params)
out.attrs['MILES_info'] = json.dumps(miles_pars)
out.attrs['IRTF_info'] = json.dumps(irtf_pars)
out.close()

# make version with same parameters as fsps version
fpars = np.zeros(len(params), dtype=np.dtype([(d, '<f8') for d in ['Z', 'logg', 'logt']]))
fpars['Z'] = 0.0134 * 10**params['feh']
fpars['logg'] = params['logg']
fpars['logt'] = params['logt']
out = h5py.File('lores/irtf/ckc+dMall_miles+irtf.forpsi.h5', 'w')
w = out.create_dataset('wavelengths', data=wave)
cspec = out.create_dataset('spectra', data=np.hstack([mspec, ispec]))
cpars = out.create_dataset('parameters', data=fpars)
out.attrs['MILES_info'] = json.dumps(miles_pars)
out.attrs['IRTF_info'] = json.dumps(irtf_pars)
out.close()


