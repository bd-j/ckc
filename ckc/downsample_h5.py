# This module contains methods for downsampling the full R=200K C3K spectra to
# other resolutions.

import sys, time, gc
import json
import numpy as np
import h5py
#from ykc_data import sigma_to_fwhm
try:
    from bsfh.utils import smoothing
except(ImportError):
    from prospect.utils import smoothing

from libparams import *


__all__ = ["construct_grism_outwave", "downsample_one_h5", "downsample_all_h5"]

h5dir_default = '/Users/bjohnson/code/ckc/ckc/data/h5/'


def construct_grism_outwave(min_wave_smooth=0.0, max_wave_smooth=np.inf,
                            dispersion=1.0, oversample=2.0,
                            resolution=3e5, logarithmic=False,
                            **extras):
    """Given parameters describing the output spectrum, generate a wavelength
    grid that properly samples the resolution.
    """
    if logarithmic:
        dlnlam = 1.0/resolution/2/oversample  # critically sample the resolution
        lnmin, lnmax = np.log(min_wave_smooth), np.log(max_wave_smooth)
        #print(lnmin, lnmax, dlnlam, resolution, oversample)
        out = np.exp(np.arange(lnmin, lnmax + dlnlam, dlnlam))
    else:
        out = np.arange(min_wave_smooth, max_wave_smooth, dispersion / oversample)
    return out    


def downsample_one_h5(fullres_hname, resolution=1.0, **conv_pars):
    """Read one full resolution h5 file, downsample every spectrum in
    that file, and return the result as ndarrays
    """
    outwave = construct_grism_outwave(resolution=resolution, **conv_pars)
    #print(resolution, len(outwave), conv_pars['smoothtype'])
    with h5py.File(fullres_hname, 'r') as fullres:
        params = np.array(fullres['parameters'])
        whires = np.array(fullres['wavelengths'])
        flores = np.zeros([len(params), len(outwave)])
        for i, p in enumerate(params):
            fhires = fullres['spectra'][i, :]
            s = smoothing.smoothspec(whires, fhires, resolution,
                                     outwave=outwave, **conv_pars)
            flores[i, :] = s
    gc.collect()
    return outwave, flores, params


class function_wrapper(object):
    """A hack to make the downsampling function pickleable for MPI.
    """
    def __init__(self, function, function_kwargs):
        self.function = function
        self.kwargs = function_kwargs

    def __call__(self, args):
        return self.function(*args, **self.kwargs)


def downsample_all_h5(conv_pars, pool=None,
                       zlist = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]):
    h5dir = conv_pars.get('h5dir', h5dir_default)
    htemplate = h5dir + '/ckc_feh={:+3.2f}.full.h5'
    hnames = [[htemplate.format(z)] for z in zlist]

    downsample_with_pars = function_wrapper(downsample_one_h5, conv_pars)
    
    if pool is not None:
        M = pool.map
    else:
        M = map

    results = M(downsample_with_pars, hnames)
    wave = results[0][0]
    spectra = np.vstack([r[1] for r in results])
    params = np.concatenate([r[2] for r in results])

    outdir = conv_pars.get('outdir', 'lores')
    outname = '{}/ckc_{}.h5'.format(outdir, conv_pars['name'])
    with h5py.File(outname, "w") as f:
        wave = f.create_dataset('wavelengths', data=wave)
        spectra = f.create_dataset('spectra', data=spectra)
        par = f.create_dataset('parameters', data=params)
        for k, v in list(conv_pars.items()):
            f.attrs[k] = json.dumps(v)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        grism = sys.argv[1]
        conv_pars = globals()[grism]
    else:
        conv_pars = R500

    pool = None
    import multiprocessing
    zlist = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.25]
    nproc = min([6, len(zlist)])
    pool = multiprocessing.Pool(nproc)
    downsample_all_h5(conv_pars, zlist=zlist, pool=pool)
    try:
        pool.close()
    except:
        pass


