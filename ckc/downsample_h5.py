# This module contains methods for downsampling the full R=200K C3K spectra to
# other resolutions.

import sys, time, gc
import json
import numpy as np
from functools import partial
from itertools import imap
import multiprocessing

import h5py
#from ykc_data import sigma_to_fwhm
from prospect.utils import smoothing

from libparams import *


__all__ = ["construct_grism_outwave", "downsample_one_h5", "downsample_all_h5"]


def construct_grism_outwave(min_wave_smooth=0.0, max_wave_smooth=np.inf,
                            dispersion=1.0, oversample=2.0,
                            resolution=3e5, logarithmic=False,
                            **extras):
    """Given parameters describing the output spectrum, generate a wavelength
    grid that properly samples the resolution.
    """
    if logarithmic:
        # critically sample the resolution
        dlnlam = 1.0 / resolution / oversample  
        lnmin, lnmax = np.log(min_wave_smooth), np.log(max_wave_smooth)
        #print(lnmin, lnmax, dlnlam, resolution, oversample)
        out = np.exp(np.arange(lnmin, lnmax + dlnlam, dlnlam))
    else:
        out = np.arange(min_wave_smooth, max_wave_smooth,
                        dispersion / oversample)
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


def downsample_all_h5(pool=None, zlist=[-4.0, -3.0, -2.0, -1.0, 0.0],
                      htemp='ckc_feh={:+3.2f}.full.h5',
                      **conv_pars):

    if pool is not None:
        M = pool.imap
    else:
        M = imap

    hnames = [htemp.format(z) for z in zlist]
    #downsample_with_pars = function_wrapper(downsample_one_h5, conv_pars)
    downsample_with_pars = partial(downsample_one_h5, **conv_pars)
    mapper = M(downsample_with_pars, hnames)

    outdir = conv_pars.get('outdir', 'lores')
    outname = '{}/{}.h5'.format(outdir, conv_pars['name'])

    for i, result in enumerate(mapper):
        if i == 0:
            out, spectra, par = initialize_h5(outname, *result)
            #wave = out.create_dataset('wavelengths', data=result[0])
        nmod, nw = spectra.shape
        nnew = len(result[2])
        spectra.resize(nmod + nnew, axis=0)
        spectra[nmod:, :] = result[1]
        par.resize(nmod + nnew, axis=0)
        par[nmod:] = result[2]
        out.flush()

    for k, v in list(conv_pars.items()):
        out.attrs[k] = json.dumps(v)
    out.close()

        
def initialize_h5(name, wave, spec, par):
    out = h5py.File(name, "w")
    nmod, nw = spec.shape
    spectra = out.create_dataset('spectra', shape=(0, nw),
                                 maxshape=(None, nw))
    par = out.create_dataset('parameters', shape=(0,),
                             maxshape=(None,), dtype=par.dtype)
    wave = out.create_dataset('wavelengths', data=wave)
    out.flush()
    return out, spectra, par


if __name__ == "__main__":

    #zlist = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.25]
    #zlist = [-2.5, -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
    #         0.0, 0.25, 0.50, 0.75]
    zlist = [-4.0, -3.5, -3.0, -2.75, -2.5, -2.25, -2.0,
             -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
             0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    zlist = [0.0]
        
    htemp_default = '/Users/bjohnson/code/ckc/ckc/h5/ckc_feh={:+3.2f}.full.h5'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='R500',
                        help=("Name of dictionary that describes the "
                              "output spectral parameters"))
    parser.add_argument("--np", type=int, default=6,
                        help="number of processors")
    parser.add_argument("--hname", type=str, default=htemp_default,
                        help=("A string that gives the full reolution "
                              "h5 file template."))
    args = parser.parse_args()


    conv_pars = globals()[args.config]
    ncpu = args.np
    h5temp = args.hname

    ncpu = min([ncpu, len(zlist)])
    if ncpu == 1:
        pool = None
    else:
        pool = Pool(ncpu)

    print(ncpu, h5temp, args.config)
    downsample_all_h5(zlist=zlist, pool=pool, htemp=h5temp, **conv_pars)
    try:
        pool.close()
    except:
        pass
