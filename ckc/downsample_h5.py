# This module contains methods for downsampling the full R=200K C3K spectra to
# other resolutions.

import sys, time, gc, os
import json
import numpy as np
from functools import partial
from itertools import imap
from multiprocessing import Pool


import h5py
#from ykc_data import sigma_to_fwhm
from prospect.utils import smoothing


from libparams import sigma_to_fwhm


__all__ = ["construct_outwave", "initialize_h5",
           "smooth_onez", "downsample_allz_map", # These 2 get used together
           "smooth_onez_map", "downsample_allz"] # These 2 get used together


tiny_number = 1e-33


def construct_outwave(min_wave_smooth=0.0, max_wave_smooth=np.inf,
                      resolution=3e5, oversample=2.0,
                      logarithmic=False, **extras):
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
                        resolution / oversample)
    return out    


def smooth_onez(fullres_hname, resolution=1.0, **conv_pars):
    """Read one full resolution h5 file, downsample every spectrum in
    that file, and return the result as ndarrays.

    Should probably be using imap here instead of in downsample_all_h5.
    """
    outwave = construct_outwave(resolution=resolution, **conv_pars)
    #print(resolution, len(outwave), conv_pars['smoothtype'])
    with h5py.File(fullres_hname, 'r') as fullres:
        params = np.array(fullres['parameters'])
        whires = np.array(fullres['wavelengths'])
        flores = np.zeros([len(params), len(outwave)])
        for i, p in enumerate(params):
            fhires = fullres['spectra'][i, :]
            s = mappable_smoothspec(fhires, wave=whires, resolution=resolution,
                                    outwave=outwave, **conv_pars)
            flores[i, :] = s
    gc.collect()
    return outwave, flores, params, None


def mappable_smoothspec(flux, wave=None, resolution=None,
                        outwave=None, **extras):
    """Convert keyword to positional arguments. Also replace floating underflow
    and NaN with zero"""
    bad = np.isnan(flux) | (flux < tiny_number)
    flux[bad] = 0.0
    s = smoothing.smoothspec(wave, flux, resolution,
                             outwave=outwave, **extras)
    return s


def smooth_onez_map(fullres_hname, resolution=1.0,
                    outfile=None, datasets=None, pool=None,
                    **conv_pars):
    """Read one full resolution h5 file, downsample every spectrum in that
    file, and put in the supplied hdf5 datasets.  Uses `map` to distribute the
    spectra to different processors.
    """
    # get the imap ready
    if pool is not None:
        M = pool.imap
    else:
        M = imap

    # prepare output
    if datasets is not None:
        wave, spec, pars = datasets
    else:
        wave = outfile['wavelengths']
        spec = outfile['spectra']
        pars = outfile['parameters']

    # existing output wave and number of models
    outwave = np.array(wave)
    nmod = len(pars)

    # now map/loop through a single highres h5 file
    with h5py.File(fullres_hname, 'r') as fullres:

        # Get the hires parameters
        params = np.array(fullres['parameters'])
        whires = np.array(fullres['wavelengths'])

        # expand output arrays
        nnew = len(params)
        spec.resize(nmod + nnew, axis=0)
        pars.resize(nmod + nnew, axis=0)
        
        # build a mappable function and iterator
        smooth = partial(mappable_smoothspec,
                         wave=whires, resolution=resolution,
                         outwave=outwave, **conv_pars)
        mapper = M(smooth, fullres['spectra'])

        # iterate over the hires spectra placing result in output
        for i, result in enumerate(mapper):
            spec[nmod+i, :] = result
            pars[nmod+i] = params[i]
            outfile.flush()

    # give the output dataset objects back
    return wave, spec, pars, outfile


def downsample_allz(pool=None, htemp='ckc_feh={:+3.2f}.full.h5',
                    zlist=[-4.0, -3.0, -2.0, -1.0, 0.0],
                    **conv_pars):
    """Simple loop over hdf5 files (one for each feh) but use `map` within each
    loop to distribute the spectra in each file to different processors to be
    smoothed. Calls `smooth_onez_map`.

    "Map over spectra"
    """
    
    hnames = [htemp.format(*np.atleast_1d(z)) for z in zlist]

    # Output filename and wavelength grid
    outdir = conv_pars.get('outdir', 'lores')
    outname = '{}/{}.h5'.format(outdir, conv_pars['name'])
    outwave = construct_outwave(**conv_pars)
    # Output h5 datasets
    with h5py.File(hnames[0], 'r') as f:
        pars = f['parameters']
        output = initialize_h5(outname, outwave,
                               np.atleast_2d(outwave), pars)
    outfile, dsets = output[-1], output[:-1]

    # loop over h5 files
    for i, hfile in enumerate(hnames):
        if os.path.exists(hfile) is False:
            continue
        output = smooth_onez_map(hfile, pool=pool, outfile=outfile,
                                 datasets=dsets, **conv_pars)
        outfile = output[-1]
        dsets = output[:-1]

    # write useful info and close
    for k, v in list(conv_pars.items()):
        outfile.attrs[k] = json.dumps(v)
    outfile.close()


def downsample_allz_map(pool=None, htemp='ckc_feh={:+3.2f}.full.h5',
                        zlist=[-4.0, -3.0, -2.0, -1.0, 0.0],
                        **conv_pars):
    """ Use `map` to distribute the loop over hdf5 files (one for each feh) to
    different processors.  Calls `smooth_onez`.

    "Map over files"
    """
    if pool is not None:
        M = pool.imap
    else:
        M = imap

    hnames = [htemp.format(*np.atleast_1d(z)) for z in zlist]
    hnames = [hn for hn in hnames if os.path.exists(hn)]

    # Output filename and wavelength grid
    outdir = conv_pars.get('outdir', 'lores')
    outname = '{}/{}.h5'.format(outdir, conv_pars['name'])
    outwave = construct_outwave(**conv_pars)
    # Output h5 datasets
    with h5py.File(hnames[0], 'r') as f:
        pars = f['parameters']
        output = initialize_h5(outname, outwave,
                               np.atleast_2d(outwave), pars)
    wave, spectra, par, out = output

    # Map iterable
    downsample_with_pars = partial(smooth_onez, **conv_pars)
    mapper = M(downsample_with_pars, hnames)

    # loop over h5 files
    for i, (w, s, p, x) in enumerate(mapper):
        nmod, nw = spectra.shape
        nnew = len(p)
        spectra.resize(nmod + nnew, axis=0)
        spectra[nmod:, :] = s
        par.resize(nmod + nnew, axis=0)
        par[nmod:] = p
        out.flush()

    # write useful info and close
    for k, v in list(conv_pars.items()):
        out.attrs[k] = json.dumps(v)
    out.close()


def initialize_h5(name, wave, spec, par):
    out = h5py.File(name, "w")
    nmod, nw = len(par), len(wave)
    spectra = out.create_dataset('spectra', shape=(0, nw),
                                 maxshape=(None, nw))
    params = out.create_dataset('parameters', shape=(0,),
                                maxshape=(None,), dtype=par.dtype)
    wavelength = out.create_dataset('wavelengths', data=wave)
    out.flush()
    return wavelength, spectra, params, out


def convert_resolution(R_fwhm, R_library=3e5):
    """Convert from standard 'R" values based on lambda/FWHM to the
    lambda/sigma values expected by smoothspec
    """
    R_sigma = sigma_to_fwhm / np.sqrt(1/R_fwhm**2 - 1/R_library**2)
    return R_sigma


if __name__ == "__main__":

    fehlist = [-4.0, -3.5, -3.0, -2.75, -2.5, -2.25, -2.0,
               -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
               0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    afelist = [-0.2, 0.0, 0.2, 0.4, 0.6]
    from itertools import product
    zlist = list(product(fehlist, afelist))

    # --- Arguments -----
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--np", type=int, default=6,
                        help="number of processors")
    # Smoothing parameters
    parser.add_argument("--resolution", type=float, default=5000.,
                        help=("Resolution in lambda/dlambda where dlambda is FWHM"))
    parser.add_argument("--smoothtype", type=str, default='R',
                        help=("see smoothspec"))
    parser.add_argument("--oversample", type=int, default=3,
                        help=("Pixels per FWHM (resolution element)"))
    parser.add_argument("--fftsmooth", type=bool, default=True,
                        help=("Whether to use FFT for the smoothing"))
    parser.add_argument("--min_wave_smooth", type=float, default=0.1e4,
                        help=("minimum wavelength for smoothing"))
    parser.add_argument("--max_wave_smooth", type=float, default=2.5e4,
                        help=("maximum wavelength for smoothing"))
    # Filenames
    parser.add_argument("--fullres_hname", type=str, default="{}/{}_feh{{:+3.2f}}_afe{{:+2.1f}}.full.h5",
                        help=("A string that gives the full resolution "
                              "h5 filename template (to be formatted later)."))
    parser.add_argument("--ck_vers", type=str, default="c3k_v1.3",
                        help=("Name of directory that contains the "
                              "version of C3K spectra to use."))
    parser.add_argument("--fulldir", type=str, default='/n/conroyfs1/bdjohnson/data/stars/{}/h5/',
                        help=("Location of the HDF5 versions of .spec and .flux files"))
    parser.add_argument("--outname", type=str, default='./{}_R5K.h5',
                        help=("Full path and name of the output HDF5 file."))

    # Mess with some args
    args = parser.parse_args()
    args.fulldir = args.fulldir.format(args.ck_vers)
    args.outname = args.outname.format(args.ck_vers)
    hname_template = args.fullres_hname.format(args.fulldir, args.ck_vers)
    params = vars(args)
    if args.smoothtype == "R":
        params["resolution"] = convert_resolution(params["resolution"])
        params["oversample"] = params["oversample"] / sigma_to_fwhm
        
    # --- Set up the pool ----
    ncpu = args.np
    #ncpu = min([ncpu, len(zlist)])
    if ncpu == 1:
        pool = None
    else:
        pool = Pool(ncpu)

    # --- GO! ----
    print(ncpu, h5temp)
    downsample_allz(zlist=zlist, pool=pool, htemp=hname_template, **params)

    # --- Cleanup ---
    try:
        pool.close()
    except(AttributeError):
        pass
