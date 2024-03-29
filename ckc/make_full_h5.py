#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This module exists to create HDF5 files of the full resolution ckc grid
from the ascii files.
The ascii read-in takes about 15s per spectrum fort hires
The output is split into separate hdf5 files based on `feh` and `afe`
"""

import os, sys, time
from itertools import product
import numpy as np
import h5py
from multiprocessing import Pool
from functools import partial

param_order = ['t', 'g', 'feh', 'afe']


__all__ = ["get_hires_spectrum", "get_lores_spectrum", "get_lores_wavegrid",
           "existing_params", "files_and_params", "transform_params",
           "specset", "full_h5"]


t = [np.array([2500., 2800., 3000., 3200.]),
     np.arange(3500., 13000., 250.),
     np.arange(13000., 51000., 1000.)]

full_params = {'t': np.concatenate(t),
               'g': np.arange(-1.0, 5.5, 0.5),
               'feh': np.concatenate([np.array([-4, -3.5]),
                                     np.arange(-3, 1.0, 0.25)]),
               'afe': [0.0]
               }

pname_map = {'t': 'logt', 'g': 'logg', 'feh': 'feh', 'afe': 'afe'}
pnames = [pname_map[p] for p in param_order]


def full_h5(z, args=None):
    """Make the full spectrum HDF file for a single feh, afe combo.
    Handles much of the file formatting based on `args`

    Parameters
    -----
    z : 2 element tuple of floats (feh, afe)
    """

    z = np.atleast_1d(z)

    # Paths and filename templates
    ck_vers = args.ck_vers  # ckc_v1.2 | c3k_v1.3
    basedir = args.basedir
    if args.spec_type == 'hires':
        dirname = 'spec/'
        ext = '.spec.gz'
        h5_outname = os.path.join(args.fulldir, ck_vers + '_feh{:+3.2f}_afe{:+2.1f}.full.h5')
    elif args.spec_type == 'lores':
        dirname = 'flux/'
        ext = '.flux'
        h5_outname = os.path.join(args.fulldir, ck_vers + '_feh{:+3.2f}_afe{:+2.1f}.flux.h5')
    else:
        raise(ValueError, "spec_type must be one of 'hires' or 'lores'")
    dstring = os.path.join("at12_feh{:+3.2f}_afe{:+2.1f}", dirname)
    dstring = os.path.join(basedir, ck_vers, dstring)
    fstring = "at12_feh{feh:+3.2f}_afe{afe:+2.1f}_t{t:05.0f}g{g:.4s}" + ext
    searchstring = os.path.join(dstring, '*' + ext)

    # Actually do the thing
    #z = (feh, afe)
    fn = specset(z, h5template=h5_outname, searchstring=searchstring,
                 fstring=fstring, dstring=dstring)
    return fn


def get_hires_spectrum(filename=None, param=None,
                       fstring='', dstring='', **extras):
    """Read a hires spectrum from ascii files.

    :param param:
        An iterable of ckc parameters.

    :param fstring:
        Format string for the ascii filename

    :returns full_spec:
        The flux vector of high resolution spectrum, including both lines and
        continuum.

    :returns full_cont:
        The flux vector of the continuum, which can be sued to make continuum
        normalized spectra

    :returns wave:
        The wavelength vector of the high-resolution spectrum.
    """
    if filename is None:
        pars = dict(zip(param_order, param))
        # hack to get correct g widths
        pars['g'] = '{:4.2f}'.format(pars['g'])
        dirname = dstring.format(pars['feh'], pars['afe'])
        fn = os.path.join(dirname, fstring.format(**pars))
    else:
        fn = filename

    if os.path.exists(fn) is False:
        print('did not find {}'.format(fn))
        return 0, 0, None
    fulldf = np.loadtxt(fn)
    nc = fulldf.shape[-1]
    wave = np.array(fulldf[:, 0])
    full_spec = np.array(fulldf[:, 1])  # spectra
    if nc > 2:
        full_cont = np.array(fulldf[:, 2])  # continuum
    else:
        full_cont = 0.0

    return full_spec, full_cont, wave


def get_lores_spectrum(filename=None, param=None,
                       fstring='', dstring='', **extras):
    if filename is None:
        pars = dict(zip(param_order, param))
        # hack to get correct g widths
        pars['g'] = '{:4.2f}'.format(pars['g'])
        dirname = dstring.format(pars['feh'], pars['afe'])
        fn = os.path.join(dirname, fstring.format(**pars))
    else:
        fn = filename

    if os.path.exists(fn) is False:
        print('did not find {}'.format(fn))
        return 0, 0, None
    with open(fn, "r") as f:
        lines = f.readlines()
    dat = [l.replace('\n', '').split() for l in lines[2:]]
    wave = np.array([float(d[0]) for d in dat])
    flux = np.array([float(d[1]) for d in dat])
    #dlam = np.grad(wave)
    #(wave / dlam).mean()

    return flux, flux, wave


def get_lores_wavegrid(params, fstring, dstring):
    wave = []
    for i, p in enumerate(params):
        s, c, w = get_lores_spectrum(param=p, fstring=fstring, dstring=dstring)
        wave = np.unique(np.concatenate([wave, w]))
    return np.sort(wave)


def existing_params(params, fstring='', dstring=''):
    """Test for the existence of a spec file for each of the
    parameters in the given list, and return a list of only those
    parameters for which a spec file exists
    """
    exists = []
    for i, p in enumerate(params):
        pars = dict(zip(param_order, p))
        # hack to get correct g widths
        pars['g'] = '{:4.2f}'.format(pars['g'])
        dirname = dstring.format(pars['feh'], pars['afe'])
        fn = dirname + fstring.format(**pars)
        if os.path.exists(fn):
            exists.append(p)
    return exists


def files_and_params(searchstring='.'):
    """Given a path to search for files, return all matching files as well as
    row matched lists of the corresponding spectral parameters.  This assumes
    a certain filename format.
    """

    import glob, re

    # get all matching files
    files = glob.glob(searchstring)

    print(searchstring, len(files))
    # set up parsing tools
    tpat, ts = re.compile(r"_t.{5}"), slice(2, None)
    zpat, zs = re.compile(r"_feh.{5}"), slice(4, None)
    gpat, gs = re.compile(r"g.{4}."), slice(1, 5)
    apat, asl = re.compile(r"_afe.{4}"), slice(4, None)

    # parse filenames for parameters
    feh = [float(re.findall(zpat, f)[-1][zs]) for f in files]
    teff = [float(re.findall(tpat, f)[-1][ts]) for f in files]
    logg = [float(re.findall(gpat, f)[-1][gs]) for f in files]
    try:
        afe = [float(re.findall(apat, f)[-1][asl]) for f in files]
    except(IndexError):
        afe = len(logg) * [0.0]

    # make sure we get the param order right using the global param_order
    parlists = {'t': teff, 'g': logg, 'feh': feh, 'afe': afe}
    params = zip(*[parlists[p] for p in param_order])
    return files, list(params)


def transform_params(ps):
    """Logify Teff
    """
    ps[0] = np.log10(ps[0])
    return tuple(ps)


def specset(z, h5template='h5/ckc_feh={:+3.2f}.full.h5',
            fstring='', dstring='', searchstring=None):
    """Make an HDF5 file containing the full resolution spectrum (and
    continuum) of all the ckc spectra with a given `feh` (and `afe`) value.
    This function should have minimal kwargs, so it can be easily mapped.

    :param z:
        Two element sequence giving the value of `feh` and `afe`.

    :param h5template:
        The oputput h5 name template
    """
    z = np.atleast_1d(z)
    h5name = h5template.format(*z)

    # Get a set of existing parameters for this feh value
    if searchstring is None:
        # Use a specified grid of parameters
        paramlists = full_params.copy()
        paramlists['feh'] = [z]
        # 1D array of parameters
        params = list(product(*[paramlists[p] for p in param_order]))
        # restricted to those actually existing
        params = existing_params(params, fstring=fstring, dstring=dstring)
    else:
        # just pull everything from the directory matching searchstring
        files, params = files_and_params(searchstring.format(*z))
        params = np.array(params)

    # skip directories that don't exist
    try:
        nfile = len(params)
        if nfile == 0:
            return (h5name, 0)
    except(TypeError):
        return (h5name, 0)

    # choose which reader to use and get the wavelength grid
    lores = fstring.split('.')[-1] == 'flux'
    if lores:
        get_spectrum = get_lores_spectrum
        wave = get_lores_wavegrid(params, fstring, dstring)
    else:
        get_spectrum = get_hires_spectrum
        _, _, wave = get_spectrum(param=params[0], fstring=fstring, dstring=dstring)

    # Build and fill the output HDF5 file
    nspec, nwave, npar = len(params), len(wave), len(params[0])
    dt = np.dtype(list(zip(pnames, npar * [float])))
    pars = np.empty(nspec, dtype=dt)
    with h5py.File(h5name, 'w') as f:
        spec = f.create_dataset('spectra', (nspec, nwave))
        cont = f.create_dataset('continuua', (nspec, nwave))
        pset = f.create_dataset('parameters', data=pars)
        wave = f.create_dataset('wavelengths', data=wave)
        for i, p in enumerate(params):
            s, c, w = get_spectrum(param=p, fstring=fstring, dstring=dstring)
            pset[i] = tuple(transform_params(list(p)))
            if w is None:
                continue
            if lores:
                s = np.interp(wave, w, s, left=0., right=0.)
                c = 0.
            try:
                spec[i, :] = s
                cont[i, :] = c
            except:
                spec[i, :] = 0
                cont[i, :] = 0
                print('problem storing spectrum @ params {}'.format(dict(zip(pnames, p))))
            if (i % 10) == 0:
                f.flush()

    print('finished {}'.format(h5name))
    return h5name, nfile


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str,
                        default="/n/holystore01/LABS/conroy_lab/Lab/cconroy/kurucz/grids/")
    parser.add_argument("--outdir", type=str,
                        default="/n/holystore01/LABS/conroy_lab/Lab/bdjohnson/data/kurucz/")
    parser.add_argument("--ck_vers", type=str,
                        default="c3k_v1.3")
    parser.add_argument("--subvers", type=str,
                        default="OPTFAL_vt10", help="Use 'spec' or 'flux' for default hires or lores respecitively.")
    parser.add_argument("--np", type=int,
                        default=20)
    parser.add_argument("--feh", type=float, nargs="*",
                        default=[-4.0, -3.5, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75, -1.5, -1.25,
                                 -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5])
    parser.add_argument("--afe", type=float, nargs="*",
                        default=[-0.2, 0.0, 0.2, 0.4, 0.6])
    args = parser.parse_args()

    ncpu = args.np
    if ncpu == 1:
        M = map
    else:
        pool = Pool(ncpu)
        M = pool.map

    # --- Metallicities to loop over/map ---
    fehlist = args.feh
    afelist = args.afe
    metlist = list(product(fehlist, afelist))
    print(len(metlist))

    # ---- Paths and filename templates -----
    if args.subvers == "flux":
        ext = "flux"
    else:
        ext = "spec"
    h5_outname = (os.path.join(args.outdir, args.ck_vers, args.subvers.lower(), args.ck_vers) +
                  "_feh{:+3.2f}_afe{:+2.1f}"+f".{args.subvers}.h5")
    dstring = os.path.join(args.basedir, args.ck_vers, "at12_feh{:+3.2f}_afe{:+2.1f}/", args.subvers)
    fstring = "at12_feh{feh:+3.2f}_afe{afe:+2.1f}_t{t:05.0f}g{g:.4s}." + ext
    searchstring = os.path.join(dstring, f"*.{ext}")

    print("Looking for files in directories of the form:\n{}".format(dstring))
    print("Each file should have a name of the form:\n{}".format(fstring))
    print("Writing output files to:\n{}".format(h5_outname))

    # String to use for looking for files in a given zdirectory
    #searchstring = 'data/fullres/dM_all/dM_feh??.??/spec/*spec.gz'

    # --- Run -----
    partial_specset = partial(specset, h5template=h5_outname,
                              searchstring=searchstring,
                              dstring=dstring, fstring=fstring)
    ts = time.time()
    filenames = list(M(partial_specset, list(metlist)))
    dur = time.time() - ts

    print(filenames)
    print('took {}s'.format(dur))
    try:
        pool.terminate()
    except:
        pass
