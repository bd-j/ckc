# This module exists to create HDF5 files of the full resolution ckc grid from
# the ascii files.
# The ascii read-in takes about 15s per spectrum.
# The output is split into separate hdf5 files based on `feh`

import os, sys, time
from itertools import product
import numpy as np
import h5py
from multiprocessing import Pool
from functools import partial

t = [np.array([2500., 2800., 3000., 3200.]),
     np.arange(3500., 13000., 250.),
     np.arange(13000., 51000., 1000.)]

full_params = {'t': np.concatenate(t),
               'g': np.arange(-1.0, 5.5, 0.5),
               'feh': np.concatenate([np.array([-4, -3.5]),
                                     np.arange(-3, 1.0, 0.25)]),
               'afe': [0.0]
               }

param_order = ['t', 'g', 'feh', 'afe']


def param_map(ps):
    """Logify Teff
    """
    ps[0] = np.log10(ps[0])
    return tuple(ps)

pname_map = {'t':'logt', 'g':'logg', 'feh':'feh', 'afe':'afe'}
pnames = [pname_map[p] for p in param_order]


def files_and_params(searchstring='.'):

    import glob, re

    # get all matching files
    files = glob.glob(searchstring)

    print(searchstring, len(files))
    # set up parsing tools
    tpat, ts = re.compile(r"_t.{5}"), slice(2, None)
    zpat, zs = re.compile(r"_feh.{5}"), slice(4, None)
    gpat, gs = re.compile(r"g.{4}.spec.gz"), slice(1, 5)
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
    return files, params


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
        dirname = dstring.format(pars['feh'])
        fn = dirname + fstring.format(**pars)
    else:
        fn = filename

    if os.path.exists(fn) is False:
        print('did not find {}'.format(fn))
        return 0, 0, None
    fulldf = np.loadtxt(fn)
    nc = fulldf.shape[-1]
    wave = np.array(fulldf[:,0])
    full_spec = np.array(fulldf[:,1]) # spectra
    if nc > 2:
        full_cont = np.array(fulldf[:,2]) # continuum
    else:
        full_cont = 0.0

    return full_spec, full_cont, wave


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
        dirname = dstring.format(pars['feh'])
        fn = dirname + fstring.format(**pars)
        if os.path.exists(fn):
            exists.append(p)
    return exists


def specset(z, h5template='h5/ckc_feh={:+3.2f}.full.h5',
            fstring='', dstring='', searchstring=None):
    """Make an HDF5 file containing the full resolution spectrum (and
    continuum) of all the ckc spectra with a given `feh` value.  This function
    should have minimal kwargs, so it can be easily mapped.

    :param z:
        The value of `feh`.

    :param h5template:
        The oputput h5 name template
    """
    h5name = h5template.format(z)
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
        files, params = files_and_params(searchstring.format(z))
        params = np.array(params)

    nspec, npar = len(params), len(params[0])
    dt = np.dtype(zip(pnames, npar * [np.float]))
    pars = np.empty(nspec, dtype=dt)
    with h5py.File(h5name, 'w') as f:
        s, c, w = get_hires_spectrum(param=params[0], fstring=fstring, dstring=dstring)
        nwave = len(w)
        spec = f.create_dataset('spectra', (nspec, nwave))
        cont = f.create_dataset('continuua', (nspec, nwave))
        pset = f.create_dataset('parameters', data=pars)
        wave = f.create_dataset('wavelengths', data=w)
        for i, p in enumerate(params):
            s, c, w = get_hires_spectrum(param=p, fstring=fstring, dstring=dstring)
            pset[i] = tuple(param_map(list(p)))
            if w is None:
                continue
            try:
                spec[i,:] = s
                cont[i,:] = c
            except:
                spec[i,:] = 0
                cont[i,:] = 0
                print('problem storing spectrum @ params {}'.format(dict(zip(param_order, p))))
            if (i % 10) == 0:
                f.flush()

    return h5name


if __name__ == "__main__":

    try:
        ncpu = int(sys.argv[1])
    except(IndexError):
        ncpu = 6

    if ncpu == 1:
        M = map
    else:
        pool = Pool(ncpu)
        M = pool.map
    #M = map

    # --- Metallicities to loop over/map ---
    zlist = [-4.0, -3.5, -3.0, -2.75, -2.5, -2.25, -2.0,
             -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
             0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    #zlist = full_params['feh']
    #zlist = [0.5]

    # ---- Paths and filename templates -----
    ck_vers = 'c3k_v1.3'  # ckc_v1.2
    basedir = '/n/conroyfs1/cconroy/kurucz/grids'
    #basedir = 'data/fullres'

    hires_dstring = os.path.join(basedir, ck_vers,
                                 "at12_feh{:+3.2f}_afe+0.0/spec/")
    hires_fstring = ("at12_feh{feh:+3.2f}_afe{afe:+2.1f}_"
                     "t{t:05.0f}g{g:.4s}.spec.gz")
    hires_searchstring = os.path.join(hires_dstring, '*spec.gz')

    lores_dstring = os.path.join(basedir, ck_vers,
                                 "at12_feh{:+3.2f}_afe+0.0/sed/")
    lores_fstring = ("at12_feh{feh:+3.2f}_afe{afe:+2.1f}_"
                     "t{t:05.0f}g{g:.4s}.sed")
    lores_searchstring = os.path.join(lores_dstring, '*spec.gz')

    # String to use for looking for files in a given zdirectory
    #searchstring = 'data/fullres/dM_all/dM_feh??.??/spec/*spec.gz'

    # output
    h5name = 'h5/'+ck_vers+'_feh{:+3.2f}.full.h5'

    # --- Run -----
    partial_specset = partial(specset, h5template=h5name, searchstring=hires_searchstring,
                              dstring=hires_dstring, fstring=hires_fstring)
    ts = time.time()
    filenames = list(M(partial_specset, list(zlist)))
    dur = time.time() - ts

    print(filenames)
    print('took {}s'.format(dur))
    try:
        pool.terminate()
    except:
        pass
