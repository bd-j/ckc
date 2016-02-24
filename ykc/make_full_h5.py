# This module exists to create HDF5 files of the full resolution ykc grid from
# the ascii files.
# The ascii read-in takes about 5s per spectrum.
# The output is split into separate hdf5 files based on `feh`

import os, sys, time
from itertools import product
import numpy as np
import h5py
from multiprocessing import Pool
from ykc_data import full_params, param_order, hires_fstring


def param_map(ps):
    """Logify Teff
    """
    ps[0] = np.log10(ps[0])
    return tuple(ps)

pname_map = {'t':'logt', 'g':'logg', 'feh':'feh', 'afe':'afe', 'nfe':'nfe', 'cfe':'cfe', 'vturb':'vturb'}
pnames = [pname_map[p] for p in param_order]


def get_hires_spectrum(param, fstring=hires_fstring):
    """Read a hires spectrum from ascii files.

    :param param:
        An iterable of ykc parameters.

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
    pars = dict(zip(param_order, param))
    dirname = "data/Plan_Dan_Large_Grid/Sync_Spectra_All_Vt={:3.1f}/".format(pars['vturb'])
    fn = dirname + fstring.format(**pars)
    if os.path.exists(fn) is False:
        print('did not find {}'.format(fn))
        return 0, 0
    fulldf = np.loadtxt(fn)
    wave = np.array(fulldf[:,0])
    full_spec = np.array(fulldf[:,1]) # spectra
    full_cont = np.array(fulldf[:,2]) # continuum
    return full_spec, full_cont, wave


def specset(z, h5template='h5/ykc_feh={:3.1f}.full.h5'):
    """Make an HDF5 file containing the full resolution spectrum (and
    continuum) of all the ykc spectra with a given `feh` value.  This function
    should have minimal kwargs, so it can be easily mapped.

    :param z:
        The value of `feh`.

    :param h5template:
        The oputput h5 name template
    """
    h5name = h5template.format(z)
    paramlists = full_params.copy()
    paramlists['feh'] = [z]
    params = list(product(*[paramlists[p] for p in param_order]))
    nspec, npar = len(params), len(params[0])
    dt = np.dtype(zip(pnames, npar * [np.float]))
    pars = np.empty(nspec, dtype=dt)
    with h5py.File(h5name, 'w') as f:
        s, c, w = get_hires_spectrum(params[0])
        nwave = len(w)
        spec = f.create_dataset('spectra', (nspec, nwave))
        cont = f.create_dataset('continuua', (nspec, nwave))
        pset = f.create_dataset('parameters', data=pars)
        wave = f.create_dataset('wavelengths', data=w)
        for i, p in enumerate(params):
            s, c, w = get_hires_spectrum(p)
            spec[i,:] = s
            cont[i,:] = c
            pset[i] = tuple(param_map(list(p)))
    return h5name


if __name__ == "__main__":

    ncpu = 4
    pool = Pool(ncpu)

    zlist = full_params['feh']
    filenames = list(pool.map(specset, list(zlist)))
    print(filenames)
    pool.terminate()
