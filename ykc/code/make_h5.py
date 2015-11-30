import os, sys, time
from itertools import product
import numpy as np
import h5py
import ykc

full_params = {'t': np.arange(4000, 5600, 100),
               'g': [1.0, 1.5, 2.0],
               'feh': np.arange(-4, 1.0, 0.5),
               'afe': [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
               'nfe': [0, 0.3], 'cfe': [0, 0.3],
               'vturb':[0.5, 3.5]}

def get_spectrum(param):
    pars = dict(zip(ykc.param_order, param))
    w, s = ykc.convolve_lam_onestep(fwhm=1.0, wlo=3.5e3, whi=1.1e4, **pars)
    return w, s


def param_map(ps):
    """Logify Teff
    """
    ps[0] = np.log10(ps[0])
    return tuple(ps)
    

pname_map = {'t':'logt', 'g':'logg', 'feh':'feh', 'afe':'afe', 'nfe':'nfe', 'cfe':'cfe', 'vturb':'vturb'}
pnames = [pname_map[p] for p in ykc.param_order]


if __name__ == "__main__":
    h5name = 'ykc_deimos.h5'
    paramlists = {'t': [5000],
                  'g': [1.0],
                  'feh': [0.0],
                  'afe': [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
                  'nfe': [0, 0.3], 'cfe': [0, 0.3],
                  'vturb':[0.5, 3.5]}

    paramlists = full_params
    
    params = list(product(*[paramlists[p] for p in ykc.param_order]))
    nspec, npar = len(params), len(params[0])
    dt = np.dtype(zip(pnames, npar * [np.float]))
    pars = np.empty(nspec, dtype=dt)

    with h5py.File(h5name, 'w') as f:
        w, s = get_spectrum(params[0])
        nwave = len(w)
        spec = f.create_dataset('spectra', (nspec, nwave))
        pset = f.create_dataset('parameters', data=pars)
        wave = f.create_dataset('wavelengths', data=w)
        for i, p in enumerate(params):
            w, s = get_spectrum(p)
            spec[i,:] = s
            pset[i] = tuple(param_map(list(p)))

