import os, sys, time
from itertools import product
import numpy as np
import h5py
import ykc
from multiprocessing import Pool

hires_fstring = ("at12_teff={t:4.0f}_g={g:3.2f}_feh={feh:3.1f}_"
                 "afe={afe:3.1f}_cfe={cfe:3.1f}_nfe={nfe:3.1f}_"
                 "vturb={vturb:3.1f}.spec.gz")
h5template = '../h5/ykc_feh={:3.1f}.full.h5'
full_params = {'t': np.arange(4000, 5600, 100),
               'g': [1.0, 1.5, 2.0],
               'feh': np.arange(-4, 1.0, 0.5),
               'afe': [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
               'nfe': [0, 0.3], 'cfe': [0, 0.3],
               'vturb':[0.5, 3.5]}


def param_map(ps):
    """Logify Teff
    """
    ps[0] = np.log10(ps[0])
    return tuple(ps)

pname_map = {'t':'logt', 'g':'logg', 'feh':'feh', 'afe':'afe', 'nfe':'nfe', 'cfe':'cfe', 'vturb':'vturb'}
pnames = [pname_map[p] for p in ykc.param_order]


def get_hires_spectrum(param, fstring=hires_fstring, spectype='full'):
    pars = dict(zip(ykc.param_order, param))
    dirname = "../Plan_Dan_Large_Grid/Sync_Spectra_All_Vt={:3.1f}/".format(pars['vturb'])
    fn = dirname + fstring.format(**pars)
    if os.path.exists(fn) is False:
        print('did not find {}'.format(fn))
        return 0, 0
    fulldf = np.loadtxt(fn)
    wave = np.array(fulldf[:,0])
    full_spec = np.array(fulldf[:,1]) # spectra
    full_cont = np.array(fulldf[:,2]) # continuum
    return full_spec, full_cont, wave


def specset(z):
    h5name = h5template.format(z)
    paramlists = full_params.copy()
    paramlists['feh'] = [z]
    params = list(product(*[paramlists[p] for p in ykc.param_order]))
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
