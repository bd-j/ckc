import numpy as np


import ckc_params, read_binary_spec

# Build binary files directly from the CKC synthe output

ckc14_hdf = ''

def get_coverage(cckc_hdf):
    with h5py.File(ckc14_hdf, "r") as f:
        logg = f['logg'][:]
        logt = f['logt'][:]
        zstr = f['spectra'].keys()
        flag = np.zeros([nz, ng, nt])
        for i, z in enumerate(zstr):
            flag[i, :, :] = f['spectra'][z].max(axis=-1) > 1e-32
    return zstr, logg, logt, flag


           

def binary_to_hdf(z, hname, bindir='.'):
    wave = np.loadtxt('{0}/ckc14.lambda'.format(ckc_dir))
    nw = len(wave)
    zlegend, logg, logt = ckc_params()
    nspec = len(logg) * len(logt)
    with h5py.File(hname, "w") as f:
        spgr = f.create_group('spectra')
        fw = f.create_dataset('wavelengths', data=wave)
        fg = f.create_dataset('logg', data=logg)
        ft = f.create_dataset('logt', data=logt)

        name = '{0}/bin/ckc14_z{1}.spectra.bin'.format(ckc_dir, z)
        spec = read_binary_spec(name, nw, nspec)
        spec = spec.reshape(len(logg), len(logt), nw)
        fl = spgr.create_dataset('z{0}'.format(z), data=spec)
        f.flush()
