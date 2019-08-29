import numpy as np
import struct

from ckc.utils import ckc_params
from prospect.sources import StarBasis

def write_binary(z, logg, logt, sps, outroot='test', zsolar=0.0134, **extras):
    """Convert a *flat* hdf5 spectral data file to the binary format
    appropriate for FSPS, interpolating the hdf spectra to target points.  This
    also writes a .lambda file and a zlegend file.  A simple ```wc *.lambda```
    will give the number of wavelength points to specify in sps_vars.f90
    """
    params = {'feh': np.log10(z/zsolar)}

    # Write the spectral file
    name = '{0}_z{1:6.4f}.spectra.bin'.format(outroot, z)
    outfile = open(name, 'wb')
    for g in logg:
        for t in logt:
            params['logg'] = g
            params['logt'] = t
            try:
                w, spec, _ = sps.get_star_spectrum(**params)
            except(ValueError):
                print('Could not build spectrum for {}'.format(params))
                spec = np.zeros(len(sps.wavelengths))
            for flux in spec:
                outfile.write(struct.pack('f', flux))
    outfile.close()
    return None


def write_all_binaries(zlist=[], outroot='test', sps=None, **kwargs):

    _, logg, logt = ckc_params()

    # Write the wavelength file
    wfile = open('{}.lambda'.format(outroot), 'w')
    for wave in sps.wavelengths:
            wfile.write('{}\n'.format(wave))
    wfile.close()

    # Loop over Z, writing a binary for each Z and writing zlegend
    with open('{}_zlegend.dat'.format(outroot), 'w') as zfile:
        for z in zlist:
            write_binary(z, logg, logt, sps, outroot=outroot, **kwargs)
            zfile.write('{:6.4f}\n'.format(z))

    return None


if __name__ == "__main__":

    runparams = {'verbose': True,
                 # Interpolator
                 'libname':'lores/manga/c3k+dMall_manga-sigma50kms.h5',
                 'in_memory': False,
                 'use_params': ['logt', 'logg', 'feh'],
                 'logify_Z': False,
                 # Z definition
                 'zsolar': 0.0134,
                 'zlist': np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5]),
                 # Output
                 'outroot': 'lores/manga/c3kdM_manga'
                 }

    runparams['zlist'] = runparams['zsolar'] * 10**runparams['zlist']
    sps = StarBasis(**runparams)

    write_all_binaries(sps=sps, **runparams)
