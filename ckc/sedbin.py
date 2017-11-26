# Module for converting single metallicity SED files in HDF format to the
# parameters and binary format expected by fsps

import numpy as np
import struct, os
from itertools import product
from prospect.sources import StarBasis


def dict_struct(struct):
    """Convert from a structured array to a dictionary.  This shouldn't really
    be necessary.
    """
    return dict([(n, [struct[n]]) for n in struct.dtype.names])


def get_basel_params():
    """Get the BaSeL grid parameters as a 1-d list of parameter tuples.  The
    binary files are written in the order logg, logt, wave with wave changing
    fastest and logg the slowest.
    """
    fsps_dir = os.path.join(os.environ["SPS_HOME"], "SPECTRA", "BaSeL3.1")
    logg = np.genfromtxt("{}/basel_logg.dat".format(fsps_dir))
    logt = np.genfromtxt("{}/basel_logt.dat".format(fsps_dir))
    ngrid = len(logg) * len(logt)
    dt = np.dtype([('logg', np.float), ('logt', np.float)])
    basel_params = np.array(list(product(logg, logt)), dtype=dt)
    return basel_params


def get_valid_spectra(ngrid, zind=-2, speclib="BaSeL3.1/basel"):
    zlist = ["0.0002", "0.0006", "0.0020", "0.0063", "0.0200", "0.0632"]
    z = zlist[zind]
    from binary_utils import read_binary_spec
    specname = "{}/SPECTRA/{}".format(os.environ["SPS_HOME"], speclib)
    wave = np.genfromtxt("{}.lambda".format(specname))
    ss = read_binary_spec("{}_wlbc_z{}.spectra.bin".format(specname, z), len(wave), ngrid)
    #logg = np.genfromtxt("{}_logg.dat".format(specname))
    #logt = np.genfromtxt("{}_logt.dat".format(specname))
    #spec = ss.reshape(len(logg), len(logt), len(wave))
    valid = ss.max(axis=1) > 1e-33
    return valid


def interpolate_to_basel():
    pass


def sed_to_bin(sedfile, outname):

    interpolator = StarBasis(sedfile, usepars=['logg', 'logt'], logify_Z=False,
                             n_neighbors=0)
    bpars = get_basel_params()
    valid = get_valid_spectra()
    for p, v in zip(bpars, valid):
        if v:
            bspec, _, _ = interpolator.get_spectrum(**dict_struct(p))
        else:
            bspec = np.zeros_like(interpolator.wavelengths)
        for flux in bspec:
            outfile.write(struct.pack('f', flux))

        
if __name__ == "__main__":
    zsol = 0.0134
    feh = 0.0

    sedfile = 'c3k_v1.3_feh+0.00_afe+0.0.sed.h5'
    z = zsol * 10**feh 
    outname = 'c3k_legac_z{:1.4f}.spectra.bin'.format(z)

    interpolator = StarBasis(sedfile, use_params=['logg', 'logt'], logify_Z=False,
                             verbose=True, n_neighbors=1)
    bpars = get_basel_params()
    valid = get_valid_spectra(len(bpars))
