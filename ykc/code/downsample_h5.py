import os, sys, time
from itertools import product
import numpy as np
import h5py
from ykc_data import full_params, param_order

conv_pars = {'fwhm': 1.0, 'wlo': 3.5e3, 'whi':1.1e4}
pname_map = {'t':'logt', 'g':'logg', 'feh':'feh', 'afe':'afe', 'nfe':'nfe', 'cfe':'cfe', 'vturb':'vturb'}
pnames = [pname_map[p] for p in param_order]


def downsample_ykc_from_h5(zlist, pool=None):
    """Unfinished
    """
    if pool is None:
        M = map
    else:
        M = pool.map

    hnames = [ for z in zlist]
    lores = M(downsample_ykc_lam, hnames)
    wave = lores[0][0]
    params = np.concatenate([lo[2] for lo in lores])
    spec = np.concatenate([lo[1] for lo in lores])

    
def downsample_ykc_lam(fullres_hname):
    """Read one full resolution h5 file, downsample every spectrum in
    that file, and treturn the result"""
    with h5py.File(fullres_hname, 'r') as fullres:
        params = np.array(fullres['parameters'])
        whires = np.array(fullres['wavelengths'])

        w, s = convolve_lam_one(whires, fullres['spectra'][0, :], **conv_pars)
        flores = np.empty(len(params), len(s))
        for i, p in enumerate(params):
            fhires = fullres['spectra'][i, :]
            w, s = convolve_lam_one(whires, fhires, **conv_pars)
            flores[i, :] = s

    return w, flores, params


def convolve_lam_one(whires, fhires, fwhm=1.0, wlo=4e3, whi=1e4, wpad=20.0, **pars):
    """Do convolution in lambda directly, assuming the wavelength dependent
    sigma of the input library is unimportant.  This is often a bad assumption,
    and is worse the higher the resolution of the output
    """
    sigma = fwhm / sigma_to_fwhm
    wlim = wlo - wpad, whi + wpad

    # Interpolate to linear-lambda grid
    good = (whires > wlim[0]) & (whires < wlim[1])
    dw = np.diff(whires[good]).min()
    whires_constdlam = np.arange(wlo, whi, dw/2)
    fhires_constdlam = np.interp(whires_constdlam, whires[good], fhires[good])

    # now apply a 
    ts = time.time()
    outwave = np.arange(wlo, whi, sigma / 2.0)
    flux = smooth_wave_fft(whires_constdlam, fhires_constdlam, outwave=outwave,
                           wlo=wlo, whi=whi, sigma_out=sigma, nsigma=20)
    print('final took {}s'.format(time.time() - ts))
    return outwave, flux
 

if __name__ == "__main__":
    do_something
    
