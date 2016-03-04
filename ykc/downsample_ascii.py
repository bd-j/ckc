# This module contains methods for downsampling the ascii formatted ykc spectra
# and putting them in a h5 file.
#  Currently this is set up to only work for downsampling to constant FWHM_lambda

import os, time, sys
from itertools import product
import numpy as np
from bsfh.utils import smoothing
from ykc_data import ckms, sigma_to_fwhm, Rykc, hires_fstring
from ykc_data import full_params, param_order

__all__ = ["downsample_ykc_from_ascii", "getflux_hires"]

pname_map = {'t':'logt', 'g':'logg', 'feh':'feh', 'afe':'afe', 'nfe':'nfe', 'cfe':'cfe', 'vturb':'vturb'}
pnames = [pname_map[p] for p in param_order]
conv_pars = {'fwhm': 1.0, 'wlo': 3.5e3, 'whi':1.1e4}

def downsample_ykc_from_ascii(h5name):
    """Build a downsampled ykc library directly from the ascii files, and write
    to an hdf5 file given by `h5name`.  This is painfully slow due to the very
    long ascii read times.
    """
    paramlists = full_params
    
    params = list(product(*[paramlists[p] for p in param_order]))
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

        spec.attrs['fwhm'] = u'{} AA'.format(conv_pars['fwhm'])


def get_spectrum(param):
    """Get (from ascii) and downsample a single spectrum
    """
    pars = dict(zip(param_order, param))
    pars.update(conv_pars)
    w, s = convolve_lam_onestep(**pars)
    return w, s


def param_map(ps):
    """Logify Teff
    """
    ps[0] = np.log10(ps[0])
    return tuple(ps)
    

def getflux_hires(fstring=hires_fstring, spectype='full', **pars):
    """Read a hires spectrum from ascii files.  The parameters are given as a
    set of keyword arguments to this function.  They are
    * t - temperature
    * g - logg
    * feh - [Fe/H]
    * afe  - [alpha/Fe]
    * cfe - 
    * nfe - 
    * vturb - 

    :param fstring:
        Format string for the ascii filename

    :param spectype: (default: "full")
        String, one of:
        * "full" - return the full spectrum (continuum and lines)
        * "continuum"  - return just the continuum
        * "normalized" - return the continuum normalized spectrum.

    :returns spectrum:
        The flux vector of high resolution spectrum.  The form of the output is
        determined by the `spectype` keyword.

    :returns wave:
        The wavelength vector of the high-resolution spectrum.
    """
    dirname = "data/Plan_Dan_Large_Grid/Sync_Spectra_All_Vt={:3.1f}/".format(pars['vturb'])
    fn = dirname + fstring.format(**pars)
    print(fn)
    if os.path.exists(fn) is False:
        print('did not find {}'.format(fn))
        return 0, 0
    fulldf = np.loadtxt(fn)
    wave = np.array(fulldf[:,0])
    full_spec = np.array(fulldf[:,1]) # spectra
    full_cont = np.array(fulldf[:,2]) # continuum
    if spectype == 'normalized':
        full_spec /= full_cont # normalized spectra
    elif spectype == 'continuum':
        full_spec = full_cont        
    return full_spec, wave


def convolve_lnlam(R=10000, wlo=3e3, whi=1e4, wpad=20.0, **pars):
    """ Convolve to lower R.

    :param R:
        Desired resolution in lambda/delta_lambda_FWHM
    """
    wlim = wlo - wpad, whi + wpad
    fhires, whires = getflux_hires(**pars)
    good = (whires > wlim[0]) & (whires < wlim[1])

    R_sigma = R * sigma_to_fwhm
    dlnlam = 1.0 / (R_sigma *2.0)
    outwave = np.arange(np.log(wlo), np.log(whi), dlnlam)
    outwave = np.exp(outwave)
    # convert desired resolution to velocity and sigma instead of FWHM
    sigma_v = ckms / R / sigma_to_fwhm
    flux = smoothing.smooth_vel_fft(whires[good], fhires[good], outwave, sigma_v)
    return outwave, flux


def convolve_lam_onestep(fwhm=1.0, wlo=4e3, whi=1e4, wpad=20.0, **pars):
    """Read a hires ascii spectrum and convolution in lambda directly, assuming
    the wavelength dependent sigma of the input library is unimportant.  This
    is often a bad assumption, and is worse the higher the resolution of the
    output
    """
    sigma = fwhm / sigma_to_fwhm
    wlim = wlo - wpad, whi + wpad

    # Read-in ascii and mask.  Interpolation to constant lambda grid handled by
    # smoothing function.
    ts = time.time()
    fhires, whires = getflux_hires(**pars)
    good = (whires > wlim[0]) & (whires < wlim[1])
    print('read in took {}s'.format(time.time() - ts))

    # now generate output grid and smooth
    ts = time.time()
    outwave = np.arange(wlo, whi, sigma / 2.0)
    # should switch to bsfh function here # Done!
    flux = smoothing.smooth_wave_fft(whires[good], fhires[good],
                                     outwave=outwave, sigma_out=sigma)
    print('final took {}s'.format(time.time() - ts))
    return outwave, flux


def convolve_lam(fwhm=1.0, wlo=4e3, whi=1e4, wpad=20.0,
                 rconv=sigma_to_fwhm, fast=False, **pars):
    """Convolve first to lower resolution (in terms of R) then do the
    wavelength dependent convolution to a constant sigma_lambda resolution.
    
    :param fwhm:
        Desired resolution delta_lambda_FWHM in AA
    """
    sigma = fwhm / sigma_to_fwhm
    wlim = wlo - wpad, whi + wpad

    # Read the spectrum
    ts = time.time()
    fhires, whires = getflux_hires(**pars)
    good = (whires > wlim[0]) & (whires < wlim[1])
    print('read in took {}s'.format(time.time() - ts))

    if fast:
        # ** This doesn't quite work ** - gives oversmoothed spectra
        # get most of the way by smoothing via fft
        ts = time.time()
        # Maximum lambda/sigma_lambda of the output spectrum, with a little padding
        Rint_sigma = whi / sigma * 1.1
        inres = Rint_sigma
        dlnlam = 1.0 / (Rint_sigma * 2.0)
        wmres = np.exp(np.arange(np.log(wlim[0]), np.log(wlim[1]), dlnlam))
        fmres = smoothing.smooth_vel_fft(whires[good], fhires[good], outwave=wmres,
                                         Rout=Rint_sigma, Rin=Rykc*rconv)
        print('intermediate took {}s'.format(time.time() - ts))
    else:
        good = (whires > wlim[0]) & (whires < wlim[1])
        wmres, fmres = whires[good], fhires[good]
        inres = Rykc*rconv

    # Do the final smoothing *without FFT* to try and capture wavelength
    # dependence of the resolution of the input spectrum
    ts = time.time()
    outwave = np.arange(wlo, whi, sigma / 2.0)
    flux = smoothing.smooth_wave(wmres, fmres, outwave, sigma, inres=inres,
                                 in_vel=True, nsigma=20)
    print('final took {}s'.format(time.time() - ts))

    return outwave, flux

def test():
    """Test different ways of doing the convolutions, for speed and accuracy.
    """
    import matplotlib.pyplot as pl
    pars = {'t':5500, 'g':1.0, 'feh': -0.5, 'afe':0.0,
            'cfe': 0.0, 'nfe': 0.0, 'vturb':0.5}
    ts = time.time()
    w, s = convolve_lam_onestep(fwhm=1.0, wlo=4e3, whi=1e4, **pars)
    dt = time.time() - ts
    ts = time.time()
    w1, s1 = convolve_lam(fwhm=1.0, wlo=4e3, whi=1e4, fast=False, **pars)
    dt1 = time.time() - ts
    ts = time.time()
    w2, s2 = convolve_lam(fwhm=1.0, wlo=4e3, whi=1e4, fast=True, **pars)
    dt2 = time.time() - ts
    w3, s3 = convolve_lam(fwhm=1.0, wlo=4e3, whi=1e4, fast=True, rconv=1.0, **pars)
    print("took {}s, {}s, and {}s".format(dt, dt1, dt2))
    fig, ax = pl.subplots()
    ax.plot(w1, (s1-s) / s1, label = 'slow / onestep - 1')
    ax.plot(w1, (s1-s2) / s1, label = 'slow / fast - 1')
    ax.set_ylim(-0.05, 0.05)
    fig.show()
    
    #with h5py.File("test.h5",'w') as f:
        #wd = f.create_dataset('wave', data=w)
    #    sd = f.create_dataset('spectrum', data=s)
    
if __name__ == "__main__":
    downsample_ykc_from_ascii('ykc_deimos.h5')
    
