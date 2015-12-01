import os, time, sys
import numpy as np
from bsfh import smoothing
from smooth_fft import smooth_vel_fft, smooth_wave_fft

ckms = 2.998e5
Rykc = 3e5
sigma_to_fwhm = 2.355

hires_fstring = ("at12_teff={t:4.0f}_g={g:3.2f}_feh={feh:3.1f}_"
                 "afe={afe:3.1f}_cfe={cfe:3.1f}_nfe={nfe:3.1f}_"
                 "vturb={vturb:3.1f}.spec.gz")
param_order = ['t', 'g', 'feh', 'afe', 'cfe', 'nfe', 'vturb']
    
def getflux_hires(fstring=hires_fstring, spectype='full', **pars):
    dirname = "../Plan_Dan_Large_Grid/Sync_Spectra_All_Vt={:3.1f}/".format(pars['vturb'])
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
    """
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
    flux = smoothing.smooth_vel(whires[good], fhires[good], outwave, sigma_v)
    return flux
    
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
    print('read in took {}s'.format(time.time() - ts))

    if fast:
        # ** This doesn't quite work ** - gives oversmoothed spectra
        # get most of the way by smoothing via fft
        ts = time.time()
        # Maximum lambda/sigma_lambda of the output spectrum, with a little padding
        Rint_sigma = whi / sigma * 1.1
        #Rint_sigma = 15000 * sigma_to_fwhm
        inres = Rint_sigma
        dlnlam = 1.0 / (Rint_sigma * 2.0)
        wmres = np.exp(np.arange(np.log(wlim[0]), np.log(wlim[1]), dlnlam))
        fmres = smooth_vel_fft(whires, fhires, outwave=wmres,
                               wlo=wlim[0], whi=wlim[1],
                               Rout=Rint_sigma, Rin=Rykc*rconv)
        print('intermediate took {}s'.format(time.time() - ts))
    else:
        good = (whires > wlim[0]) & (whires < wlim[1])
        wmres, fmres = whires[good], fhires[good]
        inres = Rykc*rconv
        
    # Do the final smoothing
    ts = time.time()
    outwave = np.arange(wlo, whi, sigma / 2.0)
    flux = smoothing.smooth_wave(wmres, fmres, outwave, sigma,
                                 inres=inres, in_vel=True, nsigma=20)
    print('final took {}s'.format(time.time() - ts))
    
    return outwave, flux

def convolve_lam_onestep(fwhm=1.0, wlo=4e3, whi=1e4, wpad=20.0, **pars):
    """Do convolution in lambda directly, assuming the wavelength dependent
    sigma of the input library is unimportant.  This is often a bad assumption,
    and is worse the higher the resolution of the output
    """
    sigma = fwhm / sigma_to_fwhm
    wlim = wlo - wpad, whi + wpad

    # Interpolate to linear-lambda grid
    ts = time.time()
    fhires, whires = getflux_hires(**pars)
    good = (whires > wlim[0]) & (whires < wlim[1])
    dw = np.diff(whires[good]).min()
    whires_constdlam = np.arange(wlo, whi, dw/2)
    fhires_constdlam = np.interp(whires_constdlam, whires[good], fhires[good])
    print('read in took {}s'.format(time.time() - ts))

    # now apply a 
    ts = time.time()
    outwave = np.arange(wlo, whi, sigma / 2.0)
    flux = smooth_wave_fft(whires_constdlam, fhires_constdlam, outwave=outwave,
                           wlo=wlo, whi=whi, sigma_out=sigma, nsigma=20)
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
    
