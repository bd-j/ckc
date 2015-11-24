import os, time
import numpy as np
from bsfh import smoothing
import h5py

ckms = 2.998e5
Rykc = 3e5

hires_fstring = ("at12_teff={t:4.0f}_g={g:3.2f}_feh={feh:3.1f}_"
                 "afe={afe:3.1f}_cfe={cfe:3.1f}_nfe={nfe:3.1f}_"
                 "vturb={vturb:3.1f}.spec.gz")

def getflux_hires(fstring=hires_fstring, **pars):
    dirname = "../Plan_Dan_Large_Grid/Sync_Spectra_All_Vt={:3.1}/".format(pars['vturb'])
    fn = dirname + fstring.format(**pars)
    print(fn)
    if os.path.exists(fn) is False:
        return 0, 0
    fulldf = np.loadtxt(fn)
    wave = np.array(fulldf[:,0])
    full_spec = np.array(fulldf[:,1]) # spectra
    full_cont = np.array(fulldf[:,2]) # continuum
    norm_flux = full_spec/full_cont # normalized spectra
    return norm_flux, wave

def convolve_lnlam(R=10000, wlo=3e3, whi=1e4, **pars):
    """
    :param R:
        Desired resolution in lambda/delta_lambda_FWHM
    """
    fhires, whires = getflux_hires(**pars)
    good = (whires > (wlo-20)) & (whires < (whi+20))

    R_sigma = R * 2.35
    dlnlam = 1.0/(R_sigma *2.0)
    outwave = np.arange(np.log(wlo), np.log(whi), dlnlam)
    outwave = np.exp(outwave)
    # convert desired resolution to velocity and sigma instead of FWHM
    sigma_v = ckms / R / 2.35
    flux = smoothing.smooth_vel(whires[good], fhires[good], outwave, sigma_v)
    return flux
    
def convolve_lam(fwhm=1.0, wlo=3e3, whi=1e4, **pars):
    """
    :param fwhm:
        Desired resolution delta_lambda_FWHM in AA
    """
    fhires, whires = getflux_hires(**pars)
    good = (whires > (wlo-20)) & (whires < (whi+20))

    sigma = fwhm / 2.35
    outwave = np.arange(wlo, whi, sigma / 2.0)
        
    inres = ckms / Rykc
    flux = smoothing.smooth_wave(whires[good], fhires[good], outwave, sigma,
                                 inres=inres, in_vel=True)
    return outwave, flux

if __name__ == "__main__":
    pars = {'t':5500, 'g':1.0, 'feh': -0.5, 'afe':0.0,
            'cfe': 0.0, 'nfe': 0.0, 'vturb':0.5}
    ts = time.time()
    w, s = convolve_lam(fwhm=1.0, wlo=3.5e3, whi=1e4, **pars)
    dt = time.time() - ts
    print("took {}s".format(dt))
    with h5py.File("test.h5",'w') as f:
        #wd = f.create_dataset('wave', data=w)
        sd = f.create_dataset('spectrum', data=s)
    
