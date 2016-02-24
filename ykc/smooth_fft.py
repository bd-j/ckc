# import packages
import numpy as np
from numpy.fft import fft, ifft, fftfreq, rfftfreq

ckms = 2.998e5
sigma_to_fwhm = 2.355


def mask_wave(wavelength, R=20000, wlo=0, whi=np.inf, outwave=None,
              nsigma_pad=20.0, linear=False, **extras):
    """restrict wavelength range (for speed) but include some padding"""
    # Base wavelength limits
    if outwave is not None:
        wlim = np.array([outwave.min(), outwave.max()])
    else:
        wlim = np.array([wlo, whi])
    # Pad by nsigma * sigma_wave
    if linear:
        wlim += nsigma_pad * R * np.array([-1, 1])
    else:
        wlim *= (1 + nsigma_pad / R * np.array([-1, 1]))
    mask = (wavelength > wlim[0]) & (wavelength < wlim[1])
    return mask


def resample_wave(wavelength, spectrum, linear=False):
    """Resample spectrum, so that the number of elements is the next highest
    power of two.  Assumes the input spectrum is constant velocity resolution
    unless ``linear`` is True, in which case it is assumed that the spectrum has
    constant wavelength resolution (e.g. 1AA)
    """
    wmin, wmax = wavelength.min(), wavelength.max()
    nw = len(wavelength)
    nnew = 2.0**(np.ceil(np.log2(nw)))
    if linear:
        Rgrid = np.diff(wavelength) # in same units as ``wavelength`` 
        w = np.linspace(wmin, wmax, nnew)
    else:
        Rgrid = np.diff(np.log(wavelength))  # actually 1/R
        lnlam = np.linspace(np.log(wmin), np.log(wmax), nnew)
        w = np.exp(lnlam)
    # Make sure the resolution really is nearly constant
    assert Rgrid.max() / Rgrid.min() < 1.05
    s = np.interp(w, wavelength, spectrum)
    return w, s


def smooth_vel_fft(wavelength, spectrum, outwave=None,
                   Rout=20000, Rin=3e5, **extras):
    """wave, spec: A log-lam gridded spectrum
    Rin: wave / sigma_wave
    Rout: wave / sigma_wave
    """
    
    # restrict wavelength range (for speed)
    # should also make power of 2
    mask = mask_wave(wavelength, outwave=outwave, R=Rout, **extras)    
    wave, spec = resample_wave(wavelength[mask], spectrum[mask])

    # The kernel width for the convolution.
    sigma_out, sigma_in = ckms / Rout, ckms / Rin
    sigma = np.sqrt(sigma_out**2 - sigma_in**2)
    if sigma < 0:
        raise(ValueError)
    
    # get grid resolution (*not* the resolution of the input spectrum) and make
    # sure it's nearly constant.  Should be by design (see resample_wave above)
    invRgrid =  np.diff(np.log(wave))
    assert invRgrid.max() / invRgrid.min() < 1.05
    dv = ckms * np.median(invRgrid)

    # Do the convolution
    spec_conv = smooth_fft(dv, spec, sigma)
    # interpolate onto output grid
    print(len(wave), len(spec_conv), len(spec))
    if outwave is not None:
        spec_conv = np.interp(outwave, wave, spec_conv)

    return spec_conv


def smooth_wave_fft(wavelength, spectrum, outwave=None,
                    sigma_out=1.0, sigma_in=0.0, **extras):
    """Wave, spec: a linear-lam gridded spectrum.
    """
    # restrict wavelength range (for speed)
    # should also make nearest power of 2
    mask = mask_wave(wavelength, outwave=outwave, R=sigma_out, linear=True, **extras)
    wave, spec = resample_wave(wavelength[mask], spectrum[mask], linear=True)
    
    # The kernel width for the convolution.
    sigma = np.sqrt(sigma_out**2 - sigma_in**2)
    if sigma < 0:
        raise(ValueError)
    
    # get grid resolution (*not* the resolution of the input spectrum) and make
    # sure it's nearly constant.  Should be by design (see resample_wave above)
    Rgrid = np.diff(wave)
    assert Rgrid.max() / Rgrid.min() < 1.05
    dw = np.median(Rgrid)
    
    # Do the convolution
    spec_conv = smooth_fft(dw, spec, sigma)
    # interpolate onto output grid
    if outwave is not None:
        spec_conv = np.interp(outwave, wave, spec_conv)
    return spec_conv
    
def smooth_fft(dx, spec, sigma):
    """
    :param dx:
        The wavelength or velocity spacing, same units as sigma

    :param sigma:
        The width of the gaussian kernel, same units as dx

    :param spec:
        The spectrum flux vector
    """
    # The Fourier coordinate
    ss = rfftfreq(len(spec), d=dx)
    # Make the fourier space taper
    taper = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (ss ** 2))
    ss[0] = 0.01  # hack
    # Fourier transform the spectrum
    spec_ff = np.fft.rfft(spec)
    # Multiply in fourier space
    ff_tapered = spec_ff * taper
    # Fourier transform back
    spec_conv = np.fft.irfft(ff_tapered)
    return spec_conv
