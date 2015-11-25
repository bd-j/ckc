# import packages
import numpy as np
from numpy.fft import fft, ifft, fftfreq, rfftfreq

ckms = 2.998e5
sigma_to_fwhm = 2.355


def smooth_vel_fft(wavelength, spectrum, outwave=None,
                   wlo=0, whi=np.inf, nsigma_pad=10,
                   Rout=20000, Rin=3e5):
    """wave, spec: A log-lam gridded spectrum
    Rin: wave / sigma_wave
    Rout: wave / sigma_wave
    """

    # restrict wavelength range (for speed)
    # should also make power of two
    wlim = np.array([wlo, whi]) * (1 + nsigma_pad / Rout * np.array([-1, 1]))
    g = (wavelength > wlim[0]) & (wavelength < wlim[1])
    wave = wavelength[g]
    spec = spectrum[g]

    # The kernel width for the convolution.
    sigma_out, sigma_in = ckms / Rout, ckms / Rin
    sigma = np.sqrt(sigma_out**2 - sigma_in**2)
    
    # get grid resolution and make sure its nearly constant.
    Rgrid = ((wave[:-1] + wave[1:])/2.) / np.diff(wave)
    assert Rgrid.max() / Rgrid.min() < 1.05
    dv = ckms / np.median(Rgrid)

    # Do the convolution
    spec_conv = smooth_fft(dv, spec, sigma)
    # interpolate onto output grid
    if outwave is not None:
        spec_conv = np.interp(outwave, wave[:-1], spec_conv)

    return spec_conv


def smooth_wave_fft(wavelength, spectrum, outwave=None,
                    wlo=0, whi=np.inf, nsigma_pad=10,
                    sigma_out=1.0, sigma_in=0.0):
    """Wave, spec: a spectrum evenly gridded in linear-lam.
    """
    # restrict wavelength range (for speed)
    # should also make nearest power of two
    wlim = np.array([wlo, whi]) + nsigma_pad * sigma_out * np.array([-1, 1])
    g = (wavelength > wlim[0]) & (wavelength < wlim[1])
    wave = wavelength[g]
    spec = spectrum[g]
    
    # The kernel width for the convolution.
    sigma = np.sqrt(sigma_out**2 - sigma_in**2)
    
    # get grid resolution and make sure its nearly constant.
    Rgrid = np.diff(wave)
    assert Rgrid.max() / Rgrid.min() < 1.05
    dw = np.median(Rgrid)
    
    # Do the convolution
    spec_conv = smooth_fft(dw, spec, sigma)
    print(len(spec), len(wave), len(spec_conv))
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
