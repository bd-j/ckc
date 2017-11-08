import numpy as np
import h5py
from prospect.utils.smoothing import smoothspec

# lambda_lo, lambda_hi, R_{out, fwhm}
segments = [(100., 2800., 250.),
            (2800., 7000., 5000.,),
            (7000., 2.5e4, 500.),
            (2.5e4, 1e8, 50.)
            ]


def make_seds(specfile, fluxfile, segments, specres=3e5, fluxres=4340):
    """
    :params specfile:
        Handle to the HDF5 file containting the high-resolution C3K spectra

    :params fluxfile:
        Handle to the HDF5 file containing the low-resolution C3K "flux"
        spectra, which is assumed matched line by line to the specfile

    :params segments:
        A list of 3-tuples describing the wavelength segments and their
        resolution.  Each tuple should have the form (lo, hi, R)
    """

    # Wavelength arrays
    swave = np.array(specfile["wavelengths"])
    fwave = np.array(fluxfile["wavelengths"])
    outwave = [construct_outwave(lo, hi, rout, logarithmic=True, oversample=2)[:-1]
               for (lo, hi, rout) in segments]
    outwave = np.array(outwave)
    assert np.all(np.diff(outwave)) > 0, "Output wavelength grid is not scending!"

    # loop over spectra convolving segments and getting the SEDs
    for i, spec, flux in enumerate(zip(specfile["spectra"], fluxfile["spectra"])):
        wave, sed = make_one_sed(swave, spec, fwave, flux, segments,
                                 specres=specres, fluxres=fluxres)
        assert len(sed) == len(outwave), ("SED is not the same length as the desired "
                                          "output wavelength grid! ({} != {})".format(len(sed), len(outwave)))

def make_one_sed(swave, spec, fwave, flux, segments,
                 specres=3e5, fluxres=):
    sed = []
    outwave = []
    for j, (lo, hi, rout) in enumerate(segments):
        # get the output wavelength vector for this segment, throwing away
        # the last point (which will be the same as the first of the next
        # segment)
        out = construct_outwave(lo, hi, rout, logarithmic=True, oversample=2)[:-1]
        # do we use the highres or lores spectrum?
        if (lo > swave.min()) and (hi < swave.max()):
            inspec = spec
            inres = specres
            inwave = swave
        else:
            inspec = flux
            inres = fluxres
            inwave = fwave
        assert rout < inres, "You are trying to smooth to a higher resolution than C3K provides!"
        # account for C3K resolution
        rsmooth = (rout**(-2.) - inres**(-2))**(-0.5)
        # convert to lambda/sigma_lambda
        rsmooth *= sigma_to_fwhm  
        s = smoothspec(inwave, inspec, rsmooth, smoothtype="R", outwave=out)
        sed.append(s)
        outwave.append(out)

    return np.concatenate(outwave), np.concatenate(sed)
