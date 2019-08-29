#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Module for producing full range spectra from c3k .spec and .flux files (or
rather their hdf5 versions) This involves convolving the appropriate spectra by
the appropriate amounts for a set of segements and stitching them together. 
Note that the .flux files actually are top-hat filtered versions of the spectra,
with the value at each point giving the sum of total flux within some +/- from
that point, and have some weird effective resolution.
"""
import numpy as np
import h5py, json
from prospect.utils.smoothing import smoothspec
from .utils import construct_outwave, sigma_to_fwhm, param_order

__all__ = ["make_seds", "make_one_sed"]


def make_seds(specfile, fluxfile, segments=[()],
              specres=3e5, fluxres=500, outname=None,
              verbose=True, oversample=2):
    """
    Parameters
    -------
    specfile : Dict-like with keys "parameters", "wavelengths", and "spectra"
        Handle to the HDF5 file containting the high-resolution C3K spectra

    fluxfile: Dict-like with keys "parameters", "wavelengths", and "spectra"
        Handle to the HDF5 file containing the low-resolution C3K "flux"
        spectra, which is assumed matched line by line to the specfile

    segments : list of 4-tuples
        A list of 4-tuples describing the wavelength segments and their
        resolution, and whether to use FFT (not recommended near the edge o the
        hires file).  Each tuple should have the form (lo, hi, R, bool)

    Returns
    ------
    Generates an HDF file at `outname`
    """
    # --- Wavelength arrays ----
    swave = np.array(specfile["wavelengths"])
    fwave = np.array(fluxfile["wavelengths"])
    outwave = [construct_outwave(lo, hi, resolution=rout, logarithmic=True,
                                 oversample=oversample)[:-1]
               for (lo, hi, rout, _) in segments]
    outwave = np.concatenate(outwave)
    assert np.all(np.diff(outwave) > 0), "Output wavelength grid is not ascending!"
    nw = len(outwave)

    # --- Match specfile to fluxfile ----
    # this is probably a stupid way to do this.
    sind, find = [], []
    for i, spec in enumerate(specfile["spectra"]):
        # find the matching flux entry
        params = specfile["parameters"][i]
        ind = np.array([fluxfile["parameters"][f] == params[f]
                        for f in params.dtype.names])
        ind = ind.prod(axis=0).astype(int)
        if ind.sum() != 1:
            msg = "could not find unique flux spectrum @ params {}"
            print(msg.format(dict(zip(param_order, params))))
        else:
            sind.append(i)
            find.append(ind.tolist().index(1))
    sind = np.array(sind)
    find = np.array(find)

    # --- Setup the output h5 file ---
    if outname is not None:
        out = h5py.File(outname, "w")
        partype = specfile["parameters"].dtype
        sedout = out.create_dataset('spectra', shape=(0, nw),
                                    maxshape=(None, nw))
        parsout = out.create_dataset('parameters', shape=(0,),
                                    maxshape=(None,), dtype=partype)
        wavelength = out.create_dataset('wavelengths', data=outwave)
        out.attrs["segments"] = json.dumps(segments)
        out.attrs["segments_desc"] = "(lo, hi, R_fwhm, FFT)"
    else:
        sedout = np.zeros([nsed, nw])
        parsout = np.zeros([nsed], dtype=partype)

    #  --- Fill H5 file ---
    # loop over spectra convolving segments and getting the SEDs, and putting
    # them in the SED file
    #matches = zip(specfile["parameters"][sind], specfile["spectra"][sind, :], fluxfile["spectra"][find, :])
    for i, (s, f) in enumerate(zip(sind, find)):
        spec = specfile["spectra"][s, :]
        flux = fluxfile["spectra"][f, :]
        wave, sed = make_one_sed(swave, spec, fwave, flux, segments, oversample=oversample,
                                 specres=specres, fluxres=fluxres, verbose=verbose)
        assert len(sed) == nw, ("SED is not the same length as the desired "
                                "output wavelength grid! ({} != {})".format(len(sed), len(outwave)))

        sed[np.isnan(sed)] = 0.0
        if outname is not None:
            sedout.resize(i+1, axis=0)
            parsout.resize(i+1, axis=0)
        sedout[i, :] = sed
        parsout[i] = specfile["parameters"][s]
        #out.flush()

    if outname is not None:
        out.close()
        return
    else:
        return np.array(parsout), np.array(sedout)


def make_one_sed(swave, spec, fwave, flux, segments=[()], clip=1e-33,
                 specres=3e5, fluxres=5000., oversample=2, verbose=True):
    """
    Parameters
    -----------
    swave : ndarray of shape (nwh,)
        Full resolution spectrum wavelength vector for the `.spec` output of synthe.

    spec : ndarray of shape (nws,)
        Full resolution flux vector.

    fwave : ndarray of shape (nwl,)
        Wavelength vector of the lower resolution `flux` spectrum provided by synthe.

    flux : ndarray of shape (nwl,)
        Low resolution flux vector, same units as `spec`

    segments: list of tuples
        Specification of the wavelength segments, of the form
        (wave_min, wave_max, resolution, use_fft)

    clip : float (default: 1e-33)
        Lower limit for the output SED fluxes

    specres : float (default: 3e5)
        The resolution of the input spectrum given by the `swave` and `spec` vectors.

    fluxres : float (default: 5e3)
        The resolution of the input spectrum given by the `fwave` and `flux` vectors.

    oversample : float (default, 2)
        Number of output pixels per FWHM of the line-spread function.

    Returns
    --------

    outwave : ndarray of shape (nout,)
        Wavelength vector of the output downsampled SED

    sed : ndarray of shape (nout,)
        Flux vector of the output downsampled SED (same units as `spec` and `flux`)
    """
    sed = []
    outwave = []
    for j, (lo, hi, rout, fftsmooth) in enumerate(segments):
        # get the output wavelength vector for this segment, throwing away
        # the last point (which will be the same as the first of the next
        # segment)
        out = construct_outwave(lo, hi, resolution=rout, logarithmic=True, oversample=oversample)[:-1]
        # Do we use the hires or lores spectrum?
        if (lo > swave.min()) and (hi < swave.max()):
            inspec = spec
            inres = specres
            inwave = swave
            msg = "using hires for {} - {} @ R={}".format(lo, hi, rout)
        else:
            inspec = flux
            inres = fluxres
            inwave = fwave
            msg = "using lores for {} - {} @ R={}".format(lo, hi, rout)

        if fftsmooth:
            msg += "; using FFT"
        if verbose:
            print(msg)
        assert rout <= inres, "You are trying to smooth to a higher resolution than C3K provides!"
        # account for C3K resolution
        rsmooth = (rout**(-2.) - inres**(-2))**(-0.5)
        # convert to lambda/sigma_lambda
        rsmooth *= sigma_to_fwhm
        s = smoothspec(inwave, inspec, rsmooth, smoothtype="R", outwave=out, fftsmooth=fftsmooth)
        if clip > 0:
            np.clip(s, clip, np.inf, out=s)

        sed.append(s)
        outwave.append(out)

    outwave = np.concatenate(outwave)
    sed = np.concatenate(sed)
    # now replace the lambda > fwave.max() (plus some padding) with a BB
    # Assumes units are fnu
    fwave_max = np.max(fwave[flux > 0])
    ind_max = np.searchsorted(outwave, fwave_max)
    sed[ind_max-9:] = sed[ind_max - 10] * (outwave[ind_max - 10] / outwave[ind_max-9:])**2


    return outwave, sed


if __name__ == "__main__":

    # lambda_lo, lambda_hi, R_{out, fwhm}, use_fft
    segments = [(100., 910., 250., False),
                (910., 2800., 250., False), 
                (2800., 7000., 5000., True),
                (7000., 2.0e4, 500., True),
                (2.0e4, 1e8, 50., False)
                ]

    from .utils import get_ckc_parser
    # key arguments are:
    #  * --feh
    #  * --afe
    #  * --ck_vers
    #  * --basedir
    #  * --sedname
    parser = get_ckc_parser()
    args = parser.parse_args()

    # --- Filenames ----
    template = "{}/{}_feh{:+3.2f}_afe{:+2.1f}.{}.h5"
    specname = template.format(args.basedir, args.ck_vers, args.feh, args.afe, "full")
    fluxname = template.format(args.basedir, args.ck_vers, args.feh, args.afe, "flux")
    outname = template.format(args.basedir, args.ck_vers, args.feh, args.afe, args.sedname)

    msg = "Reading from {}\nReading from {}\nWriting to {}".format(specname, fluxname, outname)
    print(msg)

    # --- Wavelength segments ---
    segments = segments
    msg = "Using the following parameters:\n"
    for seg in segments:
        msg += "lo={}AA, hi={}AA, R_fwhm={}, FFT={}\n".format(*seg)
    print(msg)

    # --- Read Files and make the sed file ---
    specfile = h5py.File(specname, "r")
    fluxfile = h5py.File(fluxname, "r")
    make_seds(specfile, fluxfile, fluxres=5e3, outname=outname, segments=segments)
