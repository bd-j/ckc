# Module for producing full range spectra from c3k .spec and .flux files (or
# rather their hdf5 versions)
# This involves convolving the appropriate spectra by the appropriate amounts
# for a set of segements and stitching them together.
# Note that the .flux files actually are top-hat filtered versions of the
# spectra, with the value at each point giving the sum of total flux within
# some +/- from that point, and have some weird effective resolution.
import numpy as np
import h5py, json
from prospect.utils.smoothing import smoothspec


param_order = ['logt', 'logg', 'feh', 'afe']


# lambda_lo, lambda_hi, R_{out, fwhm}, use_fft
segments = [(100., 910., 250., False),
            (910., 2800., 250., False), 
            (2800., 7000., 5000., True),
            (7000., 2.0e4, 500., True),
            (2.0e4, 1e8, 50., False)
            ]

sigma_to_fwhm = 2 * np.sqrt(2 * np.log(2.))


def make_seds(specfile, fluxfile, segments=segments, specres=3e5, fluxres=500, outname=''):
    """
    :params specfile:
        Handle to the HDF5 file containting the high-resolution C3K spectra

    :params fluxfile:
        Handle to the HDF5 file containing the low-resolution C3K "flux"
        spectra, which is assumed matched line by line to the specfile

    :params segments:
        A list of 4-tuples describing the wavelength segments and their
        resolution, and whether to use FFT (not recommended near the edge o the
        hires file).  Each tuple should have the form (lo, hi, R, bool)
    """
    # --- Wavelength arrays ----
    swave = np.array(specfile["wavelengths"])
    fwave = np.array(fluxfile["wavelengths"])
    outwave = [construct_outwave(lo, hi, resolution=rout, logarithmic=True, oversample=2)[:-1]
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
            print("could not find unique flux spectrum @ params {}".format(dict(zip(param_order, params))))
        else:
            sind.append(i)
            find.append(ind.tolist().index(1))
    sind = np.array(sind)
    find = np.array(find)

    # --- Setup the output h5 file ---
    out = h5py.File(outname, "w")
    partype = specfile["parameters"].dtype
    sedout = out.create_dataset('spectra', shape=(0, nw),
                                 maxshape=(None, nw))
    parsout = out.create_dataset('parameters', shape=(0,),
                                maxshape=(None,), dtype=partype)
    wavelength = out.create_dataset('wavelengths', data=outwave)
    out.attrs["segments"] = json.dumps(segments)
    out.attrs["segments_desc"] = "(lo, hi, R_fwhm, FFT)"

    #  --- Fill H5 file ---
    # loop over spectra convolving segments and getting the SEDs, and putting
    # them in the SED file
    #matches = zip(specfile["parameters"][sind], specfile["spectra"][sind, :], fluxfile["spectra"][find, :])
    for i, (s, f) in enumerate(zip(sind, find)):
        spec = specfile["spectra"][s, :]
        flux = fluxfile["spectra"][f, :]
        wave, sed = make_one_sed(swave, spec, fwave, flux, segments,
                                 specres=specres, fluxres=fluxres)
        assert len(sed) == nw, ("SED is not the same length as the desired "
                                "output wavelength grid! ({} != {})".format(len(sed), len(outwave)))

        sed[np.isnan(sed)] = 0.0
        sedout.resize(i+1, axis=0)
        parsout.resize(i+1, axis=0)
        sedout[i, :] = sed
        parsout[i] = specfile["parameters"][s]
        out.flush()

    out.close()


def make_one_sed(swave, spec, fwave, flux, segments=segments,
                 specres=3e5, fluxres=500, oversample=2):
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
            print("using hires for {} - {} @ R={}".format(lo, hi, rout))
        else:
            inspec = flux
            inres = fluxres
            inwave = fwave
            print("using lores for {} - {} @ R={}".format(lo, hi, rout))

        if fftsmooth:
            print("using FFT")
        assert rout <= inres, "You are trying to smooth to a higher resolution than C3K provides!"
        # account for C3K resolution
        rsmooth = (rout**(-2.) - inres**(-2))**(-0.5)
        # convert to lambda/sigma_lambda
        rsmooth *= sigma_to_fwhm
        s = smoothspec(inwave, inspec, rsmooth, smoothtype="R", outwave=out, fftsmooth=fftsmooth)

        sed.append(s)
        outwave.append(out)

    outwave = np.concatenate(outwave)
    sed = np.concatenate(sed)
    # now replace the lambda > fwave.max() (plus some padding) with a BB
    fwave_max = np.max(fwave[flux > 0])
    ind_max = np.searchsorted(outwave, fwave_max)
    sed[ind_max-9:] = sed[ind_max - 10] * (outwave[ind_max - 10] / outwave[ind_max-9:])**2


    return outwave, sed


def construct_outwave(min_wave_smooth=0.0, max_wave_smooth=np.inf,
                      dispersion=1.0, oversample=2.0,
                      resolution=3e5, logarithmic=False,
                      **extras):
    """Given parameters describing the output spectrum, generate a wavelength
    grid that properly samples the resolution.
    """
    if logarithmic:
        # critically sample the resolution
        dlnlam = 1.0 / resolution / oversample  
        lnmin, lnmax = np.log(min_wave_smooth), np.log(max_wave_smooth)
        #print(lnmin, lnmax, dlnlam, resolution, oversample)
        out = np.exp(np.arange(lnmin, lnmax + dlnlam, dlnlam))
    else:
        out = np.arange(min_wave_smooth, max_wave_smooth,
                        dispersion / oversample)
    return out    



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feh", type=float, default=0.0,
                        help=("The feh value to process."))
    parser.add_argument("--afe", type=float, default=0.0,
                        help=("The afe value to process."))
    parser.add_argument("--ck_vers", type=str, default='c3k_v1.3',
                        help=("Name of directory that contains the "
                              "version of C3K spectra to use."))
    parser.add_argument("--basedir", type=str, default='fullres/c3k/',
                        help=("Path to the directory containing fullres and "
                              "flux HDF5 files. Output will be placed here too."))
    parser.add_argument("--sedname", type=str, default="sed",
                        help=("nickname for the SED file, e.g. sedR500"))
    #parser.add_argument("--specname", type=str, default="",
    #                    help=("Full name and path to the spec HDF5 file."))
    #parser.add_argument("--fluxname", type=str, default="",
    #                    help=("Full name and path to the flux HDF5 file."))

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
