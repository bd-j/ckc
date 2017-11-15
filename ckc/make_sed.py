import numpy as np
import h5py, json
from prospect.utils.smoothing import smoothspec, construct_outwave


param_order = ['logt', 'logg', 'feh', 'afe']


# lambda_lo, lambda_hi, R_{out, fwhm}
segments = [(100., 2800., 250.),
            (2800., 7000., 5000.,),
            (7000., 2.5e4, 500.),
            (2.5e4, 1e8, 50.)
            ]


def make_seds(specfile, fluxfile, segments=segments, specres=3e5, fluxres=500, outname=''):
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
    # --- Wavelength arrays ----
    swave = np.array(specfile["wavelengths"])
    fwave = np.array(fluxfile["wavelengths"])
    outwave = [construct_outwave(lo, hi, resolution=rout, logarithmic=True, oversample=2)[:-1]
               for (lo, hi, rout) in segments]
    outwave = np.concatenate(outwave)
    assert np.all(np.diff(outwave) > 0), "Output wavelength grid is not scending!"
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

    #  --- Fill H5 file ---
    # loop over spectra convolving segments and getting the SEDs, and putting
    # them in the SED file
    matches = zip(specfile["parameters"][sind], specfile["spectra"][sind], fluxfile["spectra"][find])
    for i, (pars, spec, flux) in enumerate(matches):
        wave, sed = make_one_sed(swave, spec, fwave, flux, segments,
                                 specres=specres, fluxres=fluxres)
        assert len(sed) == nw, ("SED is not the same length as the desired "
                                "output wavelength grid! ({} != {})".format(len(sed), len(outwave)))

        sedout.resize(i+1, axis=0)
        parsout.resize(i+1, axis=0)
        sedout[i, :] = sed
        parsout[i] = pars
        out.flush()

    out.close()


def make_one_sed(swave, spec, fwave, flux, segments=segments,
                 specres=3e5, fluxres=500):
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
