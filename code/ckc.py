import numpy as np
import struct, glob

ckc_dir = '/Users/bjohnson/Codes/SPS/ckc/'


def construct_outwave(resolution, wlo, whi, velocity=True,
                      absminwave=100, absmaxwave=1e8, **extras):
    """Given a spectral range of interest and a resolution in that
    range, construct wavelength vectors and resolution intervals that will
    cover this range at the desired reolution, but also ranges outside
    this at lower resolution (suitable for photometry, ionizing flux,
    and dust emission calculations)
    """
    if velocity:
        lores = 100  # R
    else:
        lores = 30  # AA
    wave = [(absminwave, wlo), (wlo, whi), (whi, absmaxwave)]
    res = [lores, resolution, lores]

    out = []
    for (wmin, wmax), r in zip(wave, res):
        if velocity:
            dlnlam = 1.0/r/2  # critically sample the resolution
            out += [np.exp(np.arange(np.log(wmin), np.log(wmax), dlnlam))]
        else:
            dlam = r/2.0  # critically sample the resolution
            out += [np.arange(wmin, wmax, dlam)]
    return out, res


def spec_params(expanded=False, **extras):
    """Get paramaters (Z, g, T) for the CKC library.
    """
    zlegend = np.loadtxt('{0}/zlegend.dat'.format(ckc_dir))
    logg = np.loadtxt('{0}/basel_logg.dat'.format(ckc_dir))
    logt = np.loadtxt('{0}/basel_logt.dat'.format(ckc_dir))
    if not expanded:
        return zlegend, logg, logt
    else:
        dt = [('Z', np.float), ('logg', np.float), ('logt', np.float)]
        pars = np.empty(len(zlegend) * len(logg) * len(logt), dtype=dt)
        i = 0
        for z in zlegend:
            for g in logg:
                for t in logt:
                    pars[i] = (z, g, t)
                    i += 1
        return pars


def read_and_downsample_spectra(outwave, outres,
                                velocity=True, write_binary=False,
                                binout='ckc_new_z{0}.bin', **kwargs):
    """Loop through the metallicities of the CKC library, read the
    binary files, downsample them, and return an array of shape (nz,
    ng*nt, nw)
    """
    newspec = []
    zlegend, logg, logt = spec_params()
    zlist = ['{0:06.4f}'.format(z) for z in zlegend]
    # names = glob.glob('{0}/bin/ckc14_z*.spectra.bin'.format(ckc_dir))
    for z in zlist:
        zspec = read_and_downsample_onez(z, outwave, outres, binout=binout,
                                         write_binary=write_binary, **kwargs)
        newspec.append(zspec)

    return np.array(newspec)


def read_and_downsample_onez(z, outwave, outres, velocity=True,
                             binout='ckc_new_z{0}.bin',
                             write_binary=False, **kwargs):
    """Read one of the CKC binary files, loop through the spectra in
    it, downsample them, optionally write to a new binary file, and
    return the spectra.
    """
    if write_binary:
        outfile = open(binout.format(z), 'wb')
    name = '{0}/bin/ckc14_z{1}.spectra.bin'.format(ckc_dir, z)

    wave = np.loadtxt('{0}/ckc14.lambda'.format(ckc_dir))
    zlegend, logg, logt = spec_params()
    nw = len(wave)
    nspec = len(logg) * len(logt)

    newspec = []
    specs = read_binary_spec(name, nw, nspec)
    for i, spec in enumerate(specs):
        outspec = downsample_onespec(wave, spec, outwave, outres,
                                     velocity=velocity, **kwargs)
        ospec = np.concatenate(outspec)
        if write_binary:
            for flux in ospec:
                outfile.write(struct.pack('f', flux))

        newspec.append([ospec])
    return np.array(newspec)


def downsample_onespec(wave, spec, outwave, outres,
                       velocity=True, **kwargs):
    outspec = []
    # loop over the output segments
    for owave, ores in zip(outwave, outres):
        wmin, wmax = owave.min(), owave.max()
        if velocity:
            sigma = 2.998e5 / ores  # in km/s
            smin = wmin - 5 * wmin/ores
            smax = wmax + 5 * wmax/ores
        else:
            sigma = ores  # in AA
            smin = wmin - 5 * sigma
            smax = wmax + 5 * sigma
        imin = np.argmin(np.abs(smin - wave))
        imax = np.argmin(np.abs(smax - wave))
        ospec = smooth(wave[imin:imax], spec[imin:imax], sigma,
                       velocity=velocity, outwave=owave, **kwargs)
        outspec += [ospec]
    return outspec


def read_binary_spec(filename, nw, nspec):
    count = 0
    spec = np.empty([nspec, nw])
    with open(filename, 'rb') as f:
        while count < nspec:
                count += 1
                for iw in range(nw):
                    byte = f.read(4)
                    spec[count-1, iw] = struct.unpack('f', byte)[0]
    return spec


def binary_to_hdf(hname):
    import h5py
    wave = np.loadtxt('{0}/ckc14.lambda'.format(ckc_dir))
    nw = len(wave)
    zlegend, logg, logt = spec_params()
    nspec = len(logg) * len(logt)

    with h5py.File(hname, "w") as f:
        spgr = f.create_group('spectra')
        fw = f.create_dataset('wavelengths', data=wave)
        fg = f.create_dataset('logg', data=logg)
        ft = f.create_dataset('logt', data=logt)

        zlist = ['{0:06.4f}'.format(z) for z in zlegend]
        for z in zlist:
            name = '{0}/bin/ckc14_z{1}.spectra.bin'.format(ckc_dir, z)
            spec = read_binary_spec(name, nw, nspec)
            spec = spec.reshape(len(logg), len(logt), nw)
            fl = spgr.create_dataset('z{0}'.format(z), data=spec)
            f.flush()


def resample_ckc(R=3000, wmin=3500, wmax=10000, velocity=True,
                 outname='test.h5', **extras):
        """Non-working method
        """
        import h5py

        outwave, outres = construct_outwave(R, wmin, wmax, velocity=velocity)
        wave = np.concatenate(outwave)
        params = spec_params(expanded=True)
        spectra = read_and_downsample_spectra(outwave, outres,
                                              velocity=velocity, **extras)
        with h5py.File(outname, 'r') as f:
                dspec = f.create_dataset("spectra", data=spectra)
                dwave = f.create_dataset("wavelengths", data=wave)
                dpar = f.create_dataset("parameters", data=params)


def wave_from_ssp():
    out = open('ckc14.lambda', 'w')
    fname = 'SSP_Padova_CKC14_Salpeter_Z0.0002.out.spec'
    f = open(fname, "r")
    for i in range(9):
        j = f.readline()
    wave = f.readline()
    for w in wave.split():
        out.write(float(w), '\n')
    f.close()
    out.close()


def find_segments(wave, restol=0.1):
    """ Find places where the resolution of a spectrum changes by
    using changes in the wavelength sampling.

    :param restol:
        Fractional resolution change between adjacent elements that is
        used to define a new segment.

    :returns segments:
        A list of tuples contaning the lower and upper index of each
        segement and the average `resolution` (lambda/dlambda)
    """

    dwave = np.diff(wave)
    mwave = wave[:-1] + dwave/2
    res = mwave/dwave
    res = np.array(res.tolist() + [res[-1]])
    dres = np.diff(res)
    breakpoints = np.where(abs(dres/res[:-1]) > restol)[0] + 1
    lims = [0] + breakpoints.tolist() + [len(wave)]
    segments = []
    for i in range(len(lims) - 1):
        segments.append((lims[i], lims[i+1], res[lims[i]:lims[i+1]].mean()))
    return segments


def smooth(wave, spec, sigma, velocity=True, **kwargs):
    if velocity:
        return smooth_vel(wave, spec, sigma, **kwargs)
    else:
        return smooth_wave(wave, spec, sigma, **kwargs)


def smooth_vel(wave, spec, sigma, outwave=None, inres=0,
               nsigma=10):
    """Smooth a spectrum in velocity space.  This is insanely slow,
    but general and correct.

    :param sigma:
        desired velocity resolution (km/s)

    :param nsigma:
        Number of sigma away from the output wavelength to consider in
        the integral.  If less than zero, all wavelengths are used.
        Setting this to some positive number decreses the scaling
        constant in the O(N_out * N_in) algorithm used here.
    """
    sigma_eff = np.sqrt(sigma**2 - inres**2)/2.998e5
    if outwave is None:
        outwave = wave
    if sigma <= 0.0:
        return np.interp(wave, outwave, flux)

    lnwave = np.log(wave)
    flux = np.zeros(len(outwave))
    # norm = 1/np.sqrt(2 * np.pi)/sigma
    maxdiff = nsigma * sigma

    for i, w in enumerate(outwave):
        x = np.log(w) - lnwave
        if nsigma > 0:
            good = (x > -maxdiff) & (x < maxdiff)
            x = x[good]
            _spec = spec[good]
        else:
            _spec = spec
        f = np.exp(-0.5 * (x / sigma_eff)**2)
        flux[i] = np.trapz(f * _spec, x) / np.trapz(f, x)
    return flux


def smooth_wave(wave, spec, sigma, outwave=None,
                inres=0, in_vel=False, **extras):
    """Smooth a spectrum in wavelength space.  This is insanely slow,
    but general and correct (except for the treatment of the input
    resolution if it is velocity)

    :param sigma:
        Desired reolution in wavelength units

    :param inres:
        Resolution of the input, in either wavelength units or
        lambda/dlambda (c/v)

    :param in_vel:
        If True, the input spectrum has been smoothed in velocity
        space, and inres is in dlambda/lambda.
    """
    if outwave is None:
        outwave = wave
    if inres <= 0:
        sigma_eff = sigma
    elif in_vel:
        sigma_min = np.max(outwave)/inres
        if sigma < sigma_min:
            raise ValueError("Desired wavelength sigma is lower "
                             "than the value possible for this input "
                             "spectrum ({0}).".format(sigma_min))
        # Make an approximate correction for the intrinsic wavelength
        # dependent dispersion.  This doesn't really work.
        sigma_eff = np.sqrt(sigma**2 - (wave/inres)**2)
    else:
        if sigma < inres:
            raise ValueError("Desired wavelength sigma is lower "
                             "than the value possible for this input "
                             "spectrum ({0}).".format(sigma_min))
        sigma_eff = np.sqrt(sigma**2 - inres**2)

    flux = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        x = (wave-w)/sigma_eff
        f = np.exp(-0.5 * x**2)
        flux[i] = np.trapz(f * spec, wave) / np.trapz(f, wave)
    return flux
