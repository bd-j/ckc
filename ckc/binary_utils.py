import numpy as np
import struct, glob

__all__ = ["downsample_onespec", "smooth", "smooth_vel", "smooth_wave",
           "read_and_downsample_onez", "read_and_downsample_spectra",
           "binary_to_hdf", "read_binary_spec",
           "construct_outwave", "ckc_params", "find_segments"]

ckc_dir = '/Users/bjohnson/Codes/SPS/ckc/ckc/'


def smooth(wave, spec, sigma, velocity=True, **kwargs):
    """Smooth a spectrum in velocity or wavelength space.
    """
    if velocity:
        return smooth_vel(wave, spec, sigma, **kwargs)
    else:
        return smooth_wave(wave, spec, sigma, **kwargs)


def smooth_vel(wave, spec, sigma, outwave=None,
               inres=0, nsigma=10, **extras):
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
    if sigma_eff <= 0.0:
        return np.interp(outwave, wave, spec)

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
                inres=0, in_vel=False, nsigma=10,
                **extras):
    """Smooth a spectrum in wavelength space.  This is insanely slow,
    but general and correct (except for the treatment of the input
    resolution if it is velocity)

    :param sigma:
        Desired resolution (*not* FWHM) in wavelength units.

    :param inres:
        Resolution of the input, in either wavelength units or
        velocity.  This is sigma units, not FWHM

    :param in_vel:
        If True, the input spectrum has been smoothed in velocity
        space, and ``inres`` is in km/s (sigma not FWHM).

    :param nsigma: (default=10)
        Number of sigma away from the output wavelength to consider in
        the integral.  If less than zero, all wavelengths are used.
        Setting this to some positive number decreses the scaling
        constant in the O(N_out * N_in) algorithm used here.
    """
    if outwave is None:
        outwave = wave

    if inres <= 0:
        sigma_eff = sigma
    elif in_vel:
        R_in = (2.998e5 / inres)
        sigma_min = np.max(outwave) / R_in
        if sigma < sigma_min:
            raise ValueError("Desired wavelength sigma is lower "
                             "than the value possible for this input "
                             "spectrum ({0}).".format(sigma_min))
        # Make an approximate correction for the intrinsic wavelength
        # dependent dispersion.  This doesn't really work.
        sigma_eff = np.sqrt(sigma**2 - (wave / R_in)**2)
    else:
        if sigma < inres:
            raise ValueError("Desired wavelength sigma is lower "
                             "than the value possible for this input "
                             "spectrum ({0}).".format(sigma_min))
        sigma_eff = np.sqrt(sigma**2 - inres**2)

    flux = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        x = (wave - w) / sigma_eff
        if nsigma > 0:
            good = np.abs(x) < nsigma
            x = x[good]
            _spec = spec[good]
            _wave = wave[good]
        else:
            _spec = spec
            _wave = wave
        f = np.exp(-0.5 * x**2)
        flux[i] = np.trapz(f * _spec, _wave) / np.trapz(f, _wave)
    return flux


def downsample_onespec(wave, spec, outwave, outres, inres=None,
                       velocity=True, nsigma=10, **kwargs):
    """This is the basic spectrum downsampling function, which loops
    over wavelength segments in a given spectrum with differnt
    resolutions, getting things ready for each call to ``smooth``
    """
    outspec = []
    if inres is None:
        inres = len(outwave) * [0]
    # loop over the output segments
    for owave, ores, ires in zip(outwave, outres, inres):
        wmin, wmax = owave.min(), owave.max()
        if velocity:
            sigma = 2.998e5 / ores  # in km/s
            smin = wmin - nsigma * wmin/ores
            smax = wmax + nsigma * wmax/ores
        else:
            sigma = ores  # in AA
            smin = wmin - nsigma * sigma
            smax = wmax + nsigma * sigma
        imin = np.argmin(np.abs(smin - wave))
        imax = np.argmin(np.abs(smax - wave))
        ospec = smooth(wave[imin:imax], spec[imin:imax], sigma, inres=ires,
                       velocity=velocity, outwave=owave, nsigma=nsigma,
                       **kwargs)
        outspec += [ospec]
    return outspec


def construct_outwave(resolution, wlo, whi, velocity=True,
                      absminwave=100, absmaxwave=1e8, lores=None,
                      **extras):
    """Given a spectral range of interest and a resolution in that
    range, construct wavelength vectors and resolution intervals that will
    cover this range at the desired reolution, but also ranges outside
    this at lower resolution (suitable for photometry, ionizing flux,
    and dust emission calculations)
    """

    if velocity:
        if lores is None:
            lores = 100  # R
    else:
        if lores is None:
            lores = 30  # AA
    resolution = [r for r in np.atleast_1d(resolution)]
    wlo = np.atleast_1d(wlo)
    whi = np.atleast_1d(whi)
    wave = [(l,h) for l,h in zip(wlo, whi)]
    wave = [(absminwave, wlo[0])] + wave + [(whi[-1], absmaxwave)]
    res = [lores] + resolution + [lores]

    out = []
    for (wmin, wmax), r in zip(wave, res):
        if velocity:
            dlnlam = 1.0/r/2  # critically sample the resolution
            out += [np.exp(np.arange(np.log(wmin), np.log(wmax), dlnlam))]
        else:
            dlam = r/2.0  # critically sample the resolution
            out += [np.arange(wmin, wmax, dlam)]
    return out, res


def read_and_downsample_onez(z, outwave, outres, velocity=True,
                             binout='ckc_new_z{0}.bin',
                             write_binary=False, **kwargs):
    """Read one of the CKC binary files, loop through the spectra in
    it, downsample them, optionally write to a new binary file, and
    return the spectra.
    """
    if write_binary:
        outfile = open(binout.format(z), 'wb')
    name = '{0}/fullres/fsps/ckc14/binary/ckc14_z{1}.spectra.bin'.format(ckc_dir, z)

    wave = np.loadtxt('{0}/fullres/fsps/ckc14/ckc14.lambda'.format(ckc_dir))
    zlegend, logg, logt = ckc_params()
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


def read_and_downsample_spectra(outwave, outres,
                                velocity=True, write_binary=False,
                                binout='ckc_new_z{0}.bin', **kwargs):
    """Loop through the metallicities of the CKC library, read the
    binary files, downsample them, and return an array of shape (nz,
    ng*nt, nw)
    """
    newspec = []
    zlegend, logg, logt = ckc_params()
    zlist = ['{0:06.4f}'.format(z) for z in zlegend]
    # names = glob.glob('{0}/bin/ckc14_z*.spectra.bin'.format(ckc_dir))
    for z in zlist:
        zspec = read_and_downsample_onez(z, outwave, outres, binout=binout,
                                         write_binary=write_binary, **kwargs)
        newspec.append(zspec)

    return np.array(newspec)


def read_binary_spec(filename, nw, nspec):
    """Read a binary file with name ``filename`` containing ``nspec``
    spectra each of length ``nw`` wavelength points and return an
    array of shape (nspec, nw)
    """
    count = 0
    spec = np.empty([nspec, nw])
    with open(filename, 'rb') as f:
        while count < nspec:
                count += 1
                for iw in range(nw):
                    byte = f.read(4)
                    spec[count-1, iw] = struct.unpack('f', byte)[0]
    return spec

def flathdf_to_binary(hname):
    """Convert a *flat* hdf5 spectral data file to the binary format
    appropriate for FSPS.  This also writes a .lambda file and a
    zlegend file.  A simple ```wc *.lambda``` will give the number of
    wavelength points to specify in sps_vars.f90
    """
    outroot = '.'.join(hname.split('.')[:-2])
    import h5py
    with h5py.File(hname, "r") as f:
        w = f['wavelengths'][:]
        p = f['parameters'][:]
        zlegend = np.unique(p['Z'])
        zlist = ['{0:06.4f}'.format(z) for z in zlegend]

        # Write the wavelength file
        wfile = open('{}.lambda'.format(outroot), 'w')
        for wave in w:
            wfile.write('{}\n'.format(wave))
        wfile.close()

        zfile = open('{}_zlegend.dat'.format(outroot), 'w')
        for i, z in enumerate(zlist):
            name = '{0}_z{1}.spectra.bin'.format(outroot, z)
            outfile = open(name, 'wb')
            thisz = p['Z'] == zlegend[i]
            for s in f['spectra'][thisz, :]:
                for flux in s:
                    outfile.write(struct.pack('f', flux))
            outfile.close()
            zfile.write('{}\n'.format(zlegend[i]))
        zfile.close()


def binary_to_hdf(hname):
    """Convert a set of binary files containing spectra into an hdf5
    file containing those same spectra.  The binary files are assumed
    to be in ``"{0}/bin/ckc14_z{1}.spectra.bin".format(ckc_dir, z) for
    z given by the zlegend.dat file.

    The output hdf5 file has datasets ``wavelengths``, ``logg``, and
    ``logt``, and then the group ``spectra`` which has as members
    the datasets ``z{}``, each of which is an array of shape (len(logg),
    len(logt), nw) for that metallicity"
    """
    import h5py
    wave = np.loadtxt('{0}/fullres/fsps/ckc14/ckc14.lambda'.format(ckc_dir))
    nw = len(wave)
    zlegend, logg, logt = ckc_params()
    nspec = len(logg) * len(logt)

    with h5py.File(hname, "w") as f:
        spgr = f.create_group('spectra')
        fw = f.create_dataset('wavelengths', data=wave)
        fg = f.create_dataset('logg', data=logg)
        ft = f.create_dataset('logt', data=logt)

        zlist = ['{0:06.4f}'.format(z) for z in zlegend]
        for z in zlist:
            name = '{0}/fullres/fsps/ckc14/binary/ckc14_z{1}.spectra.bin'.format(ckc_dir, z)
            spec = read_binary_spec(name, nw, nspec)
            spec = spec.reshape(len(logg), len(logt), nw)
            fl = spgr.create_dataset('z{0}'.format(z), data=spec)
            f.flush()


def ckc_params(expanded=False, zlegend=None, logg=None, logt=None,
                **extras):
    """Get parameters (Z, g, T) for the CKC/BaSeL library.

    :param expanded:
        If True, return a structured array of length (n_logg * n_logt
        * n_Z) with three fields, 'Z, 'logg', and 'logt'.  This means
        that each spectrum in the library has an entry in the
        structured array that gives the parameters of that spectrum.
    """
    if zlegend is None:
        zlegend = np.loadtxt('{0}/fullres/fsps/ckc14/zlegend.dat'.format(ckc_dir))
    if logg is None:
        logg = np.loadtxt('{0}/fullres/fsps/basel_logg.dat'.format(ckc_dir))
    if logt is None:
        logt = np.loadtxt('{0}/fullres/fsps/basel_logt.dat'.format(ckc_dir))
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


def wave_from_ssp():
    out = open('{0}/fullres/fsps/ckc14/ckc14.lambda'.format(ckc_dir), 'w')
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


