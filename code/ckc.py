import numpy as np
import struct, glob

ckc_dir = '/Users/bjohnson/Codes/SPS/CKC/'


def resample_ckc(R=3000, wmin=3500, wmax=10000, velocity=True,
                 outname='test.h5'):
        """
        """
        import h5py

        outwave, outres = construct_outwave(R, wmin, wmax, velocity)
        wave = np.concatenate(outwave)
        params = spec_params(expanded=True)
        spectra = read_and_downsample_spectra(outwave, outres, velocity=velocity):
        with h5py.File(outname, 'r') as f:
                dspec = f.create_dataset("spectra", data=spectra)
                dwave = f.create_dataset("wavelengths", data=wave)
                dpar =  f.create_dataset("parameters", data=params)


def construct_outwave(resolution, wlo, whi, velocity):
    """Given a spectral range of interest and a resolution in that
    range, construct wavelength vectors and resolution intervals that will
    cover this range at the desired reolution, but also ranges outside
    this at lower resolution (suitable for photometry calculations)
    """
    if velocity:
        lores = 100 #R
    else:
        lores = 30  # AA
    wave = [(2000, wlo), (wlo, whi), (wlo, 2e4)]
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


def read_and_downsample_spectra(outwave, outres,
                                velocity=True,
                                write_binary=False,
                                binout='ckc_new_z{0}.bin'):

    wave = np.loadtxt('{0}/ckc14.lambda'.format(ckc_dir))
    nw = len(wave)
    zlegend, logg, logt = spec_params()
    nspec = len(logg) * len(logt)

    spec = np.empty(nw)
    newspec = []
    zlist = ['{0:06.4f}'.format(z) for z in zlegend]
    # names = glob.glob('{0}/bin/ckc14_z*.spectra.bin'.format(ckc_dir))
    for z in zlist:
        count = 0
        name = '{0}/bin/ckc14_z{1}.spectra.bin'.format(ckc_dir, z)
        if write_binary:
            outfile = open(binout.format(z), 'wb')
        with open(name, 'rb') as f:
            while count < nspec:
                count += 1
                for iw in range(nw):
                    byte = f.read(4)
                    spec[iw] = struct.unpack('f', byte)[0]
                # sigma = [2.998e5/s[2] * 2 for s in segments]
                outspec = []
                # loop over the output segments
                for owave, ores in zip(outwave, outres):
                    wmin, wmax = owave.min(), owave.max()
                    if velocity:
                        sigma = 2.998e5 / ores #in km/s
                        smin = wmin - 5 * wmin/ores
                        smax = wmax + 5 * wmax/ores
                    else:
                        sigma = ores  # in AA
                        smin = wmin - 5 * sigma
                        smax = wmax + 5 * sigma
                    imin = np.argmin(np.abs(smin - wave))
                    imax = np.argmin(np.abs(smax - wave))
                    ospec = smooth(wave[imin:imax], spec[imin:imax], sigma,
                                   velocity=velocity, outwave=owave)
                    outspec += [ospec]
                    if write_binary:
                        for flux in ospec:
                            outfile.write(struct.pack('f', flux))

                newspec.append([np.concatenate(outspec)])

    return np.array(newspec)


def spec_params(expanded=False):
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
                    pars[i,:] = np.array([z, g, t])
                    i += 1
        return pars

def binary_to_hdf():
    pass

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
    for i in range(len(lims) -1):
        segments.append((lims[i], lims[i+1], res[lims[i]:lims[i+1]].mean()))
    return segments


def smooth_vel(wave, spec, sigma, outwave=None, inres=0):
    """Smooth a spectrum in velocity space
    :param sigma:
        desired velocity resolution (km/s)
    """
    sigma_eff = np.sqrt(sigma**2-inres**2)/2.998e5
    if outwave is None:
        outwave = wave
    if sigma <= 0.0:
        return np.interp(wave, outwave, flux)
    
    lnwave = np.log(wave)
    flux = np.zeros(len(outwave))
    norm = 1/np.sqrt(2 * np.pi)/sigma
    
    for i, w in enumerate(outwave):
        x = np.log(w) - lnwave
        f = np.exp( -0.5*(x/sigma_eff)**2 )
        flux[i] = np.trapz( f * spec, x) / np.trapz(f, x)
    return flux


def smooth_wave(wave, spec, sigma, outwave=None,
                inres=0, in_vel=False, **extras):
    """
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
        sigma_eff = np.sqrt(sigma**2- inres**2)

    flux = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        #if in_vel:
        #    sigma_eff = np.sqrt(sigma**2 - (w/inres)**2)
        x = (wave-w)/sigma_eff
        f = np.exp( -0.5*(x)**2 )
        flux[i] = np.trapz( f * spec, wave) / np.trapz(f, wave)
    return flux

def smooth(wave, spec, sigma, velocity=True, **kwargs):
    if velocity:
        return smooth_vel(wave, spec, sigma, **kwargs)
    else:
        return smooth_wave(wave, spec, sigma, **kwargs)

