import numpy as np
import struct

def find_segments(wave, restol=0.1):
    """
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
    """
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

def smooth(wave, spec, sigma, smooth_velocity=True, **kwargs):
    if smooth_velocity:
        return smooth_vel(wave, spec, sigma, **kwargs)
    else:
        return smooth_wave(wave, spec, sigma, **kwargs)
       
def construct_outwave(wave, res, logres=False):
    out = []
    for (wmin, wmax), r in zip(wave, res):
        if logres:
            dlnlam = 1.0/r/2 # critically sample the resolution
            out += [np.exp(np.arange(np.log(wmin), np.log(wmax), dlnlam))]
        else:
            dlam = r/2.0 #critically sample the resolution
            out += [np.arange(wmin, wmax, dlam)]
    return out
    
if __name__=="__main__":
    
    
    wave = np.loadtxt('ckc14.lambda')
    zlegend = np.loadtxt('zlegend.dat')
    logg = np.loadtxt('../BaSeL3.1/basel_logg.dat')
    logt = np.loadtxt('../BaSeL3.1/basel_logt.dat')
    nspec = len(logg) * len(logt) #len(zlegend)
    
    segments = find_segments(wave)
    nseg = len(segments)
    vsig = [2.998e5/s[2] * 2 for s in segments]

    wlim = [wave[ list(s[0:2])] for s in segments[:-1]]
    wlim = [(2000, 3700), (3700, 8000), (8000, 2e4)]
    outres = [20.0, 1.50, 20.0] #AA
    outwave = construct_outwave(wlim, outres, logres = False)
    outname = 'ckc14_hecto.lambda'
    with open(outname, 'w') as f:
        alloutwave = np.concatenate(outwave)
        for w in alloutwave:
            f.write('{0:15.4f}\n'.format(w))

            
    nw = len(wave)
    spec = np.empty(nw)

    zlist = ['{0:06.4f}'.format(z) for z in zlegend]
    for z in zlist:
               
        count = 0
        name = 'ckc14_z{0}.spectra.bin'.format(z)
        outname = 'ckc14_hecto_z{0}.spectra.bin'.format(z)
        outfile = open(outname, 'wb')
        with open(name, 'rb') as f:
            while count < nspec:
                count += 1
                for iw in range(nw):
                    byte = f.read(4)
                    spec[iw] = struct.unpack('f', byte)[0]
                sigma = [2.998e5/s[2] * 2 for s in segments]
                outspec = []
                for owave, smin in zip(outwave, outres):
                    owmin, owmax = owave.min(), owave.max()
                    imin = np.argmin(np.abs(owmin - 5*smin - wave))
                    imax = np.argmin(np.abs(owmax + 5*smin - wave))
                    ospec = smooth(wave[imin:imax], spec[imin:imax], smin,
                                   smooth_velocity=False, outwave=owave)
                    outspec += [ospec]
                    for flux in ospec:
                        outfile.write(struct.pack('f', flux))
        outfile.close()
                                  

