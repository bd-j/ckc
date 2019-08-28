import sys, time
import numpy as np
import h5py
from ckc import construct_outwave, smooth

def downsample_wave(wave, spec, outwave, outres, inres=None,
                    velocity=False, nsigma=10, **kwargs):
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


def make_lib_flatfull(R=[1000], wmin=[1e3], wmax=[1e4],
                      h5name='../h5/ckc14_fullres.flat.h5',
                      outfile='ckc14_new.flat.h5', verbose=False, **extras):
    """Make a new downsampled CKC library, with the desired resolution
    in the desired wavelength range.  This makes an hdf5 file with the
    downsampled spectra.

    :param R:
        Desired resolution in the interval (wmin, wmax) *not including
        the native CKC resolution*, in terms of
        lambda/sigma_lambda. Ouside this interval the resolution will
        be 100.

    :param h5name:
        Full path to the *flat* version of the HDF file containing the
        full CKC grid.

    :param outfile:
        Full path to the output h5 filename.  Note that this will be a
        *flat* spectral file, suitable for use with StarBasis()
    """

    h5fullflat = h5name
    # Get the output wavelength grid as segments
    #print(R, wmin, wmax, extras)
    outwave, outres = construct_outwave(R, wmin, wmax, **extras)
    #sys.exit()
    outwave = outwave[1:-1]
    outres = outres[1:-1]
    wave = np.concatenate(outwave)
    with h5py.File(h5fullflat, "r") as full:
        # Full wavelength vector and number of spectra
        fwave = np.array(full["wavelengths"])
        nspec = full["parameters"].shape[0]
        with h5py.File(outfile, "w") as newf:
            # store the output wavelength vector
            newf.create_dataset("wavelengths", data=wave)
            # copy the spectral parameters over
            full.copy("parameters", newf)
            # create an array to hold the new spectra.
            news = newf.create_dataset("spectra", data=np.ones([nspec, len(wave)]) * 1e-33)
            # loop over the old spectra
            for i in xrange(nspec):
                ts = time.time()
                s = np.array(full["spectra"][i, :])
                # monitor progress
                if np.mod(i , np.int(nspec/10)) == 0:
                    print("doing {0} of {1} spectra".format(i, nspec))
                # don't convolve empty spectra
                if s.max() < 1e-32:
                    continue
                # Actually do the convolution
                lores = downsample_wave(fwave, s, outwave, outres, **extras)
                news[i, :] = np.concatenate(lores)
                if verbose:
                    print("done one in {}s".format(time.time() - ts))
                # flush to disk so you can read the file and monitor
                # progress in another python instance, and if
                # something dies you don't totally lose the data
                if np.mod(i , np.int(nspec/10)) == 0:
                    newf.flush()

if __name__ == "__main__":

    # example script.
    
    # The CKC resolution intervals are
    # R = 500,   100   < lambda < 1500
    # R = 10000, 1500  < lambda < 11000
    # R = 2000,  11000 < lambda < 30000
    # R = 50,   30000 < lambda < 1000000
    # where R is in terms of FWHM
    
    # Here R is defined in terms of sigma_lambda
    #set up segments to get smooth resolution transitions
    wmin = [100, 1500, 3650, 9900, 11000]
    wmax = [1500, 3650, 9900, 11000, 30000]
    r = [30, 30, 1.0, 30, 100]
    inres = [500, 10000, 10000, 10000, 2000]
        
    deimos = {'R': r, 'wmin': wmin, 'wmax': wmax,
              'inres':inres, 'in_vel': True, 'velocity': False,
              'absmaxwave': 3e4, 'lores': 100,
              'h5name': '../h5/ckc14_fullres.flat.h5',
              'outfile': '../lores/deimos_R1/ckc14_deimos_R1AA.flat.h5',             
             }

    make_lib_flatfull(verbose=True, **deimos)
    
    sys.exit()
    
    outwave, outres = construct_outwave(deimos['R'], deimos['wmin'], deimos['wmax'],
                                        **deimos)
    outwave = outwave[1:-1]
    outres = outres[1:-1]
    wave = np.concatenate(outwave)
    
    full = h5py.File(deimos['h5name'], "r")
    fwave = np.array(full["wavelengths"])
    smax = full['spectra'][:,:].max(axis=-1)
    ind = np.where(smax > 1e-32)[0][100]
    s = np.array(full["spectra"][ind, :])
    
    # Actually do the convolution
    lores = downsample_wave(fwave, s, outwave, outres, **deimos)
    plot(fwave, s)
    plot(wave, np.concatenate(lores))
