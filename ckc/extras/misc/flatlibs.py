import sys, time
import numpy as np
import h5py
from ckc.utils import downsample_onespec as downsample
from ckc.utils import construct_outwave, ckc_dir


def make_lib_flatfull(R=[1000], wmin=[1e3], wmax=[1e4],
                      h5name=ckc_dir+'/h5/ckc14_fullres.flat.h5',
                      outfile='ckc14_new.flat.h5', verbose=False,
                      test=False, **extras):
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
    outwave, outres = construct_outwave(R, wmin, wmax, **extras)
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
                lores = downsample(fwave, s, outwave, outres, **extras)
                news[i, :] = np.concatenate(lores)
                if verbose:
                    print("done one in {}s".format(time.time() - ts))
                # flush to disk so you can read the file and monitor
                # progress in another python instance, and if
                # something dies you don't totally lose the data
                if np.mod(i , np.int(nspec/10)) == 0:
                    newf.flush()
                # only do one for testing purposes
                if test:
                    break
                
if __name__ == "__main__":

    import ckc.libparams
    specparams = ckc.libparams.__dict__[sys.argv[1]]
    print(sys.argv[1])
    make_lib_flatfull(verbose=True, **specparams)
