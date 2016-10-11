# Make libraries from the FSPS resolution versions of CKC14.  Note these
# include extrapolated/interpolated spectra, and have already been downsampled
# (with different R in different wavelength regims)

import os
import utils
import numpy as np
import h5py

__all__ = ["make_lib", "make_lib_byz", "make_lib_flatfull",
           "flatten_h5", "flatten_fullspec"]

def make_lib(R=1000, wmin=1e3, wmax=1e4, velocity=True,
             dirname='./', name='ckc14_new', **extras):
    """Make a new downsampled CKC library, with the desired resolution
    in the desired wavelength range.  This makes both binary files
    (one for each metallicity) and a wavelength file, suitable for use
    in FSPS.  It also makes an hdf5 file.

    This is deprecated in favor of make_lib_byz below
    """
    downsample = utils.read_and_downsample_spectra
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    outwave, outres = utils.construct_outwave(R, wmin, wmax,
                                            velocity=velocity)
    wave = np.concatenate(outwave)
    spectra = downsample(outwave, outres, velocity=velocity,
                         write_binary=True,
                         binout=dirname+'{0}_z{{0}}.spectra.bin'.format(name))

    with open(dirname+'{0}.lambda'.format(name), 'w') as f:
        for w in np.concatenate(outwave):
            f.write('{0:15.4f}\n'.format(w))
    params = utils.spec_params(expanded=True)
    with h5py.File(dirname + '{0}.h5'.format(name), 'x') as f:
        dspec = f.create_dataset("spectra", data=spectra)
        dwave = f.create_dataset("wavelengths", data=wave)
        dpar =  f.create_dataset("parameters", data=params)
            

def make_lib_byz(R=1000, wmin=1e3, wmax=1e4, velocity=True,
             dirname='./', name='ckc14_new', **extras):

    """Make a new downsampled CKC library, with the desired resolution
    in the desired wavelength range.  This makes both binary files
    (one for each metallicity) and a wavelength file, suitable for use
    in FSPS.  It also makes an hdf5 file with the downsampled spectra.
    """
    downsample = utils.read_and_downsample_onez

    # set up output names, directories, and files
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    binout = dirname+'{0}_z{{0}}.spectra.bin'.format(name)
    wout = dirname+'{0}.lambda'.format(name)
    with h5py.File(dirname + '{0}.h5'.format(name), 'x') as f:
        spgr = f.create_group("spectra")

        # Get the output wavelength grid, write to hdf5 and a file
        outwave, outres = utils.construct_outwave(R, wmin, wmax,
                                                velocity=velocity)
        wave = np.concatenate(outwave)
        dwave = f.create_dataset("wavelengths", data=wave)
        with open(wout, 'w') as wf:
            for w in np.concatenate(outwave):
                wf.write('{0:15.4f}\n'.format(w))

        # Parameters
        params = utils.spec_params(expanded=True)
        dpar =  f.create_dataset("parameters", data=params)
        zlegend, logg, logt = utils.spec_params()

        # Get the spectra for each z binary file
        zlist = ['{0:06.4f}'.format(z) for z in zlegend]
        for z in zlist:
            print('doing z={}'.format(z))
            zspec = downsample(z, outwave, outres, binout=binout,
                               write_binary=True, velocity=velocity)
            zd = spgr.create_dataset('z{0}'.format(z), data=zspec)
            f.flush()

def make_lib_flatfull(R=1000, wmin=1e3, wmax=1e4, velocity=True,
                      h5name='data/h5/ckc14_fsps.flat.h5',
                      outfile='ckc14_new.flat.h5', **extras):
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
    from utils import downsample_onespec as downsample
    # Get the output wavelength grid as segments
    outwave, outres = utils.construct_outwave(R, wmin, wmax,
                                            velocity=velocity)
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
                s = np.array(full["spectra"][i, :])
                # monitor progress
                if np.mod(i , np.int(nspec/10)) == 0:
                    print("doing {0} of {1} spectra".format(i, nspec))
                # don't convolve empty spectra
                if s.max() < 1e-32:
                    continue
                # Actually do the convolution
                lores = downsample(fwave, s, outwave, outres,
                                   velocity=velocity)
                news[i, :] = np.concatenate(lores)
                # flush to disk so you can read the file and monitor
                # progress in another python instance, and if
                # something dies you don't totally lose the data
                if np.mod(i , np.int(nspec/10)) == 0:
                    newf.flush()


def flatten_h5(h5file, outfile=None):
    """Change the ``spectra`` group in the output from make_lib_byz to
    be a single dataset, with shape (nz*ng*nt, nwave).  This way the
    spectra match the ``parameters`` dataset line by line. This method
    creates a new file with the extension ``.flat.h5``
    """
    if outfile is None:
        outfile = h5file.replace('.h5','.flat.h5')
    with h5py.File(h5file, "r") as f:
        with h5py.File(outfile, "w") as newf:
            f.copy("wavelengths", newf)
            f.copy("parameters", newf)
            newspec = []
            zs = np.sort(f['spectra'].keys())
            for z in zs:
                newspec.append(np.squeeze(f['spectra'][z]))
            newf.create_dataset('spectra', data=np.vstack(newspec))


def flatten_fullspec(h5fullspec, outfile=None):
    """
    :param h5fullspec:
        Complete path to the CKC14 FSPS resolution h5 file.
    :param outfile: (optional)
        Complete path to the output file.  If not given the output
        will be h5fullspec with '.flat.' inserted
    """
    if outfile is None:
        outfile = h5fullspec.replace('.h5','.flat.h5')
    with h5py.File(h5fullspec, "r") as full:
        with h5py.File(outfile, "w") as newf:
            ng, nt = len(full['logg']), len(full['logt'])
            nw = len(full['wavelengths'])
            newspec = []
            zs = np.sort(full['spectra'].keys())
            zlegend = [np.float(z[1:]) for z in zs]
            # Get the full parameter list as a flat structured array,
            # and throw it in the h5file
            exparams = utils.spec_params(expanded=True, zlegend=zlegend,
                                       logt=full['logt'][:], logg=full['logg'][:])
            p = newf.create_dataset("parameters", data=exparams)
            # copy the wavelength array
            full.copy("wavelengths", newf)
            for z in zs:
                n = np.squeeze(full['spectra'][z]).reshape(ng*nt, nw)
                newspec.append(n)
            newf.create_dataset('spectra', data=np.vstack(newspec))

if __name__ == "__main__":


    manga = {'R': 2000, 'wmin': 3500, 'wmax': 11000,
             'dirname': utils.ckc_dir+'../lores/manga_R2000/',
             'name': 'ckc14_manga'
             }
    manga3 = {'R': 3145, 'wmin': 3500, 'wmax': 11000,
             'dirname': utils.ckc_dir+'../lores/manga_R3000/',
             'name': 'ckc14_manga'
             }
    deimos = {'R': 5000, 'wmin': 4000, 'wmax': 11000,
             'dirname': utils.ckc_dir+'../lores/deimos/',
             'name': 'ckc14_deimos',
             'h5out': 'ckc14_deimos.h5'
             }

    # account for intrinsic resolution of the CKC grid (10000) to get
    # a desired resolution of 2000
    r = 1/np.sqrt((1./2000)**2 - (1./4000)**2)
    irtf = {'R': r, 'wmin': 3000, 'wmax': 20000,
             'h5name': utils.ckc_dir+'data/h5/ckc14_fsps.flat.h5',
             'outfile': utils.ckc_dir+'../lores/ckc14_irtf.flat.h5'
             }
    make_lib_flatfull(**irtf)

    #params = manga3
    #make_lib_byz(**params)
