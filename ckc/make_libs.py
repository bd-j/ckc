import os
import ckc
import numpy as np
import h5py

def make_lib(R=1000, wmin=1e3, wmax=1e4, velocity=True,
             dirname='./', name='ckc14_new', **extras):
    """Make a new downsampled CKC library, with the desired resolution
    in the desired wavelength range.  This makes both binary files
    (one for each metallicity) and a wavelength file, suitable for use
    in FSPS.  It also makes an hdf5 file.

    This is deprecated in favor of make_lib_byz below
    """
    downsample = ckc.read_and_downsample_spectra
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    outwave, outres = ckc.construct_outwave(R, wmin, wmax,
                                            velocity=velocity)
    wave = np.concatenate(outwave)
    spectra = downsample(outwave, outres, velocity=velocity,
                         write_binary=True,
                         binout=dirname+'{0}_z{{0}}.spectra.bin'.format(name))

    with open(dirname+'{0}.lambda'.format(name), 'w') as f:
        for w in np.concatenate(outwave):
            f.write('{0:15.4f}\n'.format(w))
    params = ckc.spec_params(expanded=True)
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
    downsample = ckc.read_and_downsample_onez

    # set up output names, directories, and files
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    binout = dirname+'{0}_z{{0}}.spectra.bin'.format(name)
    wout = dirname+'{0}.lambda'.format(name)
    with h5py.File(dirname + '{0}.h5'.format(name), 'x') as f:
        spgr = f.create_group("spectra")

        # Get the output wavelength grid, write to hdf5 and a file
        outwave, outres = ckc.construct_outwave(R, wmin, wmax,
                                                velocity=velocity)
        wave = np.concatenate(outwave)
        dwave = f.create_dataset("wavelengths", data=wave)
        with open(wout, 'w') as wf:
            for w in np.concatenate(outwave):
                wf.write('{0:15.4f}\n'.format(w))

        # Parameters
        params = ckc.spec_params(expanded=True)
        dpar =  f.create_dataset("parameters", data=params)
        zlegend, logg, logt = ckc.spec_params()

        # Get the spectra for each z binary file
        zlist = ['{0:06.4f}'.format(z) for z in zlegend]
        for z in zlist:
            print('doing z={}'.format(z))
            zspec = downsample(z, outwave, outres, binout=binout,
                               write_binary=True, velocity=velocity)
            zd = spgr.create_dataset('z{0}'.format(z), data=zspec)
            f.flush()


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
        Complete path to the CKC14 fullspec file.
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
            exparams = ckc.spec_params(expanded=True, zlegend=zlegend,
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
             'dirname': ckc.ckc_dir+'lores/manga_R2000/',
             'name': 'ckc14_manga'
             }
    manga3 = {'R': 3145, 'wmin': 3500, 'wmax': 11000,
             'dirname': ckc.ckc_dir+'lores/manga_R3000/',
             'name': 'ckc14_manga'
             }
    deimos = {'R': 5000, 'wmin': 4000, 'wmax': 11000,
             'dirname': ckc.ckc_dir+'lores/deimos/',
             'name': 'ckc14_deimos',
             'h5out': 'ckc14_deimos.h5'
             }

    params = manga3
    make_lib_byz(**params)
