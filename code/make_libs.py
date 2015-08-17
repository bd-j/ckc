import os
import ckc
import numpy as np
import h5py


def make_lib(R=1000, wmin=1e3, wmax=1e4, velocity=True,
             dirname='./', name='ckc14_new', **extras):

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


def flatten_h5(h5file):
    with h5py.File(h5file, "r") as f:
        with h5py.File(h5file.replace('.h5','.flat.h5'), "w") as newf:
            f.copy("wavelengths", newf)
            f.copy("parameters", newf)
            newspec = []
            zs = np.sort(f['spectra'].keys())
            for z in zs:
                newspec.append(np.squeeze(f['spectra'][z]))
            newf.create_dataset('spectra', data=np.vstack(newspec))


def make_deimos():
    R, wmin, wmax = 5000, 4000, 11000
    ckc.resample_ckc(R, wmin, wmax, velocity=False,
                     outname='h5/ckc14_deimos.h5')


if __name__ == "__main__":


    manga = {'R': 2000, 'wmin': 3500, 'wmax': 11000,
             'dirname': ckc.ckc_dir+'lores/manga_R2000/',
             'name': 'ckc14_manga'
             }
    deimos = {'R': 5000, 'wmin': 4000, 'wmax': 11000,
             'dirname': ckc.ckc_dir+'lores/deimos/',
             'name': 'ckc14_deimos',
             'h5out': 'ckc14_deimos.h5'
             }

    params = deimos
    make_lib_byz(**params)
