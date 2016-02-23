import os, sys, time, gc
import json
import numpy as np
import h5py
from ykc_data import sigma_to_fwhm
from bsfh.utils import smoothing

# Note that smoothspec expects resolution to be defined in terms of sigma, not FWHM
wfc3_g102 = {'name': 'wfc3_ir_g102',
             'resolution': 48.0 / sigma_to_fwhm, 'res_units': '\AA sigma',
             'dispersion': 24.5, 'disp_units': '\AA per pixel',
             'oversample': 4.,
             'fftsmooth': True, 'smoothtype': 'lambda',
             'min_wave_smooth': 0.5e4, 'max_wave_smooth':1.3e4}

wfc3_g141 = {'name': 'wfc3_ir_g141',
             'resolution': 93.0 / sigma_to_fwhm, 'res_units': '\AA sigma',
             'dispersion': 46.5, 'disp_units': '\AA per pixel',
             'oversample': 4.,
             'fftsmooth': True, 'smoothtype': 'lambda',
             'min_wave_smooth': 0.5e4, 'max_wave_smooth':2.0e4}

spherex = {'name': 'spherex',
             'resolution': 50.0 * sigma_to_fwhm, 'res_units': '\AA sigma',
             'logarithmic': True, 'oversample': 2.,
             'fftsmooth': True, 'smoothtype': 'R',
             'min_wave_smooth': 0.4e4, 'max_wave_smooth':2.5e4}

def construct_grism_outwave(min_wave_smooth=0.0, max_wave_smooth=np.inf,
                            dispersion=1.0, oversample=2.0,
                            resolution=3e5, logarithmic=False,
                            **extras):
    if logarithmic:
        dlnlam = 1.0/resolution/2/oversample  # critically sample the resolution
        lnmin, lnmax = np.log(min_wave_smooth), np.log(max_wave_smooth)
        #print(lnmin, lnmax, dlnlam, resolution, oversample)
        out = np.exp(np.arange(lnmin, lnmax + dlnlam, dlnlam))
    else:
        out = np.arange(min_wave_smooth, max_wave_smooth, dispersion / oversample)
    return out    


#def smooth_one(wave, spec, outwave, **conv_pars):
#    cp = conv_pars.copy()
#    res = cp.pop('resolution')
#    return smoothing.smoothspec(wave, spec, res,
#                                outwave=outwave, **cp)


def downsample_one_h5(fullres_hname, resolution=1.0, **conv_pars):
    """Read one full resolution h5 file, downsample every spectrum in
    that file, and treturn the result"""
    
    outwave = construct_grism_outwave(resolution=resolution, **conv_pars)
    #print(resolution, len(outwave), conv_pars['smoothtype'])
    with h5py.File(fullres_hname, 'r') as fullres:
        params = np.array(fullres['parameters'])
        whires = np.array(fullres['wavelengths'])
        flores = np.zeros([len(params), len(outwave)])
        for i, p in enumerate(params):
            fhires = fullres['spectra'][i, :]
            s = smoothing.smoothspec(whires, fhires, resolution,
                                     outwave=outwave, **conv_pars)
            flores[i, :] = s
    gc.collect()
    return outwave, flores, params


class function_wrapper(object):

    def __init__(self, function, function_kwargs):
        self.function = function
        self.kwargs = function_kwargs

    def __call__(self, args):
        return self.function(*args, **self.kwargs)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        grism = sys.argv[1]
        conv_pars = globals()[grism]
    else:
        conv_pars = wfc3_g102
    htemplate = '../h5/ykc_feh={:3.1f}.full.h5'
    zlist = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5]
    hnames = [[htemplate.format(z)] for z in zlist]

    downsample_with_pars = function_wrapper(downsample_one_h5, conv_pars)
    
    pool = None
    import multiprocessing
    nproc = min([8, len(hnames)])
    pool = multiprocessing.Pool(nproc)
    if pool is not None:
        M = pool.map
    else:
        M = map

    results = M(downsample_with_pars, hnames)
    wave = results[0][0]
    spectra = np.vstack([r[1] for r in results])
    params = np.concatenate([r[2] for r in results])

    outname = '../ykc_{}.h5'.format(conv_pars['name'])
    with h5py.File(outname, "w") as f:
        wave = f.create_dataset('wavelengths', data=wave)
        spectra = f.create_dataset('spectra', data=spectra)
        par = f.create_dataset('parameters', data=params)
        for k, v in list(conv_pars.items()):
            f.attrs[k] = json.dumps(v)

    try:
        pool.close()
    except:
        pass
    #start = time.time()
    #w, spec, pars = downsample_one_h5(hnames[0], **wfc3_g141)
    #dt = time.time() - start
    #print('resampling took {}s'.format(dt))
