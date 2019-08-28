import sys, time, os
import json
import numpy as np


import h5py
#from prospect.utils import smoothing

#from libparams import sigma_to_fwhm
#from downsample_h5 import construct_outwave


if __name__ == "__main__":

    fn = "/Users/bjohnson/Projects/ckc/ckc/spectra/lores/c3k_v1.3_R5K.201703.h5"
    outname = "c3k_v1.3_R5K.Fstars.h5"
    
    with h5py.File(fn, "r") as lib:

        pars = lib["parameters"]
        sel = ((pars["logt"] <= 4.0) & (pars["logt"] >= np.log10(5500)) & (pars["logg"] <  5) &
               (pars["feh"] >= -1.0) & (pars["feh"] <= 0.25) & (pars["afe"] == 0.0)
               )

        with h5py.File(outname, "w") as out:
            w = out.create_dataset("wavelengths", data=lib["wavelengths"][:])
            p = out.create_dataset("parameters", data=pars[sel])
            s = out.create_dataset("spectra", data=lib["spectra"][sel, :])

            out.attrs["R"] = 5000.0
        
