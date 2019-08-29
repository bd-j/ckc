#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""script to combine individula SED h5 files (at single feh, afe) into a giant
h5 file at all metallicities.  This shouldn't actually be necessary or
desirable in most cases
"""

from itertools import product
import numpy as np
import h5py


def initialize_h5(inname, outname):
    with h5py.File(inname, "r") as f:
        wave = f["wavelengths"][:]
        dtype = f["parameters"].dtype
        nmod, nw = len(f["parameters"]), len(wave)

    out = h5py.File(outname, "w")
    spec = out.create_dataset('spectra', shape=(0, nw),
                              maxshape=(None, nw))
    pars = out.create_dataset('parameters', shape=(0,),
                              maxshape=(None,), dtype=dtype)
    wavelength = out.create_dataset('wavelengths', data=wave)
    out.flush()
    return out, spec, pars


if __name__ == "__main__":


    #fehlist = #[-2.0, -1.75, -1.5, -1.25, -1.0,
    fehlist = [-0.75, -0.5, -0.25, 0.0, 0.25]  #,0.5]
    afelist = [-0.2, 0.0, 0.4, 0.6]
    metlist = list(product(fehlist, afelist))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ck_vers", type=str, default="c3k_v1.3",
                        help=("Name of directory that contains the "
                              "version of C3K spectra to use."))
    parser.add_argument("--seddir", type=str, default='sed/',
                        help=("Path to the directory where the sed files will be placed."))
    parser.add_argument("--sedname", type=str, default="sed",
                        help=("nickname for the SED file, e.g. sedR500"))
    args = parser.parse_args()
    template = "{}/{}_feh{:+3.2f}_afe{:+2.1f}.{}.h5"

    # --- Initialize ---
    outname = "{}/{}.{}.h5".format(args.seddir, args.ck_vers, args.sedname)
    feh, afe = metlist[0]
    onesed = template.format(args.seddir, args.ck_vers, feh, afe, args.sedname)
    out, spec, pars = initialize_h5(onesed, outname)

    for i, (feh, afe) in enumerate(metlist):
        name = template.format(args.seddir, args.ck_vers, feh, afe, args.sedname)
        nmod = len(pars)
        with h5py.File(name, "r") as dat:
            # expand output arrays
            nnew = len(dat["parameters"])
            spec.resize(nmod + nnew, axis=0)
            pars.resize(nmod + nnew, axis=0)
            spec[nmod:nmod+nnew, :] = dat["spectra"][:]
            pars[nmod:nmod+nnew] = dat["parameters"][:]
        out.flush()

    out.close()
