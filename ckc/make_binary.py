#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Method and example code for converting individual metallicity (feh/afe combo) .fsps.h5 
HDF5 files of SEDs to FSPS compatible binary format with ancillary info files.

This assumes the SEDs have already been interpolated to the BaSeL stellar
parameter grid, and that the spectra in the HDF5 are ordered with logt changing
fastest and then logg.
"""

import glob
import numpy as np
import matplotlib.pyplot as pl

import h5py

from .utils import sed_to_bin

if __name__ == "__main__":

    # These are the set of feh and afe from which we will choose based on zindex
    #fehlist = [-2.0, -1.75, -1.5, -1.25, -1.0,
    #           -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
    fehlist = [-0.75, -0.5, -0.25, 0.0, 0.25]
    afelist = [-0.2, 0.0, 0.2, 0.4, 0.6]


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--zsol", type=float, default=0.0134,
                        help=("Definition of solar metallicity"))
    parser.add_argument("--verbose", type=bool, default=True,
                        help="chatter?")
    parser.add_argument("--ck_vers", type=str, default="c3k_v1.3",
                        help=("Name of directory that contains the "
                              "version of C3K spectra to use."))
    parser.add_argument("--outdir", type=str, default='./lores',
                        help=("Location to store the HDF5 versions of .spec and .flux"))
    parser.add_argument("--prefix", type=str, default='ckc14_a{}',
                        help=("prefix for the binary and wavelength files"))
    parser.add_argument("--seddir", type=str, default='lores/sed_r500',
                        help=("Path to the directory where the sed files will be placed."))
    parser.add_argument("--sedname", type=str, default="sed_r500",
                        help=("nickname for the SED file, e.g. sedR500"))

    args = parser.parse_args()
    
    template = "{}/{}_feh{:+3.2f}_afe{:+2.1f}.{}.fsps.h5"
    binary_name = "{}/{}_z{}.spectra.bin"
    lambda_name = "{}/{}.lambda"
    aname = "{}/afelegend.dat"
    zname = "{}/{}_zlegend.dat"

    alegend = open(aname.format(args.outdir), "w")
    for i, afe in enumerate(afelist):
        prefix = args.prefix.format(i+1)
        alegend.write("{} {}\n".format(i+1, afe))
        zlegend = open(zname.format(args.outdir, prefix), "w")
        for j, feh in enumerate(fehlist):
            z = args.zsol * 10**feh
            zstr = "{:1.4f}".format(z)
            zlegend.write("{}\n".format(zstr))
            sedfile = template.format(args.seddir, args.ck_vers, feh, afe, args.sedname)
            outname = binary_name.format(args.outdir, prefix, zstr)
            sed_to_bin(sedfile, outname)
        zlegend.close()
        # Now make the wavelength file
        with h5py.File(sedfile, "r") as f:
            wave = np.array(f["wavelengths"])
        with open(lambda_name.format(args.outdir, prefix), "w") as wavefile:
            for w in wave:
                wavefile.write("{}\n".format(w))

    alegend.close()
