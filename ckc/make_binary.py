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

__all__ = ["prep_for_fsps"]


def prep_for_fsps(fehlist=[-0.75, -0.5, -0.25, 0.0, 0.25],
                  zsol=0.0134, afe=0.0, args=None, outdir=None, prefix=None):

    if outdir is not None:
        args.outdir = outdir
    if prefix is not None:
        args.prefix = prefix

    template = "{}/{}_feh{:+3.2f}_afe{:+2.1f}.{}.fsps.h5"
    binary_name = "{}/{}_z{}.spectra.bin"
    lambda_name = "{}/{}.lambda"
    zname = "{}/{}_zlegend.dat"

    zlegend = open(zname.format(args.outdir, args.prefix), "w")
    for j, feh in enumerate(fehlist):
        z = zsol * 10**feh
        zstr = "{:1.4f}".format(z)
        zlegend.write("{}\n".format(zstr))
        sedfile = template.format(args.seddir, args.ck_vers, feh, afe, args.sedname)
        outname = binary_name.format(args.outdir, args.prefix, zstr)
        sed_to_bin(sedfile, outname)
    zlegend.close()
    # Now make the wavelength file
    with h5py.File(sedfile, "r") as f:
        wave = np.array(f["wavelengths"])
    with open(lambda_name.format(args.outdir, args.prefix), "w") as wavefile:
        for w in wave:
            wavefile.write("{}\n".format(w))
