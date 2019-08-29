#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
from itertools import product

import numpy as np
import h5py

from prospect.sources import StarBasis

from .make_full_h5 import full_h5
from .make_sed import make_seds
from .sed_to_fsps import get_basel_params, get_binary_spec, interpolate_to_basel


__all__ = ["sed", "to_basel"]


def sed(feh, afe, segments, args):
    template = "{}/{}_feh{:+3.2f}_afe{:+2.1f}.{}.h5"
    specname = template.format(args.fulldir, args.ck_vers, feh, afe, "full")
    fluxname = template.format(args.fulldir, args.ck_vers, feh, afe, "flux")
    outname = template.format(args.seddir, args.ck_vers, feh, afe, args.sedname)

    # Read Files and make the sed file
    specfile = h5py.File(specname, "r")
    fluxfile = h5py.File(fluxname, "r")
    make_seds(specfile, fluxfile, fluxres=5e3, segments=segments,
              outname=outname, verbose=args.verbose,
              oversample=args.oversample)
    return outname


def to_basel(feh, afe, sedfile, args):
    # Filenames
    template = "{}/{}_feh{:+3.2f}_afe{:+2.1f}.{}.fsps.h5"
    outname = template.format(args.seddir, args.ck_vers, feh, afe, args.sedname)

    # Basel Params and valid z=0.0200 spectra
    basel_pars = get_basel_params()
    cwave, cspec, valid = get_binary_spec(len(basel_pars), zstr="0.0200",
                                          speclib='BaSeL3.1/basel')
    # My interpolator
    interpolator = StarBasis(sedfile, use_params=['logg', 'logt'], logify_Z=False,
                             n_neighbors=1, verbose=args.verbose,
                             rescale_libparams=True)

    # Do the interpolation
    bwave, bspec, inds = interpolate_to_basel([basel_pars, cwave, cspec], interpolator,
                                              valid=valid, renorm=False, plot=None,
                                              verbose=args.verbose)
    # Keep track of how interpolation was done
    false = np.zeros(len(basel_pars), dtype=bool)
    o, i, e = inds
    out, interp, extreme = false.copy(), false.copy(), false.copy()
    out[o] = True
    interp[i] = True
    extreme[e] = True
    exact = (valid & (~out) & (~interp) & (~extreme))

    if args.nowrite:
        return basel_pars, bwave, bspec, inds
    # Write the output
    with h5py.File(outname, "w") as f:
        f.create_dataset("parameters", data=basel_pars)
        f.create_dataset("spectra", data=bspec)
        f.create_dataset("wavelengths", data=interpolator.wavelengths)
        idat = f.create_group("interpolation_info")
        idat.create_dataset("interpolated", data=interp)
        idat.create_dataset("exact", data=exact)
        idat.create_dataset("nearest_tg", data=extreme)


if __name__ == "__main__":

    # These are the set of feh and afe from which we will choose based on zindex
    fehlist = [-2.0, -1.75, -1.5, -1.25, -1.0,
               -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
    afelist = [-0.2, 0.0, 0.4, 0.6]
    from .utils import get_ckc_parser
    # key arguments are:
    #  * --zindex
    #  * --oversample
    #  * --ck_vers
    #  * --fulldir
    #  * --seddir
    #  * --sedname
    parser = get_ckc_parser()
    args = parser.parse_args()

    # -- Mess with some args ---
    args.fulldir = args.fulldir.format(args.ck_vers)
    args.spec_type = "lores"

    # --- CHOOSE THE METALLICITY ---
    if args.zindex < 0:
        # for testing off odyssey
        feh, afe = 0.0, 0.0
    else:
        metlist = list(product(fehlist, afelist))
        feh, afe = metlist[args.zindex]
    print(feh, afe)

    # --- make the sed file ---
    # lambda_lo, lambda_hi, R_{out, fwhm}, use_fft
    segments = [(100., 910., 250., False),
                (910., 2500., 500., False), 
                (2500., 2.0e4, 500., True),
                (2.0e4, 1e8, 50., False)
                ]
    sedfile = sed(feh, afe, segments, args)

    # --- Interpolate the SED to BaSeL logt, logg grid and write to new h5 file ---
    if "sedfile" not in locals():
        template = "{}/{}_feh{:+3.2f}_afe{:+2.1f}.{}.h5"
        sedfile = template.format(args.seddir, args.ck_vers, feh, afe, args.sedname)
    args.nowrite = False
    out = to_basel(feh, afe, sedfile, args)
    if args.nowrite:
        basel_pars, bwave, bspec, inds = out
