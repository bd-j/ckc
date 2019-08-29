#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse


__all__ = ["ckms", "sigma_to_fwhm", "tiny_number", "param_order"
           "construct_outwave", "convert_resolution",
           "get_ckc_parser"]


# Useful constants
ckms = 2.998e5
sigma_to_fwhm = 2 * np.sqrt(2 * np.log(2))
tiny_number = 1e-33
param_order = ['t', 'g', 'feh', 'afe']

def construct_outwave(min_wave_smooth=0.0, max_wave_smooth=np.inf,
                      dispersion=1.0, oversample=2.0,
                      resolution=3e5, logarithmic=False,
                      **extras):
    """Given parameters describing the output spectrum, generate a wavelength
    grid that properly samples the resolution.

    Parameters
    ----------
    min_wave_smooth : float, optional (default: 0.)
        Minimum wavelength of the output wavelength vector

    max_wave_smooth : float (default: np.inf)
        Maximum wavelength of the output wavelength vector,
        same units as min_wave_smooth

    dispersion : float, optional (default: 1)
        Wavelength units per resolution element if `logarithmic=False`

    resolution : float, optional
        The value of the wavelength divided by the resolution element,
        if `logarithmic=True`

    oversample : float, optional, (default: 2)
        The number of pixels per resolution element.

    Returns
    ---------
    wave : ndarray
        The output wavelength vector.
    """
    if logarithmic:
        # critically sample the resolution
        dlnlam = 1.0 / resolution / oversample
        lnmin, lnmax = np.log(min_wave_smooth), np.log(max_wave_smooth)
        #print(lnmin, lnmax, dlnlam, resolution, oversample)
        out = np.exp(np.arange(lnmin, lnmax + dlnlam, dlnlam))
    else:
        out = np.arange(min_wave_smooth, max_wave_smooth,
                        dispersion / oversample)
    return out


def convert_resolution(R_fwhm, R_library=3e5):
    """Convert from standard 'R" values based on lambda/FWHM to the
    lambda/sigma values expected by smoothspec
    """
    R_sigma = sigma_to_fwhm / np.sqrt(1 / R_fwhm**2 - 1 / R_library**2)
    return R_sigma



def get_ckc_parser():
    """A general purpose argument parser for ckc methods and modules.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", type=bool, default=True,
                        help="chatter?")
    parser.add_argument("--np", type=int, default=6,
                        help="number of processors")

    # Metallicity choices
    parser.add_argument("--zindex", type=int, default=-99,
                        help="index of the metallicity mixture to use")
    parser.add_argument("--feh", type=float, default=0.0,
                        help=("The feh value to process."))
    parser.add_argument("--afe", type=float, default=0.0,
                        help=("The afe value to process."))

    # Number of pixels per sigma
    parser.add_argument("--oversample", type=float, default=2,
                        help="Factor by which to oversample the dispersion")

    # Filenames, library versions, and spectrum locations
    parser.add_argument("--ck_vers", type=str, default="c3k_v1.3",
                        help=("Name of directory that contains the "
                              "version of C3K spectra to use."))
    parser.add_argument("--basedir", type=str,
                        default='/n/conroyfs1/cconroy/kurucz/grids',
                        help=("Location of the directories containing "
                              "different C3K versions."))
    parser.add_argument("--fulldir", type=str,
                        default='/n/conroyfs1/bdjohnson/data/stars/{}/h5/',
                        help=("Location to store the HDF5 versions of .spec and .flux"))
    parser.add_argument("--seddir", type=str, default='./',
                        help=("Path to the directory where the sed files will be placed."))
    parser.add_argument("--sedname", type=str, default="sed",
                        help=("nickname for the SED file, e.g. sedR500"))

    # For `make_full_h5` if making HDF5 for the first time
    parser.add_argument("--spec_type", type=str, default='lores',
                        help=("Whether to create HDF5 for the full resolution "
                              "spectra ('hires') or for the low resolution, "
                              "larger wavelength range .flux files ('lores')"))

    return parser
