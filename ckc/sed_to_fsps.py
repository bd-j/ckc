# Module for converting single metallicity SED files in HDF format to the
# parameters and binary format expected by fsps
import sys, os, struct
from itertools import product
import numpy as np

import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
import h5py

from prospect.sources import StarBasis, BigStarBasis


def dict_struct(strct):
    """Convert from a structured array to a dictionary.  This shouldn't really
    be necessary.
    """
    return dict([(n, strct[n]) for n in strct.dtype.names])


def get_basel_params():
    """Get the BaSeL grid parameters as a 1-d list of parameter tuples.  The
    binary files are written in the order logg, logt, wave with wave changing
    fastest and logg the slowest.
    """
    fsps_dir = os.path.join(os.environ["SPS_HOME"], "SPECTRA", "BaSeL3.1")
    logg = np.genfromtxt("{}/basel_logg.dat".format(fsps_dir))
    logt = np.genfromtxt("{}/basel_logt.dat".format(fsps_dir))
    logt = np.log10(np.round(10**logt))
    ngrid = len(logg) * len(logt)
    dt = np.dtype([('logg', np.float), ('logt', np.float)])
    basel_params = np.array(list(product(logg, logt)), dtype=dt)
    return basel_params


def get_binary_spec(ngrid, zstr="0.0200", speclib="BaSeL3.1/basel"):
    """
    :param zstr: for basel "0.0002", "0.0006", "0.0020", "0.0063", "0.0200", "0.0632"
    """
    from binary_utils import read_binary_spec
    specname = "{}/SPECTRA/{}".format(os.environ["SPS_HOME"], speclib)
    wave = np.genfromtxt("{}.lambda".format(specname))
    try:
        ss = read_binary_spec("{}_wlbc_z{}.spectra.bin".format(specname, zstr), len(wave), ngrid)
    except(IOError):
        ss = read_binary_spec("{}_z{}.spectra.bin".format(specname, zstr), len(wave), ngrid)
    #logg = np.genfromtxt("{}_logg.dat".format(specname))
    #logt = np.genfromtxt("{}_logt.dat".format(specname))
    #spec = ss.reshape(len(logg), len(logt), len(wave))
    valid = ss.max(axis=1) > 1e-32
    return wave, ss, valid


def disp_param(par):
    return ' '.join([str(p) for p in par])


def sed_to_bin(sedfile, outname):

    with open(outname, "wb") as outfile:
            for flux in bspec:
                outfile.write(struct.pack('f', flux))


def renorm(spec, normed_spec, wlo=5e3, whi=2e4):

    w, f = spec
    wn, fn = normed_spec
    f = np.squeeze(f)
    fn = np.squeeze(fn)
    g = (w > wlo) & (w < whi)
    l = np.trapz(f[g], w[g])
    g = (wn > wlo) & (wn < whi)
    ln = np.trapz(fn[g], wn[g])
    return ln / l, f * ln / l


def rectify_sed(sedfile):
    """Do all the crazy magic to get models that are better for interpolating
    to BaSeL params.  Necessary?
    """
    with h5py.File(sedfile, "r") as sed:
        params = np.array(sed["parameters"])
        spec = np.array(sed["spectra"])
        wave = np.array(sed["wavelengths"])
        
    # Adjust temperatures
    for t in [37000., 42000., 47000.]:
        this = 10**params["logt"] == t
        params[this]["logt"] = np.log10(t)

    # Copy logg
    sel = params["logg"] == -0.50
    params[sel]["logg"] = -0.51
    newpars = np.zeros(sel.sum, dtype=params.dtype)
    

def interpolate_to_basel(charlie, interpolator, valid=True, plot=None, renorm=True):

    bwave = interpolator.wavelengths
    if plot is not None:
        out_pdf = PdfPages('out'+plot)
        in_pdf = PdfPages('in'+plot)
        ex_pdf = PdfPages('ex'+plot)
    basel_pars, cwave, cspec = charlie
    allspec = []
    outside, inside, extreme = [], [], []

    for i, (p, v) in enumerate(zip(basel_pars, valid)):
        title = "target: {logt:4.3f}, {logg:3.2f}".format(**dict_struct(p))
        inds, wghts = interpolator.weights(**dict_struct(p))
        ex_g = extremeg(interpolator, p)
        print(i, p, v, ex_g, len(inds))
        if ex_g and v:
            inds, wghts = nearest_tg(interpolator, p)

        # valid *and* (in-hull and non-extreme g)
        # could use `or` except for normalization - need a valid sample.
        if (v and (len(inds) > 1)):
            _, bspec, _ = interpolator.get_star_spectrum(**dict_struct(p))
            if renorm:
                norm, bspec = renorm([bwave, bspec], [cwave, cspec[i,:]])
            if (wghts.max() < 1.0) & (plot is not None):
                fig, ax = plot_interp(cwave, cspec[i,:], bspec, inds, wghts, interpolator)
                ax.set_title(title + ", norm={:4.3f}".format(norm))
                ax.legend()
                inside.append(i)
                in_pdf.savefig(fig)
                pl.close(fig)

        # valid *and* (out-hull or extremeg)
        elif (v and (len(inds) == 1)):
            bspec = interpolator._spectra[inds, :]
            if renorm:
                norm, bspec = renorm([bwave, bspec], [cwave, cspec[i,:]])
            if plot is not None:
                fig, ax = plot_interp(cwave, cspec[i,:], bspec, inds, wghts, interpolator)
                ax.set_title(title + ", norm={:4.3f}".format(norm))
                ax.legend()
                if ex_g:
                    extreme.append(i)
                    ex_pdf.savefig(fig)
                else:
                    outside.append(i)
                    out_pdf.savefig(fig)
                pl.close(fig)
        # not valid
        else:
            bspec = np.zeros_like(interpolator.wavelengths) + 1e-33
        allspec.append(bspec)

    if plot is not None:
        out_pdf.close()
        in_pdf.close()
        ex_pdf.close()

    return interpolator.wavelengths, np.array(allspec), [outside, inside, extreme]


def comp_text(inds, wghts, interpolator):
    #if type(target) is not dict:
    #    target = dict_struct(target)
    #inds, wghts = interpolator.weights(**target)
    #hdr = 'wght   ' + ' '.join(interpolator.stellar_pars)
    #txt = "target: {logt:4.3f} {logg:3.2f}".format(**target)
    txt = ""
    for i, w in zip(inds, wghts):
        cd = dict_struct(interpolator._libparams[i])
        txt += "\n{w:0.2f}@{i}: {logt:4.3f} {logg:3.2f}".format(i=i, w=w, **cd)

    return txt


def plot_interp(cwave, cflux, bflux, inds, wghts, interpolator,
                show_components=True, renorm_pars={}):

    bwave = interpolator.wavelengths
    _, bs = renorm([bwave, bflux], [cwave, cflux], **renorm_pars)
    fig, ax = pl.subplots()
    ax.plot(cwave[::5], cflux[::5], label="Charlie binary")
    ax.plot(bwave, bs, label="interpolated")
    ax.set_xlim(1e3, 3e4)

    if show_components:
        txt = comp_text(inds, wghts, interpolator)
        for i, w in zip(inds, wghts):
            if w > 0:
                _, bs = renorm([bwave, interpolator._spectra[i, :]],
                                [cwave, cflux], **renorm_pars)
                ax.plot(bwave, bs, label="comp. {}, wght={}".format(i, w), alpha=0.5)
        ax.text(0.3, 0.3, txt, transform=ax.transAxes)

    return fig, ax


def compare_at(charlie, interpolator, logg=4.5, logt=np.log10(5750.),
               show_components=False, renorm_pars={}):

    bpars, cwave, cspec = charlie
    sel = (bpars["logg"] == logg) & (bpars["logt"] == logt)
    if sel.sum() != 1:
        print("This set of parameters is not in the BaSeL grid: logg={}, logt={}".format(logg, logt))
        raise(ValueError)

    cflux = np.squeeze(cspec[sel, :])
    assert cflux.max() > 1e-33, ("requested spectrum has no "
                                 "Charlie spectrum: logg={}, logt={}".format(logg, logt))

    bwave = interpolator.wavelengths
    bflux, _, _ = interpolator.get_spectrum(logg=logg, logt=logt)
    inds, wghts = interpolator.weights(logg=logg, logt=logt)

    fig, ax = plot_interp(cwave, cflux, bflux, inds, wghts, interpolator)
    ax.set_title("target: {logt:4.3f} {logg:3.2f}".format(logg=logg, logt=logt))

    return fig, ax


def nearest_tg(interpolator, pars):
    """Do nearest neighbor but, first find the nearest logt then the nearest logg
    """
    tgrid = np.unique(interpolator._libparams["logt"])
    tt = tgrid[np.argmin(abs(tgrid - pars["logt"]))]
    thist = interpolator._libparams["logt"] == tt
    ggrid = np.unique(interpolator._libparams["logg"][thist])
    gg = ggrid[np.argmin(abs(ggrid - pars["logg"]))]
    choose = ((interpolator._libparams["logt"] == tt) &
              (interpolator._libparams["logg"] == gg))
    assert choose.sum() == 1
    ind = np.where(choose)[0][0]
    wght = 1.0
    return np.array([ind]), np.array([wght])


def extremeg(interpolator, pars):
    """Find if a set of parameters is below the lowest existing gravity for
    that temperature.
    """
    tgrid = np.unique(interpolator._libparams["logt"])
    tt = tgrid[np.argmin(abs(tgrid - pars["logt"]))]
    thist = interpolator._libparams["logt"] == tt
    ggrid = np.unique(interpolator._libparams["logg"][thist])
    return (pars["logg"] < ggrid.min()) or (pars["logg"] > ggrid.max())


def show_coverage(basel_pars, libparams, inds):
    false = np.zeros(len(basel_pars), dtype=bool)
    o, i, e = inds
    out, interp, extreme = false.copy(), false.copy(), false.copy() 
    out[o] = True
    interp[i] = True
    extreme[e] = True
    exact = (valid & (~out) & (~interp) & (~extreme))
    
    fig, ax = pl.subplots()
    ax.plot(basel_pars["logt"], basel_pars["logg"], 'o', alpha=0.2,
            label="BaSeL grid")
    ax.plot(basel_pars["logt"][exact], basel_pars["logg"][exact], 'o',
            label="exact")
    ax.plot(basel_pars["logt"][out], basel_pars["logg"][out], 'o',
            label="outside C3K")
    ax.plot(basel_pars["logt"][interp], basel_pars["logg"][interp], 'o',
            label="interpolated")
    ax.plot(basel_pars["logt"][extreme], basel_pars["logg"][extreme], 'o',
            label="nearest (t, g)")
    ax.plot(libparams["logt"], libparams["logg"], 'ko', alpha=0.3, label="C3K") 
    ax.invert_yaxis()
    ax.invert_xaxis()
    #ax.legend(loc=0)

    return fig, ax


    
if __name__ == "__main__":


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feh", type=float, default=0.0,
                        help=("The feh value to process."))
    parser.add_argument("--afe", type=float, default=0.0,
                        help=("The afe value to process."))
    parser.add_argument("--ck_vers", type=str, default='c3k_v1.3',
                        help=("Name of directory that contains the "
                              "version of C3K spectra to use."))
    parser.add_argument("--basedir", type=str, default='fullres/c3k/',
                        help=("Path to the directory containing sed HDF5 "
                              "flux files. Output will be placed here too."))
    parser.add_argument("--sedname", type=str, default="sed",
                        help=("nickname for the SED file, e.g. sedR500"))
    parser.add_argument("--outname", type=str, default="",
                        help=("Full name and path to the output fsps HDF5 file."))
    parser.add_argument("--zsol", type=float, default=0.0134,
                        help=("Name of directory that contains the "
                              "version of C3K spectra to use."))
    parser.add_argument("--renorm", type=bool, default=False,
                        help=("Whether to renormalize fluxes to the existing file."))
    parser.add_argument("--show_coverage", type=bool, default=False,
                        help=("Whether to make a coverage figure."))

    args = parser.parse_args()
    
    # --- Filenames ---
    z = args.zsol * 10**args.feh
    zstr = "{:1.4f}".format(z)

    template = "{}/{}_feh{:+3.2f}_afe{:+2.1f}.{}.h5"
    sedfile = template.format(args.basedir, args.ck_vers, args.feh, args.afe, args.sedname)
    outname = 'c3k_legac_z{:1.4f}.spectra.bin'.format(z)

    metname = "feh{:+3.2f}_afe{:+2.1f}".format(args.feh, args.afe)
    plot = "interp_{}.pdf".format(metname)

    # --- Charlie's interpolation ---
    basel_pars = get_basel_params()
    cwave, cspec, valid = get_binary_spec(len(basel_pars), zstr=zstr,
                                         speclib='CKC14/ckc14')
    # --- My interpolator ---
    interpolator = StarBasis(sedfile, use_params=['logg', 'logt'], logify_Z=False,
                             n_neighbors=1, verbose=False, rescale_libparams=True)

    # --- Do the interpolation ---
    bwave, bspec, inds = interpolate_to_basel([basel_pars, None, None], interpolator,
                                              valid=valid, renorm=args.renorm,
                                              plot=None)


    
    if args.show_coverage:
        fig, ax = show_coverage(basel_pars, inds, interpolator._libparams)
        ax.legend(loc=0)
        fig.savefig("coverage_{}.pdf".format(metname))
