import numpy as np
import sys
import h5py
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as pl

def test_interp_spsbasis(lib):
    """Test interpolation error by leaving one spectrum out of the
    model, computing the triagulation, and getting the spectrum for
    the parameters of the missing model.  Returns the ratio of the
    interpolated to the computed models.
    """
    from bsfh import sps_basis
    sps = StarModel(lib=lib)
    libpars = sps._libparams
    specs = sps._spectra
    nmod = len(libpars)
    ratio = np.zeros(nmod, len(sps._wave))
    mask = np.ones(nmod, dtype=bool)
    for i in xrange(nmod):
        par, spec = libpars[i], specs[i, :]
        mask[:] = True
        mask[i] = False
        sps._libparams = libpars[mask]
        sps._spectra = specs[mask, :]
        sps.triangulate()
        pd = {}
        for n in par.dtype.names:
            pd[n] = par[n]
        w, s, u = sps.get_star_spectrum(logarithmic=True, **pd)
        ratio[i, :] = s/spec
    return ratio
        
def test_interp_linear(filename='../h5/ckc14_fullres.h5'):
    """Test interpolation error using linear interpolation.
    """
    fdat = h5py.File(filename, "r")
    
    wave = fdat['wavelengths'][:]
    zlist = fdat['spectra'].keys()
    logg = fdat['logg'][:]
    logt = fdat['logt'][:]
    nspec = len(logg) * len(logt) #len(zlegend)

    z = zlist[3]
    spec = fdat['spectra'][z][:]
    spec = spec.reshape(len(logg), len(logt), len(wave))
    intspec = np.empty_like(spec)

    wlo, whi = 3000, 10000
    wmin = np.argmin(np.abs(wave - wlo))
    wmax =  np.argmin(np.abs(wave - whi))
    wlabels = np.linspace(wlo, whi, 5)
    winds = [np.argmin(np.abs(wave - w)) - wmin for w in wlabels]

    flag = np.zeros([len(logg), len(logt)], dtype=bool)
    pdf = PdfPages('Tinterp.pdf')
    # Temperature interpolation
    for gind, g in enumerate(logg):
        for tind, t in enumerate(logt):
            tlo = max([tind-1, 0])
            thi = min([tind+1, len(logt)-1])
            dT = (logt[tind] - logt[tlo]) / (logt[thi] - logt[tlo])
            s1, s2 = spec[gind, tlo, :], spec[gind, thi, :]
            if (s1.max() < 1e-32) | (s2.max() < 1e-32) | (spec[gind, tind].max() < 1e-32):
                temp = -np.inf
                flag[gind, tind] = True
            temp = (1-dT) * np.log(s1) + dT * np.log(s2)
            intspec[gind, tind,:] = np.exp(temp)
        fig, ax = pl.subplots()
        ratio = intspec[gind, :, wmin:wmax] / spec[gind, :, wmin:wmax]
        ratio[flag[gind, :], :] = 0
        c = ax.imshow(np.clip(np.log(ratio), -1, 1), vmin=-1, vmax=1)
        ax.set_xlabel('wavelength ($\AA$)')
        ax.set_xticks(winds)
        ax.set_xticklabels(wlabels)
        ax.set_ylabel('logT')
        labinds = np.arange(0, len(logt), 10)
        ax.set_yticks(labinds)
        ax.set_yticklabels([logt[lab] for lab in labinds])
        ax.set_title('Z={0}, logg={1}'.format(z[1:], g))
        cbar = fig.colorbar(c)
        cbar.set_label('ln (interp/true)')
        fig.savefig(pdf, format='pdf')
        pl.close(fig)
    pdf.close()
    #sys.exit()

    flag = np.zeros([len(logg), len(logt)], dtype=bool)
    # logg interpolation
    pdf = PdfPages('ginterp.pdf')
    for tind, t in enumerate(logt):
        for gind, g in enumerate(logg):
            glo = max([gind-1, 0])
            ghi = min([gind+1, len(logg)-1])
            dG = (logg[gind] - logg[glo]) / (logg[ghi] - logg[glo])
            s1, s2 = spec[glo, tind, :], spec[ghi, tind, :]
            if (s1.max() < 1e-32) | (s2.max() < 1e-32) | (spec[gind, tind].max() < 1e-32):
                temp = -np.inf
                flag[gind, tind] = True
            temp = (1-dG) * np.log(s1) + dG * np.log(s2)
            intspec[gind, tind,:] = np.exp(temp)
        fig, ax = pl.subplots()
        ratio = intspec[:, tind, wmin:wmax] / spec[:, tind, wmin:wmax]
        ratio[flag[:, tind], :] = 0
        c = ax.imshow(np.clip(np.log(ratio), -1, 1), vmin=-1, vmax=1)
        ax.set_xlabel('wavelength ($\AA$)')
        ax.set_xticks(winds)
        ax.set_xticklabels(wlabels)
        ax.set_ylabel('logg')
        labinds = np.arange(0, len(logg), 5)
        ax.set_yticks(labinds)
        ax.set_yticklabels([logg[lab] for lab in labinds])
        ax.set_title('Z={0}, logT={1}'.format(z[1:], t))
        cbar = fig.colorbar(c)
        cbar.set_label('ln (interp/true)')
        fig.savefig(pdf, format='pdf')
        pl.close(fig)
    pdf.close()
    

#    # logg z interpolation
#    for iz, z 
