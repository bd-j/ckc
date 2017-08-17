import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import glob
import h5py
import re

def plot_hrd(params, **plot_kwargs):
    fig, ax = pl.subplots()
    ax.plot(params['logt'], params['logg'], 'o', **plot_kwargs)
    ax.set_xlim(4.7, 3.3)
    ax.set_ylim(5.6, -1.1)
    ax.set_ylabel('log g')
    ax.set_xlabel('log t')
    return fig, ax


def get_abundance(files):
    zpat, zs = re.compile(r"_feh.{5}"), slice(4, None)
    apat, asl = re.compile(r"_afe.{4}"), slice(4, None)

    feh = [float(re.findall(zpat, f)[-1][zs]) for f in files]
    try:
        afe = [float(re.findall(apat, f)[-1][asl]) for f in files]
    except:
        afe = afe = len(feh) * [0.0]
    
    return feh, afe


def plot_isoc(isoc, logage=9.0, feh=0.0,
              ax=None, **plot_kwargs):
    sel = (isoc['EAF']['log_age'] == logage) & (isoc['EAF']['initial_[Fe/H]'] == feh)
    x = isoc['BSP'][sel]['log_Teff']
    y = isoc['BSP'][sel]['log_g']
    order = np.argsort(isoc['EAF'][sel]['EEP'])
    ax.plot(x[order], y[order], **plot_kwargs)
    return ax


if __name__ == "__main__":

    isocfile = 'MIST_full.h5'
    isoc = h5py.File(isocfile, "r")
    strfmt = '$N_{{spec}}={},\, N_{{pix}}={}$\n$[Fe/H]={}, \, [\\alpha/Fe]={}$'

    ages = [7, 8, 9, 9.5, 10]
    isocolor = [u'#5DA5DA', u'#FAA43A', u'#60BD68', u'#F15854', u'#39CCCC',
                u'#FF70B1', u'#FFDC00', u'#85144B', u'#4D4D4D']
    pdf = PdfPages('c3k_hrd.pdf')

    files = ['../fullres/c3k/c3k_v1.3_feh+0.00.full.h5']
    files = np.sort(glob.glob("/n/home02/bdjohnson/fs/data/stars/c3k_v1.3/h5/*h5"))
    feh, afe = get_abundance(files)

    for i, fn in enumerate(files):
        with h5py.File(fn, "r") as dat:
            nw = dat['wavelengths'].shape
            params = dat['parameters'][:]
        fig, ax = plot_hrd(params, color='k')
        ax.text(0.1, 0.85, strfmt.format(len(params), nw[0], feh[i], afe[i]),
                transform=ax.transAxes)
        for j, age in enumerate(ages):
            plot_isoc(isoc, logage=age, feh=feh[i], ax=ax, color=isocolor[j])
        pdf.savefig(fig)
        pl.close(fig)

    pdf.close()
