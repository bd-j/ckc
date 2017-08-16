import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import glob
import h5py
import re

def plot_hrd(params):
    fig, ax = pl.subplots()
    ax.plot(params['logt'], params['logg'], 'o')
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
        
strfmt = '$N_{{spec}}={},\, N_{{pix}}={}$\n$[Fe/H]={}, \, [\\alpha/Fe]={}$'

pdf = PdfPages('c3k_hrd.pdf')
#files = np.sort(glob.glob("h5/*h5"))
files = ['../fullres/c3k/c3k_v1.3_feh+0.00.full.h5']
feh, afe = get_abundance(files)

for i, fn in enumerate(files):
    with h5py.File(fn, "r") as dat:
        nw = dat['wavelengths'].shape
        params = dat['parameters'][:]
    fig, ax = plot_hrd(params)
    ax.text(0.1, 0.85, strfmt.format(len(params), nw[0], feh[i], afe[i]),
            transform=ax.transAxes)
    pdf.savefig(fig)
    pl.close(fig)

pdf.close()
