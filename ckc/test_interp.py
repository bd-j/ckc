import numpy as np
import sys
import h5py

if __name__=="__main__":

    fdat = h5py.File('../h5/ckc14_fullres.h5', "r")
    
    wave = fdat['wavelengths'][:]
    zlist = fdat['spectra'].keys()
    logg = fdat['logg'][:]
    logt = fdat['logt'][:]
    nspec = len(logg) * len(logt) #len(zlegend)

    z = zlist[3]
    spec = fdat['spectra'][z][:]
    spec = spec.reshape(len(logg), len(logt), len(wave))
    intspec = np.empty_like(spec)
    
    pdf = PdfPages('Tinterp.pdf')
    # Temperature interpolation
    for gind, g in enumerate(logg):
        for tind, t in enumerate(logt):
            tlo = max([tind-1, 0])
            thi = min([tind+1, len(logt)-1])
            dT = (logt[tind] - logt[tlo]) / (logt[thi] - logt[tlo])
            intspec[gind, tind,:] = (1-dT) * spec[gind, tlo, :] + dT * spec[gind, thi, :]
        c = ax.imshow(np.log(np.clip(intspec[gind, :, wmin:wmax], 0, 3)))
        ax.set_xticks(winds)
        ax.set_xticklabels(wlabels)
        ax.set_title('logg={}'.format(g))
        ax.colorbar(c)
    sys.exit()
    # logg interpolation
    for gind, g in enumerate(logg):
        for tind, t in enumerate(logt):
            dG = (logg[gind] - logg[gind-1]) / (logg[gind+1] - logg[gind-1])
            intspec[gind, tind,:] = (1-dG) * spec[gind-1, tind, :] + dG * spec[gind+1, tind, :]

#    # logg z interpolation
#    for iz, z 
