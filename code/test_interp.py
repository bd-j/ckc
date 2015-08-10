import numpy as np
import struct

def read_binary_spec(filename, nw, nspec):
    count = 0
    spec = np.empty([nspec, nw])
    with open(filename, 'rb') as f:
        while count < nspec:
                count += 1
                for iw in range(nw):
                    byte = f.read(4)
                    spec[count-1, iw] = struct.unpack('f', byte)[0]
    return spec


if __name__=="__main__":
    
    wave = np.loadtxt('ckc14.lambda')
    zlegend = np.loadtxt('zlegend.dat')
    zlist = ['{0:06.4f}'.format(z) for z in zlegend]
    logg = np.loadtxt('../BaSeL3.1/basel_logg.dat')
    logt = np.loadtxt('../BaSeL3.1/basel_logt.dat')
    nspec = len(logg) * len(logt) #len(zlegend)

    z = zlist[3]
    name = 'ckc14_z{0}.spectra.bin'.format(z)
    spec = read_binary_spec(name, len(wave), nspec)
    spec = spec.reshape(len(logg), len(logt), len(wave))

    gind = 18
    for tind, t in enumerate(logt):
        dT = (logt[tind] - logt[tind-1]) / (logt[tind+1] - logt[tind-1])
        Tintspec[gind, tind,:] = (1-dT) * spec[gind, tind-1, :] + dT * spec[gind, tind+1, :]
