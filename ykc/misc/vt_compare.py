import os, sys
import numpy as np
import matplotlib.pyplot as pl
from smooth_fft import smooth_vel_fft, smooth_wave_fft

ckms = 2.998e5
Rin=3e5
vt=np.array(['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5'])

hires_fstring = '../Plan_Dan_Solar/Sync_Spectra_All/at12_teff={t:4.0f}_g={g:3.2f}_vturb={vt:s}.spec.gz'
def getflux_hires(fstring=hires_fstring, spectype='full', **pars):
    fn = fstring.format(**pars)
    if os.path.exists(fn) is False:
        print('did not find {}'.format(fn))
        return 0, 0
    fulldf = np.loadtxt(fn)
    wave = np.array(fulldf[:,0])
    full_spec = np.array(fulldf[:,1]) # spectra
    full_cont = np.array(fulldf[:,2]) # continuum
    if spectype == 'normalized':
        full_spec /= full_cont # normalized spectra
    elif spectype == 'continuum':
        full_spec = full_cont        
    return full_spec, wave


old_solar = '../old_solar/Spectra_R=300000/at12_teff={t:4.0f}_g={g:3.2f}_vturb={vt:s}.spec.gz'

stype = 'full'
Rout = 10000
inds = [0,3,6]
Rout_lambda = 1.0
smoothwave = True

if __name__ == "__main__":
    
    pars = {'t':4500, 'g': 1.0, 'vt': 3.0}
    flux_hires = [getflux_hires(t=pars['t'], g=pars['g'], vt=v, spectype=stype)[0] for v in vt[inds]]
    whires = getflux_hires(t=pars['t'], g=pars['g'], vt=vt[0])[1]

    wlo, whi, dlnlam, dlambda = 4e3, 1e4, 1.0 / Rout  / 2 / 2.35, Rout_lambda / 2.0 / 2.35
    if smoothwave:
        good = (whires > wlo) & (whires < whi)
        dw = np.diff(whires[good]).min()
        whires_constdlam = np.arange(wlo, whi, dw/2)
        fhires_constdlam = [np.interp(whires_constdlam, whires[good], f[good]) for f in flux_hires]
        outwave = np.arange(wlo, whi, dlambda)
        flux_lores = [smooth_wave_fft(whires_constdlam, f, outwave, sigma_out=Rout_lambda/2.35)
                      for f in fhires_constdlam]
        R = Rout_lambda
    else:
        outwave = np.arange(np.log(wlo), np.log(whi), dlnlam)
        outwave = np.exp(outwave)
        flux_lores = [smooth_vel_fft(whires, f, outwave, Rout=Rout*2.35, Rin=Rin*2.35) for f in flux_hires]
        R = Rout

    dv = ((float(vt[inds[1]])) - (float(vt[inds[0]]))) / ((float(vt[inds[2]])) - (float(vt[inds[0]])))
    log_lo = (1-dv) * np.log(flux_lores[0]) + dv * np.log(flux_lores[2])
    log_hi = (1-dv) * np.log(flux_hires[0]) + dv * np.log(flux_hires[2])

    fig, axes = pl.subplots(2,1, sharex=True)
    axes[0].plot(whires, np.exp(log_hi) / flux_hires[1],
                 label='R={} $v_t=${} interpolated / true'.format(Rin, vt[inds[1]]))
    axes[1].plot(outwave, np.exp(log_lo) / flux_lores[1],
                 label='R={} $v_t=${} interpolated / true'.format(R, vt[inds[1]]))
    [ax.legend(loc=0) for ax in axes]
    axes[0].set_xlim(4e3, 1.1e4)
    fig.show()

    sys.exit()
    
    fig, axes = pl.subplots(2,1, sharex=True)
    axes[0].plot(whires, flux_hires[0], label='R={} vt={}'.format(Rin, vt[inds[0]]))
    axes[0].plot(whires, flux_hires[2], label='R={} vt={}'.format(Rin, vt[inds[2]]))
    axes[1].plot(outwave, flux_lores[0], label='R={} vt={}'.format(R, vt[inds[0]]))
    axes[1].plot(outwave, flux_lores[2], label='R={} vt={}'.format(R, vt[inds[2]]))
    axes[0].set_xlim(5850, 5910)
    [ax.legend(loc=0) for ax in axes]
    fig.show()

    sys.exit()
    
    fig, axes = pl.subplots(2,1, sharex=True)
    axes[0].plot(whires, np.exp(log_hi), label='R={} vt={} interpolated'.format(Rin, vt[inds[1]]))
    axes[0].plot(whires, flux_hires[1], label='R={} vt={}'.format(Rin, vt[inds[1]]))
    axes[1].plot(outwave, np.exp(log_lo), label='R={} vt={} interpolated'.format(R, vt[inds[1]]))
    axes[1].plot(outwave, flux_lores[1], label='R={} vt={}'.format(R, vt[inds[1]]))
    axes[0].set_xlim(4e3, 1.1e4)
    [ax.legend(loc=0) for ax in axes]
    fig.show()

