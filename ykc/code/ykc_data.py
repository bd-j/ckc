
__all__ = ["ckms", "sigma_to_fwhm", "Rykc",
           "full_params", "param_order", "hires_fstring"]

ckms = 2.998e5
sigma_to_fwhm = 2.355
Rykc = 3e5

full_params = {'t': np.arange(4000, 5600, 100),
               'g': [1.0, 1.5, 2.0],
               'feh': np.arange(-4, 1.0, 0.5),
               'afe': [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
               'nfe': [0, 0.3], 'cfe': [0, 0.3],
               'vturb':[0.5, 3.5]}

param_order = ['t', 'g', 'feh', 'afe', 'cfe', 'nfe', 'vturb']

hires_fstring = ("at12_teff={t:4.0f}_g={g:3.2f}_feh={feh:3.1f}_"
                 "afe={afe:3.1f}_cfe={cfe:3.1f}_nfe={nfe:3.1f}_"
                 "vturb={vturb:3.1f}.spec.gz")
