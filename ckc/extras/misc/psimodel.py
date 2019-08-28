import numpy as np
from psi.model import MILESInterpolator

# The PSI Model
mlib = '/Users/bjohnson/Projects/psi/data/miles/miles_prugniel.h5'
fgk_bounds = {'teff': (3000.0, 10000.0)}
badstar_ids = np.array(allbadstars.tolist())

psi = MILESInterpolator(training_data=mlib, normalize_labels=False)
psi.features = (['logt'], ['feh'], ['logg'],
                ['logt', 'logt'], ['feh', 'feh'], ['logg', 'logg'],
                ['logt', 'feh'], ['logg', 'logt'], ['logg', 'feh'],
                ['logt', 'logt', 'logt'], ['logt', 'logt', 'logt', 'logt'], #logt high order
                ['logt', 'logt', 'logg'], ['logt', 'logg', 'logg'], # logg logt high order cross terms
                ['logt', 'logt', 'feh'], ['logt', 'feh', 'feh']  # feh logt high order cross terms
                )

