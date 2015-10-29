__all__ = ["irtf", "miles", "deimos", "manga"]

# The CKC resolution intervals are
# R = 500,   100   < lambda < 1500
# R = 10000, 1500  < lambda < 11000
# R = 2000,  11000 < lambda < 30000
# R = 50,   30000 < lambda < 100000000
# where R is defined in terms of FWHM

# account for intrinsic resolution of the CKC grid (10000) to get
# a desired resolution of R_target_FWHM

#  IRTF
wmin = [1500, 3650, 11000]
wmax = [3650, 11000, 30000]
outres = [100*2.35, 2000*2.35, 2000*2.35]  # R of sigma in terms of 
inres = [10000 * 2.35, 10000 * 2.35, 2000*2.35]
inres = [2.998e5 / r  for r in inres]

irtf = {'R': outres, 'wmin': wmin, 'wmax': wmax,
        'inres':inres, 'in_vel': True, 'velocity': True,
        'absmaxwave': 3e4, 'lores': 100,
        'h5name': '../h5/ckc14_fullres.flat.h5',
        'outfile': '../lores/irtf/ckc14_irtf.flat.h5',             
        }

# MILES
wmin = [1500, 3500]
wmax = [3500, 10000]
outres = [50 / 2.35, 2.54 / 2.35] # sigma in AA
inres = [10000 * 2.35, 10000 * 2.35, 2000 * 2.35] # sigma in R
inres = [2.998e5 / r  for r in inres] # sigma in km/s

miles = {'R': outres, 'wmin': wmin, 'wmax': wmax,
        'inres':inres, 'in_vel': True, 'velocity': False,
        'absmaxwave': 3e4, 'lores': 100,
        'h5name': '../h5/ckc14_fullres.flat.h5',
        'outfile': '../lores/irtf/ckc14_miles.flat.h5',             
        }

# Deimos
wmin = [100, 1500, 3650, 9900, 11000]
wmax = [1500, 3650, 9900, 11000, 30000]
r = [50, 30, 1.3 /2.35, 30, 100]
inres = [500, 10000, 10000, 10000, 2000]  # R of input in terms of lambda/FWHM
inres = [2.998e5 / (r * 2.35)  for r in inres] # sigma of input in km/s

deimos = {'R': r, 'wmin': wmin, 'wmax': wmax,
          'inres':inres, 'in_vel': True, 'velocity': False,
          'absmaxwave': 3e4, 'lores': 100,
          'h5name': '../h5/ckc14_fullres.flat.h5',
          'outfile': '../lores/deimos_R1/ckc14_deimos_R1AA.flat.h5',             
         }

# MANGA
wmin = [100, 1500, 3500, 11000, 30000]
wmax = [1500, 3500, 11000, 30000, 100000000]
r = [50 * 2.35, 100 * 2.35, 2400 * 2.35, 50 * 2.35, 25 * 2.35] #R of output in terms of sigma
inres = [500, 10000, 10000, 2000, 50]  # R of input in terms of lambda/FWHM
inres = [2.998e5 / (r * 2.35)  for r in inres] # sigma of input in km/s

manga = {'R': r, 'wmin': wmin, 'wmax': wmax,
          'inres':inres, 'in_vel': True, 'velocity': False,
          'absmaxwave': 3e4, 'lores': 100,
          'h5name': '../h5/ckc14_fullres.flat.h5',
          'outfile': '../lores/manga/ckc14_manga.flat.h5',             
         }
