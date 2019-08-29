import numpy as np
from flatlibs import make_lib_flatfull

if __name__ == "__main__":

    # The CKC resolution intervals are
    # R = 500,   100   < lambda < 1500
    # R = 10000, 1500  < lambda < 11000
    # R = 2000,  11000 < lambda < 30000
    # R = 50,   30000 < lambda < 1000000
    
    # account for intrinsic resolution of the CKC grid (10000) to get
    # a desired resolution of 1000
    r = 1/np.sqrt((1./4700)**2 - (1./np.array([10000, 4000])**2))
    irtf = {'R': r, 'wmin': [5000, 11000], 'wmax': [10999, 30000],
             'h5name': '../h5/ckc14_fullres.flat.h5',
             'outfile': '../lores/irtf/ckc14__R2000.flat.h5',
             'velocity': True
             }
    make_lib_flatfull(**irtf)
