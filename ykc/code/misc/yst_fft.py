# import packages
import numpy as np

import scipy as sp
from scipy import interpolate
from scipy import signal
from scipy.stats import norm
from scipy.optimize import minimize

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import random
import os
import itertools
import time
from multiprocessing import Pool

#------------------------------------------------------------------------------------------------------------
def convolve_spectrum(filename_in):

    # survey resolution
    R_res = 20000.
    
    # initialize wavelength grid
    start_wavelength = 3000.
    end_wavelength = 18000.

    # wavelength batch in Angstrom (within which kernel is the same)
    wavelength_batch = 100
    
#-----------------------------------------------------------------------------------------------------------
    # interpolation parameters
    inverse_wavelength_resolution = 100
    wavelength_resolution = 1./inverse_wavelength_resolution
    
    # FFT parameter
    kernel_support = 10
    
    # range of convolution kernel
    x_range = np.arange(inverse_wavelength_resolution*kernel_support+1)\
        *wavelength_resolution - (kernel_support/2.)
            
#-----------------------------------------------------------------------------------------------------------
    # load spectrum
    fulldf = np.load("./Sync_Spectra_All_Npz/" + filename_in + ".npz")
    
    # extract properties
    wavelength = fulldf["wavelength"]
    full_spec = fulldf["full_spec"]
    full_cont = fulldf["full_cont"]

#-----------------------------------------------------------------------------------------------------------
    # verify end points
    if start_wavelength < wavelength[0]:
        return
    if end_wavelength > wavelength[-1]:
        return
        
#-----------------------------------------------------------------------------------------------------------
    # restrict wavelength
    ind_wave = np.logical_and(wavelength >= start_wavelength, \
                                  wavelength <= end_wavelength)
    
    wavelength = wavelength[ind_wave]
    full_spec = full_spec[ind_wave]
    full_cont = full_cont[ind_wave]
 
    # round up end points
    wavelength[0] = start_wavelength
    wavelength[-1] = end_wavelength

#-----------------------------------------------------------------------------------------------------------
    # interpolate flux
    f_flux_spec = interpolate.interp1d(wavelength, full_spec)
    f_flux_cont = interpolate.interp1d(wavelength, full_cont)
    
    # make wavelength grid
    wavelength = np.arange((end_wavelength-start_wavelength)\
                               *inverse_wavelength_resolution + 1)\
                               *wavelength_resolution + start_wavelength

    # interpolate
    full_spec = f_flux_spec(wavelength)
    full_cont = f_flux_cont(wavelength)

#-----------------------------------------------------------------------------------------------------------
    # initialize convolved spectrum
    convolved_spec = []
    convolved_cont = []

    # number of entries per wavelength batch
    num_per_batch = int(wavelength_batch*inverse_wavelength_resolution)
    
    # initialize counter
    count_wave = 0
    
    # total number of wavelength pixels
    num_pixels = wavelength.shape[0]

#-------------------------------------------------------------------------------------------------------------
    # loop over all wavelengths
    while count_wave < num_pixels:

        # find the region for convolution (pad head and tail)
        start_ind = count_wave - num_per_batch
        end_ind = count_wave + 2*num_per_batch
        
        if count_wave == 0:
            start_ind = 0

#------------------------------------------------------------------------------------------------------------
        # the recording index (without padding)
        start_record = num_per_batch
        end_record = 2*num_per_batch
        
        if count_wave == 0:
            start_record = 0
            end_record = num_per_batch
        
#-----------------------------------------------------------------------------------------------------------
        # kernel std
        range_wavelength = np.median(wavelength[start_ind:end_ind])/R_res/2.355
    
        # convolution kernel
        kernel = norm.pdf(x_range, scale=range_wavelength)*wavelength_resolution

        # convolve spectra
        convolved_spec.extend(\
                signal.fftconvolve(full_spec[start_ind:end_ind],\
                                   kernel, mode='same')[start_record:end_record])
        convolved_cont.extend(\
                signal.fftconvolve(full_cont[start_ind:end_ind],\
                                   kernel, mode='same')[start_record:end_record])

#-----------------------------------------------------------------------------------------------------------
        # increase counter
        count_wave += num_per_batch

#----------------------------------------------------------------------------------------------------------
    # save the final result
    np.savez("./Sync_Spectra_All_Convolved_R=20000/" + filename_in + ".npz",\
             wavelength=wavelength,\
             convolved_spec=convolved_spec,\
             convolved_cont=convolved_cont)


#================================================
# list all the files
list_files = os.listdir("./Sync_Spectra_All")

#---------------------------------------------------------------------------------------------------------
## run in parallel ##
# number of processors
num_processor = 200

# multithread processing
p = Pool(num_processor)
p.map(convolve_spectrum,list_files)
p.terminate()
