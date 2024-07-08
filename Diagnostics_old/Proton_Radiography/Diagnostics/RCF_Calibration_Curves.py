# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:32:21 2022

@author: Adam Dearling (add525@york.ac.uk)

Finds the average pixel value and standard deviation for the RCF calibration data.

Contains the following functions for working with RCF data:
    - crop_calibration_data (removes edges from calibration data that make fitting hard)
    - gaussian_function(a gaussian function for fitting)
    - gaussian_fit (fits a gaussian function along 1/2 axis by taking the average)
    - calc_averae_circle (calculates the average value in a (semi-)circle of given radius)
    - calc_calibration_curve (uses the other functions to generate calibration data)
    - get_calibration_curve_2021 (gets the curves for 2021 data)
    - convert_intensity_OD (converts intensity to OD)
    - fit_calibration_curve (fits a calibration curve to a dataset)
    - get_calibration (gets the calibration data from a file)
    
Also includes some plotting functions.

Future improvements:
    - Calibration function won't return zero dose.
"""


# %% Libraries

import sys
import os
import numpy as np
from scipy.optimize import curve_fit

from RCF_Calibration_Extract import plot_calibration_curve
sys.path.insert(1, '../Python_Scripts/')
import Plot_Master as pm


# %% Functions

def convert_intensity_OD(sig_val, sig_dev, fog_val, fog_dev, test=False):
    '''
    Convert image intensity into OD using:
    OD = log10(I0/I) = ODsig - ODfog = log10(Ifog/Isig)
    Assumes Gaussian errors, which the pixel value error is not. 
    Additionally assumes that error is small, such that log10(1+a) with
    a = dx/x (fractional error) is equal to the Taylor expansion a+a^2/2+...
    so to first order the variation is a. In this case we then just take the
    average of the error above and below and propogate.
    '''

    n_val = fog_val/sig_val
    n_dev = n_val * np.sqrt((sig_dev/sig_val)**2 + (fog_dev/fog_val)**2)

    if (n_dev/n_val > 0.1).any():
        print("Fractional error is large, error propogation in this manner is "
              "not advised.")

    OD_val = np.log10(n_val)
    OD_dev = abs(np.log10(n_val-n_dev) - np.log10(n_val+n_dev))/2
    # OD_dev = (n_dev/n_val) * (1/np.log(10)) # Gives the exact same answer

    if test:
        sig_val_test = convert_OD_intensity(OD_val, OD_dev, fog_val, fog_dev)
        if abs(np.sum(sig_val_test-sig_val)) > np.amax(sig_val)*1e-9:
            raise RuntimeError("Error in OD conversion?")

    return OD_val, OD_dev


def convert_OD_intensity(OD_val, OD_dev, fog_val, fog_dev, test=False):
    '''
    Convert image OD to intensity using:
    Isig = Ifog / (10^OD)
    Error in n = Ifog/Isig = ln(10) * 10^OD * delta OD?
    '''

    n_val = np.power(10, OD_val)
    # n_dev = np.log(10) * n_val * OD_dev

    sig_val = fog_val/n_val
    # sig_dev = sig_val * np.sqrt((fog_dev/fog_val)**2 + (n_dev/n_val)**2)

    if test:
        OD_val_test, _ = convert_intensity_OD(sig_val, 0, fog_val, fog_dev)

    print("Converstion from OD error to px error not currently working!")

    return sig_val#, sig_dev


def convert_intensity_OD_RGB(im_val, im_dev, fog_val, fog_dev, input_OD=False):
    '''
    Convert image intensity into OD for a RGB image and vice versa
    '''

    con_val = np.zeros(im_val.shape)
    con_dev = np.zeros(im_val.shape)

    for n in range(3):
        if not input_OD:
            con_val[:,:,n], con_dev[:,:,n] = convert_intensity_OD(im_val[:,:,n], im_dev, 
                                                                  fog_val[n], fog_dev[n]) 
        else:
            con_val[:,:,n], con_dev[:,:,n] = convert_OD_intensity(im_val[:,:,n], im_dev, 
                                                                  fog_val[n], fog_dev[n])

    return con_val, con_dev


def calibration_curve_function(x, y_max, y_min, c, x_0):
    '''
    Function describing the OD calibration curve. Gafchromic suggest a function
    of the form OD =  -log( a + b*D / c + D) for HDV2 but the fit isn't great.
    It is unclear if there's a typo here, but for EBT3 they recommend a similar 
    function OD = a + b/(D-c) (which doesn't work at all).
    '''
    # return -np.log((a+b*x)/(c+x))
    # return a + b*x + c*x**2 + d*x**3
    return y_min + (y_max-y_min)/(1+10**(c*(np.log(x_0) - np.log(x))))



def fit_calibration_curve(dose_gy, average, deviation, dose_range=None, 
                          extrapolate=1, colour=None, plot=False):
    '''
    Fits a function to the calibration data.
    Currently only setup to work with OD data.
    Pixel value would require a different function.
    '''

    # Obtain parameters for fit function using data with standard deviation
    lower_bound = [0,-np.inf,0,0]
    upper_bound = [np.inf,0,np.inf,np.inf]

    popt, pcov = curve_fit(calibration_curve_function, dose_gy, average, #sigma=deviation,
                           bounds=(lower_bound, upper_bound), maxfev=10000)
    popt_lower, _ = curve_fit(calibration_curve_function, dose_gy, average-deviation,
                              bounds=(lower_bound, upper_bound), maxfev=10000)
    popt_upper, _ = curve_fit(calibration_curve_function, dose_gy, average+deviation,
                              bounds=(lower_bound, upper_bound), maxfev=10000)

    # Set dose range for fitting
    if dose_range is None:
        min_dose = np.min(dose_gy)
        max_dose = np.max(dose_gy)
    else:
        assert len(dose_range)==2, "Lower/upper doses should be provided."
        min_dose, max_dose = dose_range

    dose_gy_extra = np.logspace(np.log10(min_dose), # Not extrapolating low dose.
                                np.log10(max_dose*extrapolate), 25)



    average_fit = calibration_curve_function(dose_gy_extra, *popt)
    lower_fit = calibration_curve_function(dose_gy_extra, *popt_lower)
    upper_fit = calibration_curve_function(dose_gy_extra, *popt_upper)
    deviation_fit = abs(upper_fit-lower_fit)

    if plot:
        plot_calibration_curve(average, deviation, dose_gy, colour=colour, OD=True,
                               average_fit=average_fit, deviation_fit=deviation_fit, 
                               dose_fit=dose_gy_extra)

    return dose_gy_extra, average_fit, deviation_fit


def get_calibration(material, colour, scanner=None, material_type=None, 
                    OD=False, OD_fit=False, OD_extrapolate=1, version=None,
                    calibration="valid", plot=False):
    '''
    Gets calibration data for a given RCF material.
    This is setup for my formatted calibration files, which differ from DC's
    calibration files. Returns the dose, average px (or OD) value, and standard
    devation. The minimum/maximum values currently aren't returned, but there 
    are scenarios where this may be wanted...
    
    Data is obtained for desired material, colour channel and scanner.
    
    version: the file suffix, with the latest version by default assumed to have
        no suffix.
    calibration: "valid" only includes the region where noticable change occurs,
        roughly above null level or below saturation.
    '''

    if scanner is None:
        print("Scanner not selected. Exiting.")
        print("Available scanners are:")
        raise RuntimeError(os.listdir('Calibration/'))

    if material_type is None:
        print("Using default RCF material type (HDV2=3, EBT3=5)")
        if material == "HDV2":
            material_type = "3"
        elif material == "EBT3":
            material_type = "5"

    if version is None:
        version_suffix = ""
    else: # Suffix if latest version is not being used
        version_suffix = "_" + version

    raw_data = np.genfromtxt('Diagnostics/Proton_Radiography/Calibration/' + scanner + '/' + material
                             + "_" + material_type + "_calibration"
                             + version_suffix + '.csv',delimiter=',')

    # Would be better to find row/column numbers using headers etc.
    # Calibration files are slighly different based on material type
    if version == "Mk1": # I have functionality to remove this, but csv's must be the same.
        end = 35
    else:
        if material == "HDV2":
            end = 25 # There are 24 doses but one is missing, data start on line 3 so [2:25] = 23 points
        elif material == "EBT3":
            end = 16

    dose_gy = raw_data[2:end, 1]

    # Select calibration data for desired colour
    if colour == "R" or colour == 0:
        col = 3
    elif colour == "G" or colour == 1:
        col = 5
    elif colour == "B" or colour == 2:
        col = 7

    # Pixel values and standard deviation for dose data
    px_val = raw_data[2:end, col]
    dev_val = raw_data[2:end, col+1]

    # Null
    if version == "Mk1":
        null_val = raw_data[35, col]
        null_dev = raw_data[35, col+1]
    else:
        null_val = raw_data[25, col]
        null_dev = raw_data[25, col+1]

    # Minimum/maximum "valid" doses, a function could be written to define these
    if version == "Mk1":
        px_val_max = raw_data[36, col]
        px_val_min = raw_data[38, col] 
    else:
        px_val_max = raw_data[26, col]
        px_val_min = raw_data[28, col]

    # Format arrays so that cells which contain no data are excluded
    pos_valid = ~np.isnan(px_val)
    dose_gy = dose_gy[pos_valid]
    px_val = px_val[pos_valid]
    dev_val = dev_val[pos_valid]

    # Crop based on valid calibration data
    if calibration == "valid":
        pos_max = np.argmin(np.absolute(px_val-px_val_max))
        pos_min = np.argmin(np.absolute(px_val-px_val_min))

        dose_gy = dose_gy[pos_max:pos_min+1]
        px_val = px_val[pos_max:pos_min+1]
        dev_val = dev_val[pos_max:pos_min+1]

    elif calibration == "full":
        print("Caution: Using all calibration data.")
        # if px_val > null_val: # This data really should not be used.
        #     raise Exception("Pixel values exceed null value.")

    # For working with OD
    if OD:
        OD_val, OD_dev = convert_intensity_OD(px_val, dev_val, null_val, null_dev)
        OD_min, OD_max = convert_intensity_OD(np.array([px_val_max, px_val_min]), 0, 
                                              null_val, null_dev)[0]

        if OD_fit:
            dose_gy, OD_val, OD_dev = fit_calibration_curve(dose_gy, OD_val, OD_dev, 
                                                            extrapolate=OD_extrapolate, 
                                                            colour=colour, plot=True)

        ave_val, ave_dev = OD_val, OD_dev
        # min_val, max_val = OD_min, OD_max # max dose is max OD

    else:
        ave_val, ave_dev = px_val, dev_val
        # min_val, max_val = px_val_max, px_val_min # max dose is min px

    if plot: # For plotting calibration curve
        plot_calibration_curve(ave_val, ave_dev, dose_gy, OD=OD, colour=colour)

    return dose_gy, ave_val, ave_dev


def make_calibration_range_same(dose_gy, average, deviation, OD, interp=False, 
                                plot=False):
    ''' 
    Makes RGB calibration curves cover the same dose range. Data should not
    have been fitted at this point - can be in OD though.
    
    dose_gy, average and deviation should be tuples containing R, G, and B data.
    '''

    # Extract colour channels
    R_dose, G_dose, B_dose = dose_gy
    R_ave, G_ave, B_ave = average
    R_dev, G_dev, B_dev = deviation

    # Get min and max of dose range
    dose_min = max(R_dose[0], G_dose[0], B_dose[0])
    dose_max = min(R_dose[-1], G_dose[-1], B_dose[-1])

    if not interp: # Use calibration data as is    
        # The raw calibration values are taken at the same doses (currently...)
        if (dose_min == min(R_dose[0], G_dose[0], B_dose[0]) and 
            dose_max == max(R_dose[-1], G_dose[-1], B_dose[-1])):
            R_min = np.argmin(abs(R_dose-dose_min))
            R_max = np.argmin(abs(R_dose-dose_max))
            G_min = np.argmin(abs(G_dose-dose_min)) 
            G_max = np.argmin(abs(G_dose-dose_max))
            B_min = np.argmin(abs(B_dose-dose_min))
            B_max = np.argmin(abs(B_dose-dose_max))
        else: # They are all the same length
            R_min = G_min = B_min = 0
            R_max = G_max = B_max = len(R_ave)-1

        R_ave_s = R_ave[R_min:R_max+1]
        G_ave_s = G_ave[G_min:G_max+1]
        B_ave_s = B_ave[B_min:B_max+1]
        R_dev_s = R_dev[R_min:R_max+1]
        G_dev_s = G_dev[G_min:G_max+1]
        B_dev_s = B_dev[B_min:B_max+1]

        if (not np.array_equal(R_dose[R_min:R_max], G_dose[G_min:G_max]) or 
            not np.array_equal(R_dose[R_min:R_max], B_dose[B_min:B_max])):
            raise RuntimeError("Dose arrays are not the same.")

    else: # Interpolate between log of points to make curves smoother
        R_ave_log = np.log10(R_ave)
        R_sub_log = np.log10(R_ave-R_dev)
        R_add_log = np.log10(R_ave+R_dev)
        R_dose_log = np.log10(R_dose)
        G_ave_log = np.log10(G_ave)
        G_sub_log = np.log10(G_ave-G_dev)
        G_add_log = np.log10(G_ave+G_dev)
        G_dose_log = np.log10(G_dose)
        B_ave_log = np.log10(B_ave)
        B_sub_log = np.log10(B_ave-B_dev)
        B_add_log = np.log10(B_ave+B_dev)
        B_dose_log = np.log10(B_dose)

        dose_X = np.logspace(np.log10(dose_min), np.log10(dose_max), 100)
        dose_X_log = np.log10(dose_X)

        R_ave = 10**np.interp(dose_X_log, R_dose_log, R_ave_log)
        R_dev = abs(10**np.interp(dose_X_log, R_dose_log, R_sub_log)-10**np.interp(dose_X_log, R_dose_log, R_add_log))/2
        G_ave = 10**np.interp(dose_X_log, G_dose_log, G_ave_log)
        G_dev = abs(10**np.interp(dose_X_log, G_dose_log, G_sub_log)-10**np.interp(dose_X_log, G_dose_log, G_add_log))/2
        B_ave = 10**np.interp(dose_X_log, B_dose_log, B_ave_log)
        B_dev = abs(10**np.interp(dose_X_log, B_dose_log, B_sub_log)-10**np.interp(dose_X_log, B_dose_log, B_add_log))/2

    return (R_ave_s, G_ave_s, B_ave_s), (R_dev_s, G_dev_s, B_dev_s)


# %% Main script'
if __name__ == "__main__":

    project = "Calibration"
    # scanner = "Nikon_CoolScan9000"
    scanner = "Epson_12000XL"

    #Read calibration curves from file
    material = "HDV2"
    colour=2
    OD = True
    version=None

    dose, ave, dev = get_calibration(material, colour, scanner=scanner, material_type=None, 
                                     OD=OD, version=version, calibration="full", plot=False)

    fit_calibration_curve(dose, ave, dev, extrapolate=10, colour=colour, plot=True)
