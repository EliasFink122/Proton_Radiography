# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 17:41:38 2023

@author: Adam Dearling (add525@york.ac.uk)

Finds the average pixel value and standard deviation for the RCF calibration data.

Contains the following functions for working with RCF data:
    - crop_calibration_data (removes edges from calibration data that make fitting hard)
    - gaussian_function(a gaussian function for fitting)
    - gaussian_fit (fits a gaussian function along 1/2 axis by taking the average)
    - calc_averae_circle (calculates the average value in a (semi-)circle of given radius)
    - calc_calibration_curve (uses the other functions to generate calibration data)
    - get_calibration_curve_2021 (gets the curves for 2021 data)
    
Also includes some plotting functions.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from RCF_Basic import get_radiography_data
sys.path.insert(1, '../Python_Scripts/')
import Plot_Master as pm


# %% Functions

def crop_calibration_data(data, crop=None, material="", scanner="Nikon_CoolScan9000",
                          plot=False):
    '''Crops edges from the calibration data to make the Gaussian fitting easier.'''
    
    data_cropped = []
    
    # Set crop boundaries - I don't think this method is particularily good
    # but it is quick (just set once for all layers)
    if scanner == "Nikon_CoolScan9000":
        
        # Boundaries I used in my first calibration attempt
        # I just averaged over this region?..
        if crop == "V1":
            bound_left = 475
            bound_right = 575
            bound_lower = 475
            bound_upper = 575
        
        elif crop == "V2": # Boundaries for Gaussian fitting to EBT3
            # if material == "HDV2": # I had the same boundaries set for EBT3
            bound_left = 175
            bound_right = 875
            bound_lower = 175
            bound_upper = 875        
                
    elif scanner == "Epson_12000XL":
        bound_left = 225
        bound_right = 625
        bound_lower = 175
        bound_upper = 575

    for ndose in range(len(data)):
        
        # Crop data
        if data[ndose].shape[0] < 2*bound_lower or data[ndose].shape[1] < 2*bound_left:
            data_cropped.append(data[ndose]) # Mainly for DS0 layer (as this is v small for Epson scan)
        else:
            data_cropped.append(data[ndose][bound_lower:bound_upper, bound_left:bound_right,:])
        
        # Plot data and show crop area
        if plot == True:
            
            fig_crop, (ax_raw, ax_Rcrop, ax_Gcrop, ax_Bcrop) = plt.subplots(1,4,figsize=(16,4))
            
            ax_raw.imshow(data[ndose][:,:,0], cmap="hsv")
            ax_raw.plot([0,1000],[bound_upper,bound_upper])
            ax_raw.plot([0,1000],[bound_lower,bound_lower])
            ax_raw.plot([bound_left, bound_left],[0,1000])
            ax_raw.plot([bound_right, bound_right],[0,1000])
            ax_raw.set_ylim(ymin=data[ndose][:,:,0].shape[0]) # imshow being a pain
            ax_raw.set_xlim(xmax=data[ndose][:,:,0].shape[1])

            ax_Rcrop.imshow(data_cropped[ndose][:,:,0], cmap="hsv")
            ax_Gcrop.imshow(data_cropped[ndose][:,:,1], cmap="hsv")
            ax_Bcrop.imshow(data_cropped[ndose][:,:,2], cmap="hsv")
            
            fig_crop.tight_layout()
        
    return data_cropped


def gaussian_function(x,a,x0,sigma,b):
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) + b # var = sigma**2


def gaussian_fit(data, analysis="calib", xfit=True, yfit=True, plot=False, test=False):
    '''
    Used to fit Gaussians to RGB data sets in x and y.
    The data is averaged over the whole axis that the fitting is being performed over
    Each colour channel is treated separetly.
    
    Returns the centre position and standard deviation (width) obtained from
    the fit. If a dimension is not wanted 0 is returned.
    
    analysis: can either be performed for "calib"(ration) or "dose" data
        with the difference being related to the shape of the Gaussian and
        resulting curve fit input.
    xfit: if True (default) fit is performed along y-axis.
    yfit: if True (default) fit is performed along x-axis.
    '''
    
    if xfit == False or yfit == False:
        print("Performing Gaussian fit in 1 axis. Other axis centre will be set to 0.")
    
    nlayers = len(data)
    
    centre = np.zeros((2, 3, nlayers))
    sigma = np.zeros((2, 3, nlayers))
    
    colours = ["r", "g", "b"]
    
    for nlayer in range(nlayers):
        
        print("Fitting layer {}.".format(nlayer))
    
        layer_data = data[nlayer]
        
        nx = layer_data.shape[1]
        ny = layer_data.shape[0]
        x = np.arange(0,nx)
        y = np.arange(0,ny)
        
        if plot==True:
            if xfit and yfit:
                fig, (ax_x, ax_y) = pm.plot_figure_axis("small", 2)
            elif xfit:
                fig, ax_x = pm.plot_figure_axis("small", 1)
            elif yfit:
                fig, ax_y = pm.plot_figure_axis("small", 1)
        
        for colour in range(3): # Should be 3
            data_x = np.average(layer_data[:,:,colour], axis=0)
            data_y = np.average(layer_data[:,:,colour], axis=1)
            
            # mean = np.average(data_x)
            # sigma_x = np.sqrt(np.sum((data_x-mean)**2)/nx)
            # sigma_y = np.std(data_y)
            max_x = np.amax(data_x)
            max_y = np.amax(data_y)
            min_x = np.amin(data_x)
            min_y = np.amin(data_y)
            amp_x = min_x - max_x
            amp_y = min_y - max_y
            mid_x = int(nx/2)
            mid_y = int(ny/2)
            
            # If fitting to raw data, pixel count decreases with proton dose.
            # This means the Gaussian fit will have a negative amplitude
            # Else if fitting to dosed data Gaussian will have a positive amplitude.
            if analysis == "calib":
                amp_x = min_x - max_x
                amp_y = min_y - max_y
                
                guess_x = gaussian_function(x, amp_x, mid_x, nx/4, max_x)
                guess_y = gaussian_function(y, amp_y, mid_y, ny/4, max_y)
        
            elif analysis == "dose":
                amp_x = max_x - min_x
                amp_y = max_y - min_y
            
                guess_x = gaussian_function(x, amp_x, mid_x, nx/4, 0)
                guess_y = gaussian_function(y, amp_y, mid_y, ny/4, 0)
            
            # popt returns an array with the optimised fitting parameters
            # pcov is convariance (use np.diag to get error)
            if analysis == "calib":
                popt_x, pcov_x = curve_fit(gaussian_function, x, data_x, p0=[amp_x, mid_x, nx/4, max_x],
                                            maxfev=10000, bounds=(-np.inf,[0, +np.inf, +np.inf, +np.inf]))
                popt_y, pcov_y = curve_fit(gaussian_function, y, data_y, p0=[amp_y, mid_y, ny/4, max_y],
                                            maxfev=10000, bounds=(-np.inf,[0, +np.inf, +np.inf, +np.inf]))
            
            elif analysis == "dose":
                if xfit:
                    popt_x, pcov_x = curve_fit(gaussian_function, x, data_x, p0=[amp_x, mid_x, nx/4, 0], 
                                                maxfev=10000, bounds=([0, -np.inf, -np.inf, -np.inf], +np.inf))
                if yfit:
                    popt_y, pcov_y = curve_fit(gaussian_function, y, data_y, p0=[amp_y, mid_y, ny/4, 0],
                                                maxfev=10000, bounds=([0, -np.inf, -np.inf, -np.inf], +np.inf))
            
            else:
                popt_x, pcov_x = curve_fit(gaussian_function, x, data_x, p0=[amp_x, mid_x, nx/4, max_x],
                                            maxfev=10000)
                popt_y, pcov_y = curve_fit(gaussian_function, y, data_y, p0=[amp_y, mid_y, ny/4, max_y],
                                            maxfev=10000)
        
            if plot:
                if xfit:
                    ax_x.plot(data_x, color=colours[colour])
                    if test: # Plot guess if there are issues with fitting.
                        ax_x.plot(guess_x, color="k", linestyle="--")
                    ax_x.plot(gaussian_function(x,*popt_x), color=colours[colour], linestyle="--")
            
                    ax_x.set_xlabel("x pixel")
                    if analysis == "calib":
                        ax_x.set_ylabel("pixel value")
                    elif analysis == "dose":
                        ax_x.set_ylabel("Dose (Gy)")
                    ax_x.set_xlim(xmin=0, xmax=nx)
        
                if yfit:
                    ax_y.plot(data_y, color=colours[colour])
                    if test: # Plot guess if there are issues with fitting.
                        ax_y.plot(guess_y, color="k", linestyle="--")
                    ax_y.plot(gaussian_function(y,*popt_y), color=colours[colour], linestyle="--")
                    
                    ax_y.set_xlabel("y pixel")
                    if analysis == "calib":
                        ax_y.set_ylabel("pixel value")
                    elif analysis == "dose":
                        ax_y.set_ylabel("Dose (Gy)")
                    ax_y.set_xlim(xmin=0, xmax=ny)
                
                fig.tight_layout()
            
            if xfit and yfit:
                centre[0, colour, nlayer], centre[1, colour, nlayer] = popt_x[1], popt_y[1]
                sigma[0, colour, nlayer], sigma[1, colour, nlayer] = popt_x[2], popt_y[2]
            elif xfit:
                centre[0, colour, nlayer], centre[1, colour, nlayer] = popt_x[1], 0
                sigma[0, colour, nlayer], sigma[1, colour, nlayer] = popt_x[2], 0
            else:
                centre[0, colour, nlayer], centre[1, colour, nlayer] = 0, popt_y[1]
                sigma[0, colour, nlayer], sigma[1, colour, nlayer] = 0, popt_y[2]
                
        print("Layer {} image centre: x = {:.0f}, y = {:.0f}.".format(nlayer, centre[0,0,nlayer], centre[1,0,nlayer]))
        print("Layer {} image std: x = {:.1f}, y = {:.1f}".format(nlayer, sigma[0,0,nlayer], sigma[1,0,nlayer]))
        
    return centre, sigma
    

def calc_average_circle(data, centre, radius, fixed=False, semicircle=None, 
                        xoffset=None, yoffset=None, plot=False, test=False):
    '''
    Calculates the average value of a circle with a given radius.
    Requires the position corresponding to the centre of the circle.
    This posision can be "fixed" so that the same value is used for 
    all layers, or set for each layer separetly.
    
    Returns the average and standard deviation.
    
    semicircle: allows either the top (+1) or bottom (-1) half of a circle to
        be used instead of a full circle.
    xoffset: offset of circle from centre in x.
    yoffset: offset of circle from centre in y.
    '''
    
    nlayers = len(data)
    
    average = np.zeros((3, nlayers))
    deviation = np.zeros((3, nlayers))
    
    print("Using centre obtained from Gaussian fit to red channel.") # See "dist" variable
    
    if xoffset is not None:
        centre[0,:,:] += xoffset
    if yoffset is not None:
        centre[1,:,:] += yoffset
    
    for nlayer in range(nlayers):
        layer_data = data[nlayer]
        
        nx = layer_data.shape[1]
        ny = layer_data.shape[0]
        
        mask = np.zeros_like(layer_data[:,:,0])
        
        xx, yy = np.meshgrid(np.arange(nx),np.arange(ny))
        
        # The variable "fixed" controls whether or not the same centre position is used for all layers
        # If fixed == False, we check for each layer that the circle will be contained within the RCF
        if fixed:
            print("Using fixed centre position from layer {}".format(fixed))
            n = fixed-1
        else:
            n = nlayer
        
        if semicircle is None:
            if  (
                    ((nx - radius) < centre[0,0,n]) or # Check if centre is far enough from max nx
                    ((ny - radius) < centre[1,0,n]) or # Check if centre is far enough from max ny
                    (radius > centre[0,0,n]) or # Check if far enough from miniumum nx
                    (radius > centre[1,0,n]) # Check if far enough from minimum ny
                ):
                
                centre[0,0,n] = int(nx/2)
                centre[1,0,n] = int(ny/2)
                
                print("Error with Gaussian fit for layer {}, mask is not located within RCF boundary.".format(nlayer))
            
        elif semicircle < 0: # Bottom half of semicircle
            if  (
                    ((nx - radius) < centre[0,0,n]) or # Check if centre is far enough from max nx
                    ((ny - radius) < centre[1,0,n]) or # Check if centre is far enough from max ny
                    (radius > centre[0,0,n]) # Check if far enough from miniumum nx
                ):
                
                centre[0,0,n] = int(nx/2)
                centre[1,0,n] = int(ny/2)
                
                print("Error with Gaussian fit for layer {}, mask is not located within RCF boundary.".format(nlayer))
                
        dist = np.sqrt((xx-centre[0,0,n])**2+(yy-centre[1,0,n])**2) # Using red as this channel has strongest response
        distx = xx-centre[0,0,n]
        disty = yy-centre[1,0,n]
        
        # Create a mask that we mutliple with the data to remove points outside the circle radius
        mask[np.where(dist<radius)]=1
        if semicircle is not None:
            if semicircle < 0:
                mask[np.where(disty<0)]=0
        
        if test:
            fig, ax = pm.plot_figure_axis("small",4)
            ax[0].imshow(dist)
            ax[1].imshow(distx)
            ax[2].imshow(disty)
            ax[3].imshow(mask)
        
        nmask = np.sum(mask)
        
        for colour in range(3):
            mask_data = layer_data[:,:,colour]*mask # can also just use np.where instead here of making a mask
            # i.e. dose_data[:,:,colour][np.where(dist<radius)]
            
            average[colour, nlayer] = np.sum(mask_data)/nmask
            
            variance = np.sum(np.square(layer_data[:,:,colour][np.where(dist<radius)] - average[colour,nlayer]))/nmask
            
            deviation[colour, nlayer] = np.sqrt(variance)
            
            if plot == True:
                if colour == 0:      
                    fig, ax = pm.plot_figure_axis("small", 2)
                    ax[0].imshow(layer_data[:,:,0], cmap="hsv")
                    ax[0].scatter(centre[0,0,nlayer],centre[1,0,nlayer], color="k", marker="x", s=200)
                    ax[0].set_ylabel("y (px)")
                    ax[0].set_xlabel("x (px)")
                    ax[1].imshow(mask_data, cmap="hsv")
                    ax[1].set_ylabel("y (px)")
                    ax[1].set_xlabel("x (px)")
            
                    fig.tight_layout()
    
    return average, deviation
    

def calc_calibration_curve(project, scanner, layer, doses, radius=50 , 
                           crop="V2", test=None, null=False, plot=False):
    '''
    Generate calibration curves from RCF calibration data.
    Process is: load data, crop data, perform gaussian fitting to get centre
    and width of signal, then calculate average value in circle/semicircle
    region.
    '''
    
    # Some settings for running e
    if test is not None:
        doses = np.array([test])
        
    if null:
        doses = np.array([0])
    
    # Load RCF calibration data
    data = []
    
    for dose in doses:
        data.append(get_radiography_data(project, str(dose), layer, scanner=scanner))
            
    # Crop data - same for all data, this is quite zealous to remove all significant noise.
    data_crop = crop_calibration_data(data, crop=crop, scanner=scanner, plot=plot)

    # Gaussian fitting - plot for full testing as makes many figures
    centre, sigma = gaussian_fit(data_crop, plot=False, test=False)
    
    # Calculate average
    average, deviation = calc_average_circle(data_crop, centre, radius, plot=plot)
    
    return average, deviation


def get_calibration_curve_2021(project, scanner, layer, radius=50 , crop="V2",
                               test=None, null=False, plot=False):
    '''Generate the calibration curves for the 2021 data'''
    
    # Dose values for each layer
    dose_gy = np.array([0,0.09,0.21,0.53,1.03,2.05,5.07,9.53,24.04,49.42,67.03,
                        100.81,236.46,396.13,490.08,731.30,985.71,1510.89,
                        1765.57,2018.92,2276.23,3023.10,4983.96,7485.84,
                        10008.83])
    
    # Note of which layer is which material HDV2 received more doses
    if layer[1] in ["1","2","3"]:
        material = "HDV2"
        
    elif layer[1] in ["4","5"]:
        material = "EBT3"
        
    print("The layer material is {}.".format(material))
    
    # Set number of doses that were obtained and remove missing doses
    if material == "HDV2":
        doses = np.arange(0, 25)
        # doses = np.arange(0,12) # First half of valid HDV2 layers
        # doses = np.arange(12,25) # Second half of valid HDV2 layers
        
        doses = np.delete(doses, np.argmin(abs(doses-18)))
        
    elif material == "EBT3":
        doses = np.arange(0,15) # All EBT3 layers1
        
        if layer == "L4_D":
            doses = np.delete(doses, 5)
        
    if scanner == "Nikon_CoolScan9000":
        if layer == "L2_B": # Didn't put in the scanner?
            doses = np.delete(doses, 1)
    elif scanner == "Epson_12000XL":
        if layer == "L2_B": # Lost?
            doses = np.delete(doses, 8)
    
    average, deviation = calc_calibration_curve(project, scanner, layer, doses, radius=radius,
                                                crop=crop, test=test, null=null, plot=plot)
    
    if not test:
        plot_calibration_curve(average, deviation, dose_gy, doses=doses, material=material)
    
    return average, deviation


# %% Plotting functions

def plot_calibration_curve(average, deviation, dose_gy, doses=None, material=None,
                           OD=False, average_fit=None, deviation_fit=None, dose_fit=None,
                           colour="X"):
    '''Plot calibration curves with standard deviation and null value'''
    
    fig, ax = pm.plot_figure_axis("small", 1)
    
    colours = ["r","g","b"]
    
    if average.ndim == 2:
        
        if doses is None:
            raise Exception("Doses must be input.")
        
        for colour in range(3):
            ax.errorbar(dose_gy[doses[1:]], average[colour,1:], 
                        yerr=deviation[colour,1:], color=colours[colour], 
                        label=material+" {} calibration".format(colours[colour].upper()))
            
            ax.plot([min(dose_gy[doses[1:]]),max(dose_gy[doses[1:]])],[average[colour,0],average[colour,0]],
                    color=colours[colour], linestyle="--", label=material+" {} null".format(colours[colour].upper()))
            
            ax.fill_between([min(dose_gy[doses[1:]]),max(dose_gy[doses[1:]])], 
                            [average[colour,0]-deviation[colour,0],average[colour,0]-deviation[colour,0]],
                            [average[colour,0]+deviation[colour,0],average[colour,0]+deviation[colour,0]], 
                            color=colours[colour], alpha=0.125)
    
    elif average.ndim == 1:
        ax.errorbar(dose_gy, average, yerr=deviation, label="{} Channel".format(colour))
        if average_fit is not None:
            if dose_fit is None:
                dose_fit = dose_gy
            ax.plot(dose_fit, average_fit, linestyle="--", label="{} Channel fit".format(colour), color="r")
            ax.fill_between(dose_fit, average_fit+deviation_fit, average_fit-deviation_fit, color="r", alpha=0.125)
    
    if not OD:
        ax.set_ylabel(r"ave. Pixel (16 bit)")
        ax.legend(loc="lower left")
        ax.set_yscale("log")
    else:
        ax.set_ylabel("OD")
        ax.legend(loc="lower right")
        # ax.set_xlim(xmin=1e-1,xmax=1e3)
        # ax.set_ylim(ymin=0,ymax=2.5)
    ax.set_xlabel(r"Dose (Gy)")
    ax.set_xscale("log")
    
    fig.tight_layout()

    return

# %% Main script
if __name__ == "__main__":
    
    project = "Calibration"
    # scanner = "Nikon_CoolScan9000"
    scanner = "Epson_12000XL"
    
    if 0: # Generate calibration curves
        '# L1-3 = HDV2, L4-5 = EBT3, we use 3 and 5'
        # Uncomment line to obtain calibration curve.
        # layer = "L1_A"
        # layer = "L2_B"
        # layer = "L3_C"
        # layer = "L4_D"
        layer = "L5_E"
        # layers = ["A","B","C","D","E"] # Not sure this works anymore?
        
        # Some early thoughts...
        # For HDV2 layers 1-4 show no significant change in any channel
            # -
            # - Green channel layers 5-6 also show no significant change
            # - Blue channel layers 5-7 also show no significant change
        
        # For EBT3:
            # - Red channel all doses are fine
            # - Green channel layer 1 has no significant change
            # - Blue channel layers 1-2 have no significant change
            
        # With further analysis maybe some data could be pulled from earlier layers...
        # However, the pixel/dose curve in this range is not decreasing as it should for the EBT3...
        
        plot = False # For full plotting
        
        test = None # Reduced layers for quick testing
        
        null = False # For obtaining null result
        
        average, deviation = get_calibration_curve_2021(project, scanner, layer, radius=50, 
                                                        crop="V2", test=test, null=null, plot=plot)

plt.show()