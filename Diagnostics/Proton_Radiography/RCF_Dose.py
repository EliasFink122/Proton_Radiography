# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:27:00 2021

@author: Adam Dearling (add525@york.ac.uk)

Contains the following functions for working with RCF data:
    - letter_to_num (converts letter A to 1 etc.)
    - find_bound_points (finds points bounded by curves)
    - clean_data (removes points that lie outside calibration curve by turning to NaNs)
    - nan_helper (for working with NaNs) - legacy from before interp_nan_2D?
    - interp_1D (for interpolating px/OD-dose values)
    - interp_nan_2D (for interpolating over NaNs)
    - convert_dose (converts px/OD to dose)
    - process_data (process for converting pixel value to dose inc. cleaning etc.)
    - combine_dose (combines doses values using errors from multiple channels)
    - get_dose_data (gets raw data and converts to dose)

Also has some little functions for plotting.
    
Future improvements:
    - Swap from lists to tuples?...
    - Move away from using lists entirely?... (no real time save and looks messy?)
    - Might make more sense to run cleaning/interpolation phases separetly from dose phase.
        
When interpolating data, interpolation should be done when the data is most linear
if using a linear interpolation method. For calibration curves it isbetter to 
interpolate OD than px as it varies more linearly with dose, although logOD vs 
logDose is even more linear. When interpolating over NaNs in the image we have
to make an assumption about the underlying function to interpolate (as depending
on the field structure fluxuation in dose may not be linear). In the limit that
the points we are interpolating are very close together we can assume the function
is linear (Taylor series).
"""

# %% Libraries

import sys
import time
import numpy as np
import scipy.interpolate as interp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.path as path

import RCF_Basic as rcf
import RCF_Image_Crop as ic
from RCF_Calibration_Curves import get_calibration, make_calibration_range_same
from RCF_Calibration_Curves import convert_intensity_OD_RGB

sys.path.insert(1, '../../Codes/Python_Scripts/')
import Plot_Master as pm

rootdir = r"/Users/eliasfink/Desktop/Proton_Radiography"


# %% Functions

def find_bound_points(data_x, data_y, x, x_dev, y, y_dev, sigma, active=True):
    '''
    Find points bounded by sigma times the deviation from some curve

    Returns points within those bounds, as well as the paths of the boundaries.
    '''

    ny, nx = data_x.shape # Data arrays should have the same dimensions
    assert (data_x.shape == data_y.shape and 
            len(x) == len(y)), "Data arrays must have the same shape!!!"

    if active:
        # Find points in XY plane bounded by X deviation
        XY_pathX = np.concatenate((np.array([x-sigma*x_dev, y]),
                                   np.array([x+sigma*x_dev, y])[:,::-1]), axis=1)
        XY_pathX = XY_pathX.T

        XY_boundX = path.Path(XY_pathX)
        XY_pointsX = XY_boundX.contains_points(np.array([data_x.flatten(), data_y.flatten()]).T)

        XY_pointsX = XY_pointsX.reshape(ny,nx)

        # Find points in XY plane bounded by Y deviation
        XY_pathY = np.concatenate((np.array([x, y-sigma*y_dev]),
                                   np.array([x, y+sigma*y_dev])[:,::-1]), axis=1)
        XY_pathY = XY_pathY.T

        XY_boundY = path.Path(XY_pathY)
        XY_pointsY = XY_boundY.contains_points(np.array([data_x.flatten(), data_y.flatten()]).T)

        XY_pointsY = XY_pointsY.reshape(ny,nx)

        XY_points = XY_pointsX * XY_pointsY # Combine

    else:
        XY_points = np.ones(data_x.shape, dtype=bool) 
        XY_pathX = XY_pathY = None

    return XY_points, [XY_pathX, XY_pathY]


def clean_array(data, project, shot, layer, channels=[1,1,1], OD=False, 
                scanner=None, material_type=None, sigma=5, plot=False):
    ''' 
    Used to remove scratches/dust from RCF data
    Does this by keeping only data within n sigma of the calibraiton curve.
    Use the full calibration range - while we may be unsure about the dose,
    the pixel values are still correct. 
    
    channel: if channel is selected (=1) it will be included in the clean 
        function, order is ["RG","RB","GB"].
    '''
    print("Extracting all data within {sigma} sigma of calibration curve.")

    # Load stack composition to get correct calibration curves
    stack_composition = rcf.get_stack_design(project, shot, info="material")

    nlayer = rcf.letter_to_num(layer) 

    layer_material = stack_composition[nlayer-1] #Subtract 1 because letter 1 is A, but array start is 0

    R_dose, R_val, R_dev = get_calibration(layer_material, 0, scanner=scanner,
                                           material_type=material_type, OD=OD,
                                           calibration="full", plot=False)
    G_dose, G_val, G_dev = get_calibration(layer_material, 1, scanner=scanner,
                                           material_type=material_type, OD=OD,
                                           calibration="full", plot=False)
    B_dose, B_val, B_dev = get_calibration(layer_material, 2, scanner=scanner,
                                           material_type=material_type, OD=OD,
                                           calibration="full", plot=False)

    # Separate colour channels
    data_R = data[:,:,0]
    data_G = data[:,:,1]
    data_B = data[:,:,2]

    # Fix so that when "valid" calibration is used all calibration pixel values
    # and deviation arrays are the same size/have the same dose values.
    # This means losing some data... not sure what can be done about this for now.
    # Either arrays have to be the same size (so every dose has a corresponding
    # pixel in all 3 colours), or I need to extrapolate?
    val, dev = make_calibration_range_same([R_dose, G_dose, B_dose],
                                           [R_val, G_val, B_val],
                                           [R_dev, G_dev, B_dev],
                                           OD, plot=True)

    (R_val, G_val, B_val) = val
    (R_dev, G_dev, B_dev) = dev

    # Find coordinates of points within each pair of calibration curves
    RG_points, RG_path = find_bound_points(data_R, data_G, R_val, R_dev,
                                           G_val, G_dev, sigma, channels[0])
    RB_points, RB_path = find_bound_points(data_R, data_B, R_val, R_dev,
                                           B_val, B_dev, sigma, channels[1])
    GB_points, GB_path = find_bound_points(data_G, data_B, G_val, G_dev,
                                           B_val, B_dev, sigma, channels[2])

    points = RG_points * RB_points * GB_points # Combine

    if plot:
        plot_clean(data, points, R_val, R_dev, G_val, G_dev,
                   B_val, B_dev, sigma, RG_path, RB_path, GB_path,
                   RG_points, RB_points, GB_points, OD=OD)
    # sys.exit()
    return points


# def nan_helper(y):
#     """Helper to handle indices and logical indices of NaNs.

#     Input:
#         - y, 1d numpy array with possible NaNs (must be 1d)
#     Output:
#         - nans, logical indices of NaNs (1d boolean array where True=nan)
#         - index, a function, with signature indices= index(logical_indices),
#           to convert logical indices of NaNs to 'equivalent' indices (a function which returns positions of nonzero elements (i.e. True))
#     Example:
#         >>> # linear interpolation of NaNs
#         >>> nans, x= nan_helper(y)
#         >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
#     """

#     return np.isnan(y), lambda z: z.nonzero()[0]


def interp_1D(x, xp, yp, OD, left=None, right=None, mode=None):
    '''
    Linearly interpolate data
    
    mode: format the x/xp and yp data using a given function before interpolating
        in order to make the data more suitable for linear interpolation.
    OD: for reasons I don't understand, np.interp does not work with px values but
        scipy.interp1d does...
    '''

    if mode == "log-log":
        x = np.log10(x)
        xp = np.log10(xp)
        yp = np.log10(yp)

    if OD:
        y = np.interp(x, xp, yp, left=left, right=right)
    else:
        y_interp = interp1d(xp, yp, bounds_error=False)
        y = y_interp(x)

    if mode == "log-log":
        y = 10**y

    return y


def interp_nan_2D(data, input_valid=None, interpolator="nearest", plot=False):
    '''
    Interpolate over NaNs in an RGB array.
    
    input_valid: gives the positions of NaNs where False, otherwise this is 
        found for each layer if None.
    '''
    # input_valid=None
    data_interp = np.copy(data) # Create a copy of the array

    ny, nx = data.shape[0:2]

    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

    # If positions are provided
    if input_valid is not None:
        array_valid = input_valid
        array_nan = ~array_valid # Array where True if NaN

        # Flatten xx and yy arrays and combine
        xym = np.vstack( (np.ravel(xx[array_valid]), np.ravel(yy[array_valid])) ).T
        xym_nan = np.vstack( (np.ravel(xx[array_nan]), np.ravel(yy[array_nan])) ).T

    # Interpolate if there are NaNs
    for c in range(3): # Loop through colours

        if np.count_nonzero(np.isnan(data[:,:,c])) > 0:

            # Find positions to interp for each layer if postions aren't provided
            if input_valid is None: # "|" is bitwise "or" and "~" is bitwise "not"
                array_nan = np.isnan(data[:,:,c])
                array_valid = ~array_nan

                xym = np.vstack( (np.ravel(xx[array_valid]), np.ravel(yy[array_valid])) ).T
                xym_nan = np.vstack( (np.ravel(xx[array_nan]), np.ravel(yy[array_nan])) ).T

            # Check how many NaNs exist
            nan_frac = array_nan.sum()/(nx*ny)
            print(f"C{c+1}: {nan_frac*100:.3f} % NaN.")
            if nan_frac > 0.01:
                print("Caution: Image has more than 1% NaN values.")

            # The valid values in the first, second, third color channel
            # as 1D arrays (in the same order as their coordinates in xym)
            array_c_valid = np.ravel(data[:,:,c][array_valid])

            # Three separate interpolators for the separate color channels
            if interpolator == "linear":   
                array_c_interp = interp.LinearNDInterpolator(xym, array_c_valid)
            elif interpolator  == "nearest":
                array_c_interp = interp.NearestNDInterpolator(xym, array_c_valid)
                print("NaN interpolator set to nearest for speed.")
            # Example interpolate for whole image for one color channel
            # data_interp[:,:,c] = array_c_interp(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

            # Interpolate only nans, one color channel at a time 
            data_interp[:,:,c][array_nan] = array_c_interp(xym_nan[:,0], xym_nan[:,1])

    if plot:
        rcf.plot_comparison([data, data_interp], title_list=["Pre-interp", "Post-interp"],
                            colour=2)

    if 0: assert np.sum(np.isnan(data_interp))==0, "NaNs still remaining after interpolation."

    return data_interp


def convert_dose(data, material, OD=False, scanner=None, material_type=None, 
                 fill=np.NaN, interp_mode="log-log", plot=False, test=False):
    '''
    Convert pixel/OD to dose for and RGB array
    
    Using the calibration curve, to generate a fitting function as it gets
    a better fit. This used to use the "valid" calibration curve, but using
    combining the doses from the three channels later on using a weighted error
    means it should be preferential to use all the data here.
    
    fill: by default upper and lower limits are set to NaN and then interpolated
        over. This makes it obvious when displaying data later if there is a large 
        region for which the calibration curve is not valid (which would be bad). 
        There appears to be little difference between interpolating NaNs or using 
        the min/max value to fill.
    interp_mode: if None performs interpolation in px/OD-dose space, otherwise if
        "log-log" it is performed in log10(px/OD)-log10(dose) space where data is 
        usually more linear.
    '''

    data_dose = np.zeros(data.shape)
    data_dose_err = np.zeros(data.shape)
    if test:
        data_dose_sub_dev = np.zeros(data.shape)
        data_dose_add_dev = np.zeros(data.shape)

    for colour in range(3):
        dose, ave_val, ave_dev = get_calibration(material, colour, scanner=scanner,
                                                 material_type=material_type, OD=OD,
                                                 calibration="full", plot=False)

        # # Set fill value for lower/upper bounds
        # if fill is None:
        #     fill_lower = np.NaN
        #     fill_upper = np.NaN
        if fill is None: # Was previously "min_max"
            # fill_lower = None
            # fill_upper = None
            raise ValueError("Warning: This will result in errors equal to zero.")
            # Need to either multiply lower/upper min/max values of dose by some number
            # or interpolate over zeros at a later state?

        # Convert pixel to dose using 1D interpolation
        data_dose[:,:,colour] = interp_1D(data[:,:,colour], ave_val, dose, OD,
                                          left=fill, right=fill, mode=interp_mode)

        # Calculate errors using upper and lower dose given by standard deviation.
        # Whether this is a maximum or minimum dose depends on whether OD or px is used.
        # For OD subtracting leads to lower dose, for px subtracting leads to higher dose.
        data_dose_err[:,:,colour] = np.absolute(interp_1D(data[:,:,colour], ave_val-ave_dev, dose,
                                                          OD, left=fill, right=fill,
                                                          mode=interp_mode)
                                                - interp_1D(data[:,:,colour], ave_val+ave_dev,
                                                            dose, OD, left=fill, right=fill,
                                                            mode=interp_mode))/2

        if test:
            data_dose_sub_dev[:,:,colour] = interp_1D(data[:,:,colour], ave_val-ave_dev, dose, OD,
                                                      left=fill, right=fill,
                                                      mode=interp_mode)
            data_dose_add_dev[:,:,colour] = interp_1D(data[:,:,colour], ave_val+ave_dev, dose, OD,
                                                      left=fill, right=fill,
                                                      mode=interp_mode)

    # Plot x/y average and errors for the three channels
    if test:
        fig, ax = pm.plot_figure_axis("small", 6, shape=2, ratio=[1,0.66])
        x = np.arange(data.shape[0])
        y = np.arange(data.shape[1])
        xy = [x,y]
        for c in range(3):
            for i in [0,1]: # x/yy axis averages
                ax[c+i*3].plot(xy[i], np.average(data_dose[:,:,c], axis=1-i),
                               label=r"$\langle D \rangle$")
                ax[c+i*3].fill_between(xy[i], np.average(data_dose[:,:,c], axis=1-i)-np.average(data_dose_err[:,:,c], axis=1-i),
                                   np.average(data_dose[:,:,c], axis=1-i)+np.average(data_dose_err[:,:,c], axis=1-i),
                                   alpha=0.25)
                ax[c+i*3].plot(xy[i], np.average(data_dose_sub_dev[:,:,c], axis=1-i),
                               label=r"$\langle D - \sigma \rangle$")
                ax[c+i*3].plot(xy[i], np.average(data_dose_add_dev[:,:,c], axis=1-i),
                               label=r"$\langle D + \sigma \rangle$")
        ax[0].set_ylabel("Dose (Gy)")
        ax[3].set_ylabel("Dose (Gy)")
        ax[2].legend()
        fig.suptitle(r"RCF Dose $\langle x \rangle$ RGB")
        fig.tight_layout()

    if plot:
        rcf.plot_data(data_dose, title="RCF Dose RGB")
        # rcf.plot_data(data_dose_err, title="RCF Errors RGB")

    return data_dose, data_dose_err


def dose_data(project, data, shot, layers, OD=False, scanner=None,
              material_type=None, clean=True, sigma=5,
              clean_chan=[1,1,1], plot=False, cout=None):
    '''
    Function for converting pixel value to proton dose in Gy
    
    
    calibration: "valid" will only use what I have deemed to be valid on the
        curve, otherwise all values are used.
    scanner: must be set based on the RCF that was used.
    clean: if true any value lying outside of sigma times the standard dev
        for the calibration curve is turned to zero.
    clean_chan: list corresponding to ["RG","RB","GB"], if 1 then this channel
        combination will be used in the cleaning phase, otherwise it won't be.
    '''

    # Get stack information
    stack_composition = rcf.get_stack_design(project, shot, info="material")

    # Lists for storing processed stack
    data_dose_raw = []
    data_dose_clean = []
    # data_dose_err_clean = []
    data_dose = []
    data_dose_err = []

    # Iterate through layers
    for n, layer in enumerate(layers):
        nlayer = rcf.letter_to_num(layer)
        layer_material = stack_composition[nlayer-1] # A -> 1 but array 0-index
        print("Layer " + str(nlayer) + " material is " + layer_material)

        layer_RGB = data[n] # First layer
        ny, nx = layer_RGB.shape[0:2] # Dimensions of image

        # Convert intensity to OD if set
        if OD:
            # Get null data - should be setup so we can just take average
            null = rcf.get_radiography_data(project, "Null", layer_material)
            null_ave = np.mean(null, axis=(0,1))
            print(null_ave)
            # null_ave[0] = null_ave[0]-10500
            # null_ave[1] = null_ave[1]-8000
            # null_ave[2] = null_ave[2]-2000
            # print(null_ave)
            null_std = np.std(null, axis=(0,1))
            layer_RGB, error_RGB = convert_intensity_OD_RGB(layer_RGB, 0, null_ave, null_std)
        else:
            error_RBG = np.zeros(layer_RGB.shape) # This should probably feed into convert_dose function

        # Setup to clean RCF layer of dust/scratches etc.
        if clean: # Boolean array containing positions of "real" data (=True where real)
            print("Cleaning radiographs.")
            layer_clean = clean_array(layer_RGB, project, shot, layer, channels=clean_chan,
                                      OD=OD, scanner=scanner, material_type=material_type,
                                      sigma=sigma, plot=plot)
        else:
            layer_clean = np.ones((ny,nx))

        # Convert pixel value to dose
        print("Dosing radiographs.")
        layer_dose, layer_dose_err = convert_dose(layer_RGB, layer_material,
                                                  OD=OD, scanner=scanner,
                                                  material_type=material_type)

        # Apply cleaning array (interp next) - this makes invalid values 0
        layer_dose_clean = layer_dose * layer_clean[:,:,np.newaxis]
        layer_dose_err_clean = layer_dose_err * layer_clean[:,:,np.newaxis]

        # Store raw/cleaned data
        data_dose_raw.append(layer_dose)
        data_dose_clean.append(layer_dose_clean)
        # data_dose_err_clean.append(layer_dose_err_clean)
        if plot:
            rcf.plot_data(data_dose_clean[-1], colour=cout,
                          title=f"RCF Clean RGB (Clean={str(clean)})")

        # Interpolate/inpaint to remove NaNs - used to be an if statement, with
        # NaN values turned to zero otherwise so there was no interpolation.
        print("Interpolating over NaNs.")
        layer_dose_nan = np.copy(layer_dose_clean) # Creates a copy of the array (new address)
        layer_dose_err_nan = np.copy(layer_dose_err_clean)
        layer_dose_nan[layer_dose_clean==0] = np.nan
        layer_dose_err_nan[layer_dose_err_clean==0] = np.nan

        t0 = time.time()

        # Originally there was a seperate method for the cleaning treatment which
        # used the clean array as an input. However, this does not include NaNs that
        # arise in dose error calculation!
        # if clean:
        #     layer_dose_interp = interp_nan_2D(layer_dose_nan, layer_clean, plot=plot)
        #     layer_dose_err_interp = interp_nan_2D(layer_dose_err_nan, layer_clean, plot=False)

        # If interpolating NaNs but not cleaning. Orignal method returned
        # False wherever a NaN occurs in any layer, but this applied to all
        # channels (which I don't want if the calibration curve fit is bad.
        layer_dose_interp = interp_nan_2D(layer_dose_nan, None, plot=plot)
        layer_dose_err_interp = interp_nan_2D(layer_dose_err_nan, None, plot=False)

        # Flag in case interpolation causes values to exceed maximum possible range.
        # This was used where the array was cleaned before dosing.
        # if ((OD and (((np.amax(layer_RGB_interp, axis=(0,1)) - np.log10(null_ave/1))>0).any() or
        #              ((np.amin(layer_RGB_interp, axis=(0,1)) - np.log10(null_ave/2**16))<0).any()))
        #     or (not OD and (np.amin(layer_RGB_interp) < 1 or np.amax(layer_RGB_interp) > 2**16))):
        #     raise Exception("Interpolation has caused values to exceed maximum range.")

        t1 = time.time()

        print(f"The elapsed interpolating time for this layer was {t1-t0:.2f}.")

        data_dose.append(layer_dose_interp)
        data_dose_err.append(layer_dose_err_interp)

    return data_dose, data_dose_err


def combine_dose(dose, error, channels=[1,1,1], plot=True):
    '''
    Combines colour channels for a piece of RCF data.
    
    channels: a list corresponding to the colour channels ["R","G","B"] that
        should be combined to form the final piece of data, weighted by their
        errors.
    '''

    layers = len(dose)

    # List for storing data
    dose_combined = []
    error_combined = []

    # Indexes of selected colour channels
    channel_index = []
    for c in range(3):
        if channels[c]:
            channel_index.append(c)

    for layer in range(layers):
        if np.amin(error[layer]) == 0:
            raise RuntimeError("Error in dose should not be zero. Exiting.")

        # error = 1 / sqrt(sum(errors^-2))
        layer_error = error[layer][:,:,channel_index] # Slice out layers which are to be merged
        layer_error = np.power(layer_error, -2)
        layer_error = np.sum(layer_error, axis=2)
        layer_error = np.power(layer_error, -0.5)


        # dose = sum(doses/errors^2)*error^2
        layer_dose = dose[layer][:,:,channel_index]
        layer_dose = layer_dose / np.power(error[layer][:,:,channel_index], 2)
        layer_dose = np.sum(layer_dose, axis=2)
        layer_dose = layer_dose * np.power(layer_error, 2)

        dose_combined.append(layer_dose)
        error_combined.append(layer_error)

    return dose_combined, error_combined


def get_dose_data(project, shot, stack, layers, suffix=None, edge=0, shape="square",
                  OD=False, scanner=None, material_type=None, clean=True, sigma=5,
                  clean_chan=[1,1,1], channels=[1,1,1], plot=False, cout=None,
                  plot_output=True):
    ''' 
    Gets the raw image and converts to dose.
    
    channels: a list corresponding to the colour channels ["R","G","B"] that
        should be combined to form the final piece of data, weighted by their
        errors.
    '''

    print('Reading radiograph data.')

    data = rcf.get_stack_data(project, shot, stack, layers, suffix=suffix,
                              flipx=False, flipy=False)

    if plot:
        print("Plotting radiograph data.")
        rcf.plot_data_list(data, title="RCF Raw RGB")

    imgs = ic.crop_rot(rootdir + "/Projects/" + project + "/Experiment_Data/Proton_Radiography/Shot001/sh1_st1.tif")

    data_dose_rc, data_error_rc = dose_data(project, imgs, shot, layers, OD=OD, scanner=scanner,
                                        material_type=material_type, clean=clean, sigma=sigma,
                                        clean_chan=clean_chan, plot=plot, cout=cout)

    # data_dose, data_dose_err = dose_data(project, data, shot, layers, OD=OD, scanner=scanner,
    #                                      material_type=material_type, clean=clean, sigma=sigma,
    #                                      clean_chan=clean_chan, plot=plot, cout=cout)

    # Crop and rotate after dosing is done. Rotation before imprints a grid into
    # the data, whic is most visible during the cleaing phase. Rotating after
    # cleaning (and dosing) overcomes this issue. However, as the data has not
    # yet been cropped there is some risk that during the interpolation stage
    # artifacts are introduced.
    # data_dose_rc = rcf.rotate_crop_data(data_dose, project, shot, layers,
    #                                     edge=edge, shape=shape, rot=True, plot=False)
    # data_error_rc = rcf.rotate_crop_data(data_dose_err, project, shot, layers,
    #                                      edge=edge, shape=shape, rot=True, plot=False)

    # Combine colour channels to maximise dose range.
    data_dose_comb, data_error_comb = combine_dose(data_dose_rc, data_error_rc,
                                                   channels=channels)

    if plot_output: # Compare results before and after
        # rcf.plot_comparison([data, data_dose_rc], average=False, colour=2,
        #                     title_list=["Before Dose", "After Dose"]) # Image comparison
        rcf.plot_data_list(data_dose_comb, colour=4, cbar_label=r"Dose (kGy)",
                           cmap="viridis", logc=False, save=False)#, title="Combined Dose") # Single image

    return data_dose_comb, data_error_comb



# %% Plotting functions

def plot_clean(data, points, R_val, R_dev, G_val, G_dev, B_val, B_dev, sigma,
               RG_path, RB_path, GB_path, RG_points, RB_points, GB_points,
               OD=False, dpi=300, scale="log", norm=True):
    '''
    Plots the image RGB space with the calibration curves.
    Making the crop region smaller (focussing on dose region) can help confirm
    origin of a signal.
    '''

    import matplotlib.colors

    colours = ["Red", "Green", "Blue"]

    # Separate colour channels
    data_R = data[:,:,0]
    data_G = data[:,:,1]
    data_B = data[:,:,2]

    # if not OD and norm:
    #     data_R = data_R/2**16
    #     data_G = data_G/2**16
    #     data_B = data_B/2**16
    #     RG_path = RG_path/2**16
    #     RB_path = RB_path/2**16
    #     GB_path = GB_path/2**16

    ny, nx = data_R.shape[:2] # All channels have the same dimensions
    size = nx*ny
    # size = 1500**2

    np.random.seed(seed=1)
    random_ind = np.random.randint(0, high=size, size=100000, dtype=int)

    # Take random values from all data
    R_exp = np.take((data_R).flatten(), random_ind)
    G_exp = np.take((data_G).flatten(), random_ind)
    B_exp = np.take((data_B).flatten(), random_ind)

    # Take random values from signal
    # R_sig = np.take((data_R*points).flatten(), random_ind)

    if RG_path[0] is not None:
        RG_pathR = RG_path[0].T
        RG_pathG = RG_path[1].T
    if RB_path[0] is not None:
        RB_pathR = RB_path[0].T
        RB_pathB = RB_path[1].T
    if GB_path[0] is not None:
        GB_pathG = GB_path[0].T
        GB_pathB = GB_path[1].T

    fig, ax = pm.plot_figure_axis(25, 3)
    # fig.suptitle("RCF Data (sigma = {})".format(sigma))

    # patch = patches.PathPatch(RG_boundR, facecolor='orange', lw=2)
    # ax_clean.add_patch(patch)

    if OD:
        bins = np.linspace(0, 2.5, 200)
    else:
        if scale == "linear":
            bins = np.linspace(1, 2**16, 200)
        elif scale == "log":
            bins = np.logspace(np.log10(2**10), np.log10(2**16-2000), 200)

    ax[0].hist2d(R_exp, G_exp, bins=bins, norm=matplotlib.colors.LogNorm())
    # ax[0].hist2d(R_sig, G_sig, bins=bins, norm=matplotlib.colors.LogNorm())
    # ax[0].plot(R_val, G_val, color="k", linestyle="--")
    # ax[0].errorbar(R_val, G_val, xerr=sigma*R_dev, yerr=sigma*G_dev, color="k", linestyle="")
    if RG_path[0] is not None:
        ax[0].plot(RG_pathR[0,:], RG_pathR[1,:], color="r")
        ax[0].plot(RG_pathG[0,:], RG_pathG[1,:], color="g")
    # ax[0].set_title("R vs G")

    ax[1].hist2d(R_exp, B_exp, bins=bins, norm=matplotlib.colors.LogNorm())
    if RB_path[0] is not None:
        ax[1].plot(RB_pathR[0,:], RB_pathR[1,:], color="r")
        ax[1].plot(RB_pathB[0,:], RB_pathB[1,:], color="b")

    ax[2].hist2d(G_exp, B_exp, bins=bins, norm=matplotlib.colors.LogNorm())
    if GB_path[0] is not None:
        ax[2].plot(GB_pathG[0,:], GB_pathG[1,:], color="g")
        ax[2].plot(GB_pathB[0,:], GB_pathB[1,:], color="b")

    # Setup colour bar
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.cm import ScalarMappable
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(ScalarMappable(), cax=cax)
    # if cbar_label is not None:
    #     cbar.ax.get_yaxis().labelpad = 30
    #     cbar.ax.set_ylabel(cbar_label, rotation=270)

    n = 0
    for axi in ax:
        axi.tick_params(which="both", direction='in', length=8)
        if OD:
            axi.set_xlabel("OD")
            axi.set_xlim(xmin=0, xmax=2.5)
            axi.set_ylim(ymin=0, ymax=2.5)
        else:
            axi.set_xlabel(f"{colours[n]} channel (16-bit)")
        if not OD and scale == "log":
            axi.set_xscale("log")
            axi.set_yscale("log")
        if axi != ax[0]:
            axi.set_yticklabels([])
        n+=1
    if OD:
        ax[0].set_ylabel("OD")
    else:
        ax[0].set_ylabel(f"{colours[1]} channel (16-bit)")
    fig.subplots_adjust(wspace=0, hspace=0)

    fig.tight_layout()

    if 1:
        fig_clean, ax_clean = pm.plot_figure_axis("small", 4, shape=2, ratio=[1,(ny/nx)])
        ax_clean[0].imshow(points)
        ax_clean[0].set_title("Combined Clean Array")
        ax_clean[1].imshow(RG_points)
        ax_clean[1].set_title("RG Clean Array")
        ax_clean[2].imshow(RB_points)
        ax_clean[2].set_title("RB Clean Array")
        ax_clean[3].imshow(GB_points)
        ax_clean[3].set_title("GB Clean Array")
        fig_clean.tight_layout()
    else:
        fig_clean, ax_clean = pm.plot_figure_axis("small", 1, ratio=[1,(ny/nx)])
        res = rcf.convert_dpi_to_m(300)
        im_extent = [0, nx*res, 0, ny*res]
        ax_clean.imshow(points, extent=im_extent)
        ax_clean.set_xlabel("x (mm)")
        ax_clean.set_ylabel("y (mm)")
        fig_clean.tight_layout()


    if 0:
        fig_res, ax_res = pm.plot_figure_axis("small", 2, [[2,3]], ratio=[4/7, 1])

        ax_res[0].hist2d(G_exp, B_exp, bins=bins, norm=matplotlib.colors.LogNorm())
        if GB_path[0] is not None:
            ax_res[0].plot(GB_pathG[0,:], GB_pathG[1,:], color="g")
            ax_res[0].plot(GB_pathB[0,:], GB_pathB[1,:], color="b")
            ax_res[0].set_xlabel("Green channel (16-bit)")
            ax_res[0].set_ylabel("Blue channel (16-bit)")
            if not OD and scale == "log":
                ax_res[0].set_xscale("log")
                ax_res[0].set_yscale("log")

        res = rcf.convert_dpi_to_m(300)
        im_extent = [0, nx*res, 0, ny*res]
        ax_res[1].imshow(points, extent=im_extent)
        ax_res[1].set_xlabel("x (mm)")
        ax_res[1].set_ylabel("y (mm)")

        fig_res.tight_layout()

    # fig.savefig("temp.png", dpi=300)

    return


# %% Main script

if __name__ == "__main__":

    # Project settings
    project = "Woolsey_2019"
    # project = "Ridgers_2021"
    # project = "Carroll_2020"

    if project == "Woolsey_2019":
        shot = "056"
        stack = "56"
        layers = ["B"]
        suffix = None
        scanner = "Nikon_CoolScan9000"
        edge = [0,0]
        OD = True
        material_type = "0"
        clean = False
        channels = [0,1,0]

    elif project == "Ridgers_2021":
        shot = "048" #1,4,5,6,9 #67,68,69
        stack = "36"
        layers = ["B"]
        suffix = "HDR"
        scanner = "Nikon_CoolScan9000"
        edge = [0,0]
        OD = False
        material_type = None
        clean = True
        channels = [1,1,1]

    elif project  == "Carroll_2020":
        shot = "001"
        stack = "1"
        layers = ["J"]
        suffix = None
        scanner = "Epson_12000XL"
        edge = [60,10]
        OD = False
        material_type = None
        clean = True
        channels = [1,1,1]

    # Convert pixel count to dose
    if 1:
        RCF_dosed_data = get_dose_data(project, shot, stack, layers, suffix=suffix,
                                       edge=edge, shape="rectangle", OD=OD,
                                       scanner=scanner, material_type=material_type,
                                       clean=clean, sigma=5, clean_chan=[1,1,1],
                                       channels=channels, plot=True)

    # Saving data
    if 0:
        rcf.save_radiography_data(RCF_dosed_data, stack, layers, "colours")
