# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 20:43:28 2021

@author: Adam Dearling (add525@york.ac.uk)

Basic functions used for loading and manipulating RCF data.

The basic folder layout used in all of my codes is:
    - Diagnostics
        - Diagnostic (e.g Proton_Radiography)
            - Analysis scripts
            - Calibration data etc.
    - Projects
        - Experiments (e.g Carroll_2020)
            - Experiment_Data
                - Diagnostic (e.g Proton_Radiography)
                    - ShotXXX
                        - Raw Data
                        
To process an RCF file the following CSVs are required in the experiment Diagnostic folder:
    - RCF_Stack_Type
    - RCF_Stack_Design
    - Corner_Positions(_Diagonal)
    
In hindsight, it may have been better to write the function to work with a single
piece of data (instead of a list) and then just loop through the data...
I have started to implement this change now.
"""

# Path settings

# Path to main directory (only needed if code is not being run in the main folder)
# rootdir = r"C:\Users\benny\Desktop\Proton_Radiography_Tool\Diagnostics\Proton_Radiography"


# %% Libraries

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# if os.path.split(os.getcwd())[1] != "Proton_Radiography":
#     sys.path.insert(1, rootdir)
# else:
#     sys.path.insert(1, '../../Codes/Python_Scripts')
import Physics_Constants as pc
import Plot_Master as pm


# %% Functions

def get_radiography_data(project, shot, file, colour=None, location=None,
                         dtype=".tif", scanner=None, plot=False):
    '''
    Load RCF data "file" from dtype file for a given project/shot.
    Generally setup for txt and tif files (havent tried others).
    There are multiple files per shot for different RCF layers
    There are options to load calibration data (different file structure).
    Alpha channel is removed from tifs.
    
    scanner: set scanner/directory that calibration data comes from.
    colour: can load a single colour if set.
    '''

    # Get cwd/check we are in proton radiography directory
    owd = os.getcwd()
    if os.path.split(os.getcwd())[1] != "Proton_Radiography":
        os.chdir(rootdir)

    # Load calibration data
    if project == "Calibration":
        os.chdir(project)

        if scanner is None: # Scanner must be specified
            raise Exception("Scanner not selected. Exiting.")
        else:
            os.chdir(scanner)

        os.chdir("DS" + shot)

    # Load data from experiment directories
    elif location is not None:
        print("Enterting {} directory.".format(location))
        os.chdir("../../Projects/" + project + "/Experiment_Data/Proton_Radiography/" + location)

    else:
        os.chdir("../../Projects/" + project + "/Experiment_Data/Proton_Radiography")

        if shot == "Null":
            os.chdir("Null")
        else:
            os.chdir("Shot" + shot)

    # Check if file is in directory
    if file not in [x.split('.')[0] for x in os.listdir()]:
        print(file)
        print(os.listdir())
        os.chdir(owd)
        raise FileNotFoundError("File is not in directory (Shot={}). Exiting.".format(shot))

    # Load data
    if dtype == ".txt":
        print("Loading data from txt.")
        data = np.loadtxt(file + dtype)

    else:
        print("Loading data from tif.")

        im = cv2.imread(file + dtype, cv2.IMREAD_UNCHANGED)

        data = np.array(im)

        ###########################################################################
        # cv2 READS THE TIF IN BGR INSTEAD OF RGB!!! THE ARRAY MUST BE FLIPPED!!! #
        ###########################################################################

        if data.ndim == 3:
            data = np.flip(data, axis=2)

            if data.shape[2] == 4:
                print("Image includes alpha channel. Removing.")
                data = data[:,:,1:]

    os.chdir(owd)

    if colour is not None:
        if colour == "R":
            c = 0
        elif colour == "G":
            c = 1
        elif colour == "B":
            c = 2

        print("Removing all colour channels except {}.".format(colour))
        data = data[:,:,c]

    if plot:
        plot_data(data, None)
        if 0:
            plot_data_cRGB(data)

    return data


def get_stack_data(project, shot, stack, layers, location=False, 
                   dtype=".tif", suffix="HDR", flipx=False, flipy=False,
                   plot=False):
    '''
    Gets RCF data for an entire stack, putting it in a list.
    
    suffix: file suffix which by default it assumes is "HDR"...
    '''

    data = [] # List for storing data

    # Add underscore to suffix
    if suffix is not None:
        suffix = "_" + suffix
    else:
        suffix = ""

    for layer in layers: # Append data to list
        data.append(get_radiography_data(project, shot, "RCF_"+stack+layer+suffix))
        if flipx:
            data[-1] = data[-1][:,::-1,:]
        if flipy:
            data[-1] = data[-1][::-1,:,:]

    if plot:
        for n in range(len(layers)):
            plot_data(data[n], layers)

    return data


def save_radiography_data(data, stack, layers, project=None, shot=None,
                          info="colours", suffix=None):
    '''
    Save radiography data to a tif by default.
    If data contains floats save to a txt file.
    
    stack: stack number.
    layers: letters corresponding to each layer in stack.
    suffix: option to add a file suffix.
    '''

    if not isinstance(data, list): # Setup to handle lists
        data = [data]
        nlayers = 1
    else:
        nlayers = len(layers)

    # Current directory
    owd = os.getcwd()
    print("The current directory is: {}".format(owd))

    # Image directory
    # directory = owd

    if project is not None: # Print to current directory otherwise
        # Change the current directory to specified directory 
        if shot is None:
            tardir = "../../Projects/"+project+"/Experiment_Data/Proton_Radiography"
        else:
            tardir = "../../Projects/"+project+"/Experiment_Data/Proton_Radiography/Shot"+shot 
        os.chdir(tardir)

    # List files and directories in target directory
    # print("Before saving image:")
    # print(os.listdir(directory))

    # if isinstance(data, list): # Can only really use function like this if I make the part below its own function.

    # Iterate through stack
    for layer in range(nlayers):

        if info == "colours": # Save seperate RGB channels
            colours = ["R", "G", "B"]

            n = 0
            for colour in colours:
                filename = "RCF_"+stack+layers[layer]+"_HDR" + "_" + colour # Desired filename

                if suffix is not None:
                    filename = filename + "_" + suffix

                print("Saving file: {}".format(filename))

                # Check if object can be converted to tif or not and save image
                # If it is made of floats then we loose info so save as csv
                if data[layer].dtype == float:
                    np.savetxt(filename +'.txt', data[layer][:,:,n])

                # Save object as tif if suitable.
                else: # Save object as tif if suitable.
                    cv2.imwrite(filename +'.tif', data[layer][:,:,n])    

                n+=1

        if info == "RGB":
            filename = "RCF_"+stack+layers[layer]+"_HDR" # Desired filename

            if suffix is not None:
                filename = filename + "_" + suffix

            print("Saving file: {}".format(filename))

            raise Exception("This save method does not work correctly? Exiting.")
            # cv2.imwrite(filename +'.tif', data[layer])  

    # # List files and directories   
    # print("After saving image:")  
    # print(os.listdir(directory))
    os.chdir(owd)

    print('Successfully saved')

    return


def separate_colour_channel(data, colour):
    '''Tool for separating colour channels'''
    if colour == "R":
        c = 0
    elif colour == "G":
        c = 1
    elif colour == "B":
        c = 2

    for nlayer, _ in enumerate(data):
        data[nlayer] = data[nlayer][:,:,c]

    return data


def get_stack_type(project, shot):
    '''
    Get the type of RCF stack used on a given shot.
    Needed to get the design for the stack
    '''

    shot = int(shot)

    # owd = os.getcwd()

    # os.chdir("../../Projects/" + project + "/Experiment_Data/Proton_Radiography")

    data = pd.read_csv('RCF_Stack_Type.csv')

    # os.chdir(owd)

    data = data.values.tolist()[shot-1] # -1 because shot 1 is row 0

    design = data[1]

    # print(design)

    return design


def letter_to_num(letter):
    '''Convert a letter to number (where A=1, B=2 etc.)'''
    return ord(letter)-64


def get_stack_design(project, shot=None, design=None, info="all"):
    '''
    Get the design of each type of RCF stack.
    i.e. material (HDV2/EBT3), energy, filters, filter thickness, and depth.
    Either requires shot number or design letter.
    '''

    # owd = os.getcwd()

    # os.chdir(rootdir)

    # os.chdir("../../Projects/" + project + "/Experiment_Data/Proton_Radiography")

    data = pd.read_csv('RCF_Stack_Design.csv')

    # os.chdir(owd)

    if shot is not None:
        design = get_stack_type(project, shot)
        shot = int(shot)
    elif design is None:
        raise Exception("Either shot number or RCF design must be input.")

    design_to_num = letter_to_num(design)-1 # Convert design letter to number with "A" = 0

    data = data.values.tolist()[(5*design_to_num):(5*design_to_num+5)]

    if info == "material":
        data = data[0][1:] # Stack material

    elif info == "energy":
        data = data[1][1:] # Stack depth in MeV
        data = np.array(data, dtype=float)

    elif info == "depth":
        data = data[4][1:] # Stack depth in um
        data = np.array(data, dtype=float)/1000 # Convert to mm

    elif info == "filters":
        data = [data[2][1:], data[3][1:]]

    elif info == "all":
        for i in range(5):
            data[i] = data[i][1:]

    else:
        raise Exception("RCF information cannot be provided.")

    return data


def calc_proton_velocity(Ek_MeV, ret_gamma=False):
    '''Convert proton energy in MeV to velocity'''

    Ek = Ek_MeV * 1e6 * pc.e # Proton energy in J

    gamma = 1 + Ek/(pc.m_p * pc.c**2) # Lorentz factor from elativistic kinetic energy relation

    beta = np.sqrt(1 - gamma**-2) # Ratio of velocity to speed of light

    v0 = beta * pc.c

    if ret_gamma:
        return v0, gamma
    else:
        return v0


def get_corners(project, shot, layers, shape="rectangle"):
    '''
    Get the corner positions for a given piece of RCF.
    
    shape: can be either be "square" (returns 2 coords) or rectangular 
        (returns 3 coords) which is set during RCF_Corners script.
    '''

    shot = int(shot)

    # List containing numbers corresponding to RCF layer
    nlayer = []
    for layer in layers:    
        nlayer.append(letter_to_num(layer)-1) # Here A=0 as A is in 0th array position.

    owd = os.getcwd()

    os.chdir("../../Projects/" + project + "/Experiment_Data/Proton_Radiography")

    # Get corners of all available layers
    if shape == "square":
        data = np.genfromtxt('Corner_Positions_Square.csv', delimiter=',')
        data = data[1+(shot-1)*4:1+(shot-1)*4+4,2:]
    elif shape == "rectangle":
        data = np.genfromtxt('Corner_Positions_Rectangle.csv', delimiter=',')
        data = data[1+(shot-1)*6:1+(shot-1)*6+6,2:]

    os.chdir(owd)

    data = data[:,nlayer] # Extract only corners of desired layers

    if np.isnan(data).any():
        raise Exception("Corners have not been input. Exiting.")

    return data


def calc_angle(corners, line="horizontal"):
    '''
    Calculate the rotation of a piece of RCF based on input corner position.
    The rotation is the number or radians the top edge is rotated around the
    upper left edge. If the RCF is not actually square this will be incorrect.
    
    separation: can be either horizontal or diagonal (not sure diagonal is used).
    '''

    layers = corners.shape[1]

    angle = np.zeros(layers)

    for layer in range(layers):

        dy = -(corners[3,layer] - corners[1,layer]) # Negative because pixel value increases as y decreases
        dx = corners[2,layer] - corners[0,layer]

        if line == "diagonal":
            angle[layer] = np.arctan(dy/dx) + np.pi/4 # Change in angle in radians
            raise Exception("Is this still used? (08/11/23)")

        elif line == "horizontal":
            angle[layer] = np.arctan(dy/dx)

        print("Input RCF layer {0} is rotated by {1:.2g} degrees.".format(layer, np.degrees(angle[layer])))

    return angle


def rotate_data(data, angle, shape="rectangle", plot=False):
    '''Rotates a piece of RCF by a given angle about the centre'''

    layers = len(data)

    centre = np.zeros([2,layers])
    data_rot = []

    for layer in range(layers):

        centre[:,layer] = tuple(np.array(data[layer].shape[1::-1]) / 2)

        rot_mat = cv2.getRotationMatrix2D(tuple(centre[:,layer]), -np.degrees(angle[layer]), 1.0)

        data_rot.append(cv2.warpAffine(data[layer], rot_mat, data[layer].shape[1::-1], flags=cv2.INTER_LINEAR))

        if plot:
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(data[layer][:,:,0])
            ax[0].scatter(*centre[:,layer])
            ax[1].imshow(data_rot[layer][:,:,0])
            fig.tight_layout()

    return data_rot, centre


def calc_height(corners, angle):
    '''
    Calculate the height of a piece of RCF.
    Requires the top two corners and a position along the lower edge, as well
    as the angle of rotation.
    '''

    dx = corners[0,:] - corners[4,:]
    dy = corners[1,:] - corners[5,:]
    H = np.sqrt(dx**2 + dy**2)*np.cos(angle)
    A = abs(dy)
    phi = np.arccos(A/H)

    height = - H*np.sin(np.pi/2 - phi - angle)

    return height


def calc_missing_corners_prerot(corners, angle, shape="rectangle"):
    '''
    Finds missing corners, assuming RCF has not been yet rotated
    I will probably phase out the square shape, so will remove this with time.
    There could be a bug in here... but I haven't seen it cause real issues yet?    
    '''

    layers = corners.shape[1]

    # For storing corners
    corners00 = np.zeros([2,layers])
    corners11 = np.zeros([2,layers])

    if shape == "rectangle":
        dh = calc_height(corners, angle)
    else:
        raise Exception("Not setup to find corners for other shapes.")

    for layer in range(layers):
        if angle[layer]*180/np.pi > 5:
            print("Caution: Angle of rotation is greater than 5 deg, crop my be too large.")

        if angle[layer] < 0:
            corners00[0,layer] = corners[0,layer]
            corners00[1,layer] = corners[3,layer]
            corners11[0,layer] = corners[2,layer] - dh[layer]*np.sin(angle[layer])
            corners11[1,layer] = corners[1,layer] - dh[layer]*np.cos(angle[layer])

        if angle[layer] > 0:
            corners00[0,layer] = corners[0,layer] - dh[layer]*np.sin(angle[layer])
            corners00[1,layer] = corners[1,layer]
            corners11[0,layer] = corners[2,layer]
            corners11[1,layer] = corners[3,layer] - dh[layer]*np.cos(angle[layer])

    corners00 = np.round(corners00).astype(int)
    corners11 = np.round(corners11).astype(int)

    return corners00, corners11


def calc_missing_corners(old_corners, angle, centre, shape="rectangle"):
    '''
    Finds missing corners, assuming RCF has been rotated.
    Returns the array position of the two missing RCF corners.
    01 is top right, 10 is bottom left.
    Again, relies on RCF having been horizontal on shot
    Only top left and bottom right coordinate are required for rectangular shapes
    '''

    layers = old_corners.shape[1]

    # For storing corners
    corners00 = np.zeros([2,layers])
    corners01 = np.zeros([2,layers])
    corners11 = np.zeros([2,layers])

    old_corners0 = np.zeros_like(old_corners)

    old_corners0[[0,2],:] = - centre[0,:] + old_corners[[0,2],:]
    old_corners0[[1,3],:] = centre[1,:] - old_corners[[1,3],:]

    corners00[0,:] = old_corners0[0,:] * np.cos(-angle) - old_corners0[1,:] * np.sin(-angle)
    corners00[1,:] = old_corners0[0,:] * np.sin(-angle) + old_corners0[1,:] * np.cos(-angle)

    corners01[0,:] = old_corners0[2,:] * np.cos(-angle) - old_corners0[3,:] * np.sin(-angle)
    corners01[1,:] = old_corners0[2,:] * np.sin(-angle) + old_corners0[3,:] * np.cos(-angle)

    corners00[0,:] = centre[0,:] + corners00[0,:]
    corners00[1,:] = centre[1,:] - corners00[1,:]

    corners01[0,:] = centre[0,:] + corners01[0,:]
    corners01[1,:] = centre[1,:] - corners01[1,:]

    if shape == "square":
        dh = corners00[0,:] - corners01[0,:]
    elif shape == "rectangle":
        dh = calc_height(old_corners, angle)

    corners11[0,:] = corners01[0,:]
    corners11[1,:] = corners00[1,:] - dh

    corners00 = np.round(corners00).astype(int)
    corners11 = np.round(corners11).astype(int)

    return corners00, corners11


def crop_data(data, corners_diag=None, edge=250, shape="rectangle", corners_setup=None,
              plot=False):
    '''
    Crop RCF data using the input corner positions.
    Corners is a tuple specifying the upper left and lower right corner.
    The crop border border can be extended using "edge", which can be a single
    value or list to allow different cropping in x/y.
    '''

    layers = len(data)

    data_crop = []

    if isinstance(edge, list):
        edgeX = edge[0]
        edgeY = edge[1]
    else:
        edgeX = edge
        edgeY = edge

    print("Cropping extra {} pixels off each border due to noise.".format(edge))

    for layer in range(layers):

        if (data[layer].shape[0]<2*edgeY) or (data[layer].shape[1]<2*edgeX):
            raise Exception("Crop area is larger than image. Exiting.")

        if corners_diag is None:
            corners00 = [0,0]
            corners11 = data[layer].shape[1::-1]
        else:
            corners00 = corners_diag[0][:,layer]
            corners11 = corners_diag[1][:,layer]

        data_crop.append(data[layer][corners00[1]+edgeY:corners11[1]-edgeY,
                                     corners00[0]+edgeX:corners11[0]-edgeX,:])

        if plot: # Final array position in slice is dropped so -1 off end.
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(data[layer][:,:,0])
            ax[0].scatter(corners00[0], corners00[1], marker="x", color="r", s=200) # 00
            ax[0].scatter(corners11[0]-1, corners11[1]-1, marker="x", color="r", s=200) # 11
            ax[0].scatter(corners00[0], corners11[1]-1, marker="^", color="r", s=200) # 01
            ax[0].scatter(corners11[0]-1, corners00[1], marker="v", color="r", s=200) # 10
            ax[1].imshow(data_crop[layer][:,:,0])

            if corners_setup is not None:
                ax[0].scatter(corners_setup[0,layer], corners_setup[1,layer],
                              marker="x", color="white", s=200) # UL
                ax[0].scatter(corners_setup[2,layer], corners_setup[3,layer],
                              marker="x", color="white", s=200) # UL
                ax[0].scatter(corners_setup[4,layer], corners_setup[5,layer],
                              marker="x", color="white", s=200) # UL
                ax[0].axvline(corners_setup[0,layer], color="white")
                ax[0].axvline(corners_setup[2,layer], color="white")
                ax[0].axvline(corners_setup[4,layer], color="white")
                ax[0].axhline(corners_setup[1,layer], color="white")
                ax[0].axhline(corners_setup[3,layer], color="white")
                ax[0].axhline(corners_setup[5,layer], color="white")

            fig.tight_layout()

    return data_crop


def rotate_crop_data(data, project, shot, layers, edge=250, rot=False, 
                     shape="rectangle", plot=False):
    '''
    Function for handling rotating and then cropping RCF data.
    Rotation can introduce grid effects in data.
    Data must be provided, and project/shot/layers specified.
    Not perfect as it assumes the RCF was positioned "exactly"
    
    rot: if False rotation function is turned off.
    '''

    corners = get_corners(project, shot, layers, shape=shape)
    angle = calc_angle(corners)

    if rot:
        data_r, centre = rotate_data(data, angle, shape=shape)
        corners_rot = calc_missing_corners(corners, angle, centre, shape=shape)
        data_rc = crop_data(data_r, corners_diag=corners_rot, edge=edge, shape=shape)

        data_list = [data, data_r, data_rc]

    else:
        corners_prerot = calc_missing_corners_prerot(corners, angle, shape=shape)
        data_c = crop_data(data, corners_diag=corners_prerot, edge=edge, shape=shape)

        data_list = [data, data_c]

    if plot:
        plot_comparison(data_list, layers, colour=0)

    return data_list[-1]


def get_rotate_crop_data(project, shot, stack, layers, location=False, dtype=".tif",
                         suffix=None, edge=250, shape="rectangle", plot=False):
    '''
    Gets the rotated+cropped data for desired layers in a stack for a given shot.
    Returns a list.
    '''

    data = get_stack_data(project, shot, stack, layers, location=location, 
                          dtype=dtype, suffix=suffix, plot=plot)

    data = rotate_crop_data(data, project, shot, layers, edge=edge, shape=shape, 
                            plot=plot)

    return data


def convert_dpi_to_m(dpi, units="mm"):
    '''Converst dpi to metres'''

    if units == "m":
        scale = 1
    elif units=="mm":
        scale = 1e3

    return (25400e-6/dpi) * scale


# %% Plotting functions

def plot_data(data, title=None, colour=None, cbar_label=None, cmap="rainbow", 
              dpi=300, logc=False, save=False):
    '''
    Generic function for plotting RGB images.
     
    colour: use R=0, G=1, B=2
    '''

    res = convert_dpi_to_m(dpi, units="mm")

    # Plot all colour channels
    if colour is None:
        data_R = data[:,:,0]
        data_G = data[:,:,1]
        data_B = data[:,:,2]

        ny, nx = data_R.shape[:2]

        fig, (ax_R, ax_G, ax_B) = pm.plot_figure_axis("small", 3, ratio=[1,1])
        ax_R.imshow(data_R, cmap=cmap)
        ax_G.imshow(data_G, cmap=cmap)
        ax_B.imshow(data_B, cmap=cmap)

    # Plot single channel with colour bar
    else:
        if colour < 3:
            data_C = data[:,:,colour]
        else:
            data_C = data#/1000

        if logc:
            data_C = np.log10(data_C)

        ny, nx = data_C.shape[:2]

        if 1:
            fig, ax = pm.plot_figure_axis("small", 1, ratio=[1.25,1.25*ny/nx])
        im_extent=[0,nx*res,0,ny*res]
        im = ax.imshow(data_C, cmap=cmap, extent=im_extent)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        # Setup colour bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        if cbar_label is not None:
            cbar.ax.get_yaxis().labelpad = 30
            cbar.ax.set_ylabel(cbar_label, rotation=270)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    if save:
        fig.savefig("temp.png", dpi=300)

    return


def plot_data_list(data_list, title=None, colour=None, cbar_label=None, cmap="rainbow", 
                   dpi=300, logc=False, save=False):
    '''
    Generic function for plotting RGB image datasets.
     
    colour: use R=0, G=1, B=2
    '''

    if not isinstance(data_list, list): # Setup to handle everything as a list
        data_list = [data_list]
        nlayers = 1
    else:
        nlayers = len(data_list)

    # Iterate through layers
    for n in range(nlayers):   
        plot_data(data_list[n], title=title, colour=colour, cbar_label=cbar_label,
                  cmap=cmap, dpi=dpi, logc=logc, save=save)

    return


def plot_data_cRGB(data, flipx=True, flipy=False, dpi=300):
    '''
    Plot colour and RGB images.
    '''

    res = convert_dpi_to_m(dpi)

    ny, nx = data.shape[:2]

    im_extent = [0, nx*res, 0, ny*res]

    colours = ["Reds_r", "Greens_r", "Blues_r"]

    if flipx:
        data = data[:,::-1,:]
    if flipy:
        data = data[::-1,:,:]

    data_8bit = data/(2**16)

    fig, ax = pm.plot_figure_axis("small", 4, shape=2, ratio=[1,1/(nx/ny)+0.03])

    ax[0].imshow(data_8bit, vmin=0, vmax=1, extent=im_extent)
    ax[0].tick_params(direction='in', length=8)


    for c in range(1,4):
        ax[c].imshow(data[:,:,c-1], cmap=colours[c-1], vmin=0, vmax=2**16, extent=im_extent)
        ax[c].tick_params(axis="y", direction='in', length=8)
        ax[c].tick_params(axis="x", direction='in', length=8)

    # for axi in ax:
    ax[0].set_xticklabels([])
    ax[1].set_xticklabels([])
    ax[1].set_yticklabels([])
    ax[3].set_yticklabels([])

    ax[0].set_ylabel("y (mm)")
    ax[2].set_xlabel("x (mm)")
    ax[2].set_ylabel("y (mm)")
    ax[3].set_xlabel("x (mm)")

    fig.subplots_adjust(wspace=0, hspace=0)

    fig.tight_layout()

    # fig.savefig('temp.png', dpi=300)

    return


def plot_comparison(data_list, average=False, colour=0, title_list=None, 
                    main_title=None):
    '''
    For (mainly) comparing before/after images of RCF but can be anything.
    
    average: if True image is averaged over the 3 colour channels (2nd array axis)
    '''

    if isinstance(data_list[0], list):
        nlayers = len(data_list[0])
    else:
        data_list[0] = [data_list[0]]
        data_list[1] = [data_list[1]]
        nlayers = 1

    nimages = len(data_list)

    # Iterate through layers
    for n in range(nlayers):

        fig, ax = pm.plot_figure_axis("small", nimages)

        for i in range(nimages):
            if average:
                ax[i].imshow(np.average(data_list[i][n], axis=2))
            else:
                if np.count_nonzero(np.isnan(data_list[i][n][:,:,colour])) > 0:
                    interp="none"
                else:
                    interp=None
                ax[i].imshow(data_list[i][n][:,:,colour], interpolation=interp)

            if title_list is not None:
                ax[i].set_title(title_list[i])

        if main_title is not None:
            fig.suptitle(main_title)

        fig.tight_layout()

    return


# %% Testing

if __name__ == "__main__":

    # project = "Woolsey_2019"
    project = "Carroll_2020"

    # shot = "056"
    shot = "001"

    # stack = "56"
    stack = "1"

    layers = ["B"]
    # layers = ["B","C","D","E","F"]
    # layers = ["A","B","C","D","E"]#,"F","G","H","I","J","K","L","M","N","O","P","Q"]

    if 0: # Test getting raw data
        RCF_data = get_radiography_data(project, "Null", "EBT3", plot=True)

    if 0:
        RCF_data = get_stack_data(project, shot, stack, layers, suffix=None, plot=True)

    if 1: # Test cropping data
        RCF_data = get_stack_data(project, shot, stack, layers, suffix=None, plot=False)

        corners = get_corners(project, shot, layers, shape="rectangle")

        angle = calc_angle(corners)

        corners_prerot = calc_missing_corners_prerot(corners, angle)        

        RCF_data_c = crop_data(RCF_data, corners_diag=corners_prerot, edge=0, 
                               shape="rectangle", corners_setup=corners, plot=True)

    if 0: # Test  rotating and cropping data
        RCF_data = get_stack_data(project, shot, stack, layers, suffix=None, plot=False)

        RCF_data_rc = rotate_crop_data(RCF_data, project, shot, layers, edge=[50,0], 
                                       shape="rectangle", plot=True)

    if 0: # Test getting stack design
        stack_design = get_stack_design(project, shot, info="filters")

    if 0: # Set to save in main directory
        save_radiography_data(RCF_data, stack, layers, info="colour")

plt.show()
            