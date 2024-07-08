"""
Created on Mon Jul 08 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Finding the beam divergence of the proton laser on the RCF plates.

Methods:
    image_conversion:
        converts all RCF images into brightness values
    brightness_plot:
        determine change of brightness over width and height
    find_blob:
        find the coordinates of the dark spot in each image
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import proton_mass as m_p
from scipy.constants import elementary_charge as e
from scipy.constants import speed_of_light as c
from RCF_Plotting import ROOTDIR
import RCF_Image_Crop as ic

def image_conversion(project: str, shot: str, imshow = False, plot = False) -> list[np.ndarray]:
    '''
    Convert image from RGB to brightness

    Args:
        project: project of interest
        shot: shot of interest in xxx format (e.g. "001")
    
    Returns:
        list of converted images
    '''
    imgs = ic.crop_rot(ROOTDIR + "/Data/" + project + "/" + "Shot" + shot + "/raw.tif")

    new_imgs = []
    for img in imgs:
        # convert pixel values to brightness and slice of edges
        new_img = np.zeros(np.shape(img)[:2])
        for i, row in enumerate(img):
            for j, pixel in enumerate(row): # mean of pixel RGB is brightness value
                new_img[i, j] = np.mean(pixel)
        new_img = new_img[20:(len(new_img) - 20), 20:(len(new_img[0]) - 20)]

        new_imgs.append(new_img)

        if imshow: # show black-white brightness images
            plt.imshow(new_img, cmap = 'rainbow')
            plt.show()

        if plot: # plot 3d surfaces of brightness
            x = range(len(new_img.transpose()))
            y = range(len(new_img))

            fig = plt.figure()
            subpl = fig.add_subplot(111, projection = '3d')
            x, y = np.meshgrid(x, y)
            subpl.plot_surface(x, y, new_img)
            plt.show()

    return new_imgs

def brightness_plot(imgs: list[np.ndarray], plot = False) -> list[tuple[list[float], list[float]]]:
    '''
    Turn each brightness image into brightness plots for x and y direction

    Args:
        imgs: list of brightness-converted images

    Returns:
        list of brightness for each image in x and y
    '''
    brightness_curves = []
    for img in imgs:
        y_vals = []
        y_brightness = []
        for i, row in enumerate(img): # mean for each pixel row
            y_vals.append(i)
            y_brightness.append(np.mean(row))
        x_vals = []
        x_brightness = []
        for i, column in enumerate(img.transpose()): # mean for each pixel column
            x_vals.append(i)
            x_brightness.append(np.mean(column))

        brightness_curves.append((x_brightness, y_brightness))

        if plot: # plot x and y brightness curves
            plt.plot(y_vals, y_brightness, label = "Y-range")
            plt.plot(x_vals, x_brightness, label = "X-range")
            plt.legend()
            plt.show()

    return brightness_curves

def find_blob(brightness_curves: list[tuple[list[float], list[float]]],
              plot = False) -> tuple[list[float], list[float]]:
    '''
    Find x and y coordinates of central blob

    Args:
        brightness_curves: input brightness distributions for x and y

    Returns:
        position of central blob (from centre of image)
    '''
    xs = []
    ys = []
    for curves in brightness_curves:
        x_curve = curves[0]
        y_curve = curves[1]

        # darkest spot in image from centre
        x, y = np.argmin(x_curve), np.argmin(y_curve)
        xs.append(np.abs(x - len(x_curve)/2))
        ys.append(np.abs(y - len(y_curve)/2))

        if plot:
            plt.plot(range(len(x_curve)), x_curve)
            plt.plot(range(len(y_curve)), y_curve)
            plt.plot(x, 0, 'o')
            plt.plot(y, 0, 'o')
    if plot:
        plt.show()
    return [np.mean(xs), np.mean(ys)], [np.std(xs), np.std(ys)]

def integrated_magnetic_field(temp: float, coords: list,
                              err: list = None) -> tuple[list[float], list[float]]:
    '''
    Use central blob coordinates to determine integrated magnetic field of proton path

    Args:
        temp: temperature of protons in MeV (vx, vy assumed to be zero)
        x, y: central blob coordinates
    
    Returns:
        integrated Bx and By
    '''
    # e_j = temp * 1e6 * e + m_p * c**2
    vz = c # np.sqrt(e_j**2 - (m_p * c**2)**2) * c / e_j

    int_magn_x = coords[1] * m_p * vz / e
    int_magn_y = coords[0] * m_p * vz / e

    if err is not None:
        err_magn_x = err[1] * m_p * vz / e
        err_magn_y = err[0] * m_p * vz / e
    else:
        err_magn_x, err_magn_y = 0, 0

    return [int_magn_x, int_magn_y], [err_magn_x, err_magn_y]

if __name__ == "__main__":
    print("Reading in data...")
    images = image_conversion("Carroll_2023", "001")
    print("Creating curves...")
    curves_tuple = brightness_plot(images)
    print("Finding blob coordinates...")
    blob_coords = find_blob(curves_tuple)
    x_str = f"x = {blob_coords[0][0]:.1f} +- {blob_coords[1][0]:.1f} px"
    y_str = f"y = {blob_coords[0][1]:.1f} +- {blob_coords[1][1]:.1f} px"
    print(f"Central blob coordinates: {x_str}, {y_str}")
    print("Calculating magnetic fields...")
    b_fields = integrated_magnetic_field(20, *blob_coords)
    bx_str = f"Bx = {b_fields[0][0]:.1f} +- {b_fields[1][0]:.1f} T px"
    by_str = f"By = {b_fields[0][1]:.1f} +- {b_fields[1][1]:.1f} T px"
    print(f"Integrated magnetic fields: {bx_str}, {by_str}")
