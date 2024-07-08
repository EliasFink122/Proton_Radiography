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
import scipy.optimize as op
import RCF_Image_Crop as ic
from RCF_Plotting import lorentzian
from RCF_Dose import ROOTDIR

def image_conversion(project: str, shot: str, imshow: bool, plot: bool) -> list[np.ndarray]:
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
        new_img = np.zeros(np.shape(img)[:2])
        for i, row in enumerate(img):
            for j, pixel in enumerate(row):
                new_img[i, j] = np.mean(pixel)

        new_imgs.append(new_img)

        if imshow: # show black-white brightness images
            plt.imshow(new_img, cmap = 'rainbow')
            plt.show()

        if plot: # plot 3d surfaces of brightness
            x = range(len(new_img.transpose()))
            y = range(len(new_img))

            fig = plt.figure()
            subpl = fig.add_subplot(111, projection = '3d')
            X, Y = np.meshgrid(x, y)
            subpl.plot_surface(X, Y, new_img)
            plt.show()

    return new_imgs

def brightness_plot(imgs: list[np.ndarray], plot: bool) -> list[tuple[list, list]]:
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
        for i, row in enumerate(img):
            y_vals.append(i)
            y_brightness.append(np.mean(row))
        x_vals = []
        x_brightness = []
        for i, row in enumerate(img.transpose()):
            x_vals.append(i)
            x_brightness.append(np.mean(row))

        brightness_curves.append((x_brightness, y_brightness))

        if plot: # plot x and y brightness curves
            plt.plot(y_vals, y_brightness, label = "Y-range")
            plt.plot(x_vals, x_brightness, label = "X-range")
            plt.legend()
            plt.show()

    return brightness_curves

def find_blob(brightness_curves: list[tuple[list, list]]) -> tuple[float, float]:
    '''
    Find x and y coordinates of central blob

    Args:
        brightness_curves: input brightness distributions for x and y

    Returns:
        position of central blob
    '''
    for curves in brightness_curves:
        x_curve = curves[0]
        y_curve = curves[1]

        x, y = np.argmin(x_curve), np.argmin(y_curve)
        print(x, y)

        plt.plot(range(len(x_curve)), x_curve)
        plt.plot(range(len(y_curve)), y_curve)
        plt.plot(0, x)
        plt.plot(0, y)
        plt.show()

if __name__ == "__main__":
    images = image_conversion("Carroll_2023", "001", imshow = False, plot = False)
    curves_tuple = brightness_plot(images, plot = False)
    find_blob(curves_tuple)
