# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Converting RGB images into doses.
"""

import RCF_Image_Crop as ic
import scipy.interpolate as inter
import numpy as np
import matplotlib.pyplot as plt

ROOTDIR = "/Users/eliasfink/Desktop/Proton_Radiography/Diagnostics_v2"


def interpolate_calibration(scanner = "Epson_12000XL", log = False):
    '''
    Interpolates calibration data

    Args:
        scanner: name of camera used to scan images

    Returns:
        function used to match 16-bit RGB values to 
    '''
    calibration_path = ""
    if scanner == "Epson_12000XL":
        calibration_path = "/Epson_12000XL/HDV2_3_calibration.csv"
    calibration_data = np.loadtxt(ROOTDIR + "/Calibration/" + calibration_path,
                                  skiprows = 2, delimiter = ',')
    red_data = np.array(([calibration_data[:, 2].transpose()] + [calibration_data[:, 3].transpose()])).transpose()
    red_data = red_data[red_data[:, 1].argsort()]
    green_data = np.array(([calibration_data[:, 2].transpose()] + [calibration_data[:, 5].transpose()])).transpose()
    green_data = green_data[green_data[:, 1].argsort()]
    blue_data = np.array(([calibration_data[:, 2].transpose()] + [calibration_data[:, 7].transpose()])).transpose()
    blue_data = blue_data[blue_data[:, 1].argsort()]

    red_cs = inter.CubicSpline(red_data[:, 1], red_data[:, 0])
    green_cs = inter.CubicSpline(green_data[:, 1], green_data[:, 0])
    blue_cs = inter.CubicSpline(blue_data[:, 1], blue_data[:, 0])

    if log:
        def overall_cs_log(rgb: np.ndarray) -> float:
            return np.mean([red_cs(rgb[0]), green_cs(rgb[1]), blue_cs(rgb[2])])
        return overall_cs_log

    def overall_cs(rgb: np.ndarray) -> float:
        return np.power(10, np.mean([red_cs(rgb[0]), green_cs(rgb[1]), blue_cs(rgb[2])]))

    return overall_cs


def convert_to_dose(project: str, shot: str, log = False) -> list[np.ndarray]:
    '''
    Convert data images to dose pictures

    Args:
        project: project of interest
        shot: shot of interest
        func: output function of interpolate_calibration
    '''
    imgs = ic.crop_rot(ROOTDIR + "/Data/" + project + "/" + "Shot" + shot + "/raw.tif")

    func = interpolate_calibration(log = log)

    converted_imgs = []
    for img in imgs:
        for i, row in enumerate(img):
            for j, pixel in enumerate(row):
                if np.mean(pixel) == 255:
                    img[i, j] = 0
                    continue
                img[i, j] = func((pixel + 1)*256 - 1)
        converted_imgs.append(img)

    return converted_imgs

imgs_conv = convert_to_dose(project = "Carroll_2023", shot = "001", log = False)
plt.imshow(imgs_conv[0]/np.max(imgs_conv[0]), cmap = "rainbow")
plt.show()
