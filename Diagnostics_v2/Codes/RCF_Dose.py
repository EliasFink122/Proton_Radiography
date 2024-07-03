# -*- coding: utf-8 -*-
"""
Created on Wed Jul 03 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Converting RGB images into doses.
"""

import RCF_Image_Crop as ic
import scipy.interpolate as inter
import numpy as np

ROOTDIR = "/Users/eliasfink/Desktop/Proton_Radiography/Diagnostics_v2"


def interpolate_calibration(scanner = "Epson_12000XL"):
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
    calibration_data = np.loadtxt(ROOTDIR + "/Calibration/" + calibration_path, skiprows = 2)

    log_dose = np.array(calibration_data[:, 2])
    red_data = np.array(calibration_data[:, 3])
    green_data = np.array(calibration_data[:, 5])
    blue_data = np.array(calibration_data[:, 7])

    red_cs = inter.CubicSpline(red_data, log_dose)
    green_cs = inter.CubicSpline(green_data, log_dose)
    blue_cs = inter.CubicSpline(blue_data, log_dose)

    def overall_cs(rgb: np.ndarray) -> float:
        return np.power(10, np.mean([red_cs(rgb[0]), green_cs(rgb[1]), blue_cs(rgb[2])]))

    return overall_cs


def convert_to_dose(project: str, shot: str):
    '''
    Convert data images to dose pictures

    Args:
        project: project of interest
        shot: shot of interest
        func: output function of interpolate_calibration
    '''
    imgs = ic.crop_rot(ROOTDIR + "/Data/" + project + "/" + "Shot" + shot + "/raw.tif")

    func = interpolate_calibration()

    converted_imgs = []
    for img in imgs:
        for i, row in img:
            for j, pixel in row:
                img[i, j] = func(pixel)
        converted_imgs.append(img)

    return converted_imgs
