"""
Created on Wed Jul 03 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Converting RGB images into doses.

Methods:
    interpolate_calibration:
        imports calibration files
        takes all calibration data and interpolates it for given material
        can be used for converting RGB value to dose
    convert_to_dose:
        takes whole image and converts each pixel to dose value
        makes use of RCF_Image_Crop.crop_rot method
"""

import RCF_Image_Crop as ic
import scipy.interpolate as inter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    else:
        raise ValueError("No other scanner available.")
    hdv2_calibration_data = np.loadtxt(ROOTDIR + "/Calibration/" + calibration_path,
                                  skiprows = 2, delimiter = ',')
    hdv2_red_data = np.array(([hdv2_calibration_data[:, 2].transpose()] +
                         [hdv2_calibration_data[:, 3].transpose()])).transpose()
    hdv2_red_data = hdv2_red_data[hdv2_red_data[:, 1].argsort()]
    hdv2_green_data = np.array(([hdv2_calibration_data[:, 2].transpose()] +
                           [hdv2_calibration_data[:, 5].transpose()])).transpose()
    hdv2_green_data = hdv2_green_data[hdv2_green_data[:, 1].argsort()]
    hdv2_blue_data = np.array(([hdv2_calibration_data[:, 2].transpose()] +
                          [hdv2_calibration_data[:, 7].transpose()])).transpose()
    hdv2_blue_data = hdv2_blue_data[hdv2_blue_data[:, 1].argsort()]

    hdv2_red_cs = inter.CubicSpline(hdv2_red_data[:, 1], hdv2_red_data[:, 0])
    hdv2_green_cs = inter.CubicSpline(hdv2_green_data[:, 1], hdv2_green_data[:, 0])
    hdv2_blue_cs = inter.CubicSpline(hdv2_blue_data[:, 1], hdv2_blue_data[:, 0])

    def cs_hdv2(rgb: np.ndarray) -> float:
        cs_arr = []
        if -3 < hdv2_red_cs(rgb[0]) < 6:
            cs_arr.append(red_cs(rgb[0]))
        if -3 < hdv2_green_cs(rgb[1]) < 6:
            cs_arr.append(hdv2_green_cs(rgb[0]))
        if -3 < hdv2_blue_cs(rgb[2]) < 6:
            cs_arr.append(hdv2_blue_cs(rgb[0]))
        return np.power(10, np.mean(cs_arr))

    if scanner == "Epson_12000XL":
        calibration_path = "/Epson_12000XL/EBT3_2_calibration.csv"
    else:
        raise ValueError("No other scanner available.")
    calibration_data = np.loadtxt(ROOTDIR + "/Calibration/" + calibration_path,
                                  skiprows = 2, delimiter = ',')
    red_data = np.array(([calibration_data[:, 2].transpose()] +
                         [calibration_data[:, 3].transpose()])).transpose()
    red_data = red_data[red_data[:, 1].argsort()]
    green_data = np.array(([calibration_data[:, 2].transpose()] +
                           [calibration_data[:, 5].transpose()])).transpose()
    green_data = green_data[green_data[:, 1].argsort()]
    blue_data = np.array(([calibration_data[:, 2].transpose()] +
                          [calibration_data[:, 7].transpose()])).transpose()
    blue_data = blue_data[blue_data[:, 1].argsort()]

    red_cs = inter.CubicSpline(red_data[:, 1], red_data[:, 0])
    green_cs = inter.CubicSpline(green_data[:, 1], green_data[:, 0])
    blue_cs = inter.CubicSpline(blue_data[:, 1], blue_data[:, 0])

    def cs_ebt3(rgb: np.ndarray) -> float:
        cs_arr = []
        if -3 < red_cs(rgb[0]) < 6:
            cs_arr.append(red_cs(rgb[0]))
        if -3 < green_cs(rgb[1]) < 6:
            cs_arr.append(green_cs(rgb[0]))
        if -3 < blue_cs(rgb[2]) < 6:
            cs_arr.append(blue_cs(rgb[0]))
        return np.power(10, np.mean(cs_arr))

    return cs_hdv2, cs_ebt3


def convert_to_dose(project: str, shot: str) -> list[np.ndarray]:
    '''
    Convert data images to dose pictures

    Args:
        project: project of interest
        shot: shot of interest
        func: output function of interpolate_calibration

    Returns:
        all images converted from RGB to dose
    '''
    imgs = ic.crop_rot(ROOTDIR + "/Data/" + project + "/" + "Shot" + shot + "/raw.tif")

    path = ROOTDIR + "/Data/" + project + "/Shot" + shot + "/RCF_Stack_Design.csv"
    stack_design = pd.read_csv(path, sep = ',')

    func_hdv2, func_ebt3 = interpolate_calibration()

    converted_imgs = []
    for i, img in enumerate(imgs):
        material = stack_design[f"Layer {i+1}"][0]
        if material == "HDV2":
            func = func_hdv2
        elif material == "EBT3":
            func = func_ebt3
        else:
            raise ValueError("Invalid stack design.")
        for i, row in enumerate(img):
            for j, pixel in enumerate(row):
                hdv2_empty = np.mean(pixel) >= 120 and material == "HDV2"
                ebt3_empty = np.mean(pixel) >= 90 and material == "EBT3"
                if ebt3_empty or hdv2_empty:
                    img[i, j] = 0
                    continue
                try:
                    img[i, j] = func((pixel + 1)*256 - 1)
                except OverflowError:
                    print(np.mean(pixel)*256, material)
                    img[i, j] = 0
        converted_imgs.append(img)

    return converted_imgs


if __name__ == "__main__":
    imgs_conv = convert_to_dose(project = "Carroll_2023", shot = "001")
    plt.imshow(imgs_conv[0]/np.max(imgs_conv[0]), cmap = "rainbow")
    plt.show()
