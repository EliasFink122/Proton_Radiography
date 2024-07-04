"""
Created on Wed Jul 03 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Convert dose images to proton energy spectrum curve and fit Maxwellian to extract temperature

Methods:
    calc_active_layer_mass:
        determines mass of measured layer (deposited energy = dose * mass)
        returns mass in kg
    deposited_energy:
        uses dose and mass to determine total deposited energy in layer
        uses RCF_Dose.convert_to_dose method
"""

import RCF_Dose as dose
import pandas as pd
import numpy as np

def calc_active_layer_mass(dpi: int, material: str) -> float:
    '''
    Calculate the mass of the active layer in one pixel
    @author: Adam Dearling (add525@york.ac.uk)
    
    Args:
        dpi: resolution of images
        material: material of stack layer

    Returns:
        mass of active layer in kg
    '''
    px_size_m = 25400e-6/dpi
    px_area_m2 = px_size_m**2

    if material == "HDV2":
        active_layer_depth = 12e-6
        active_layer_density_gpercm3 = 1.08
    elif material == "EBT3":
        active_layer_depth = 28e-6
        active_layer_density_gpercm3 = 1.2
    else:
        raise RuntimeError("Invalid active layer.")

    px_volume_m3 = px_area_m2*active_layer_depth
    px_volume_cm3 = px_volume_m3*1e6

    active_layer_mass_g = active_layer_density_gpercm3 * px_volume_cm3
    active_layer_mass_kg = active_layer_mass_g / 1e3

    return active_layer_mass_kg

def deposited_energy(project: str, shot: str) -> list[float]:
    '''
    Calculate deposited energy for each layer
    
    Args:
        project: project of interest
        shot: shot of interest
    
    Returns:
        list of deposited energies for each layer in MeV
    '''
    imgs_converted = dose.convert_to_dose(project, shot, log = False)
    path = dose.ROOTDIR + "/Data/" + project + "/Shot" + shot + "/RCF_Stack_Design.csv"
    stack_design = pd.read_csv(path, sep = ',')

    deposited_energies = []
    for i, layer in enumerate(imgs_converted):
        dpi = 100
        mass = calc_active_layer_mass(dpi, stack_design[f"Layer {i+1}"][0])
        pixel_energies = layer * mass
        deposited_energies.append(np.sum(pixel_energies))

    return deposited_energies

if __name__ == "__main__":
    print(deposited_energy("Carroll_2023", "001"))
