"""
Created on Wed Jul 03 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Using stopping power data to match layers to energies of protons.
"""

import RCF_Dose as dose
import numpy as np
import astropy.units as units
from plasmapy.diagnostics.charged_particle_radiography.detector_stacks import Layer, Stack

def get_mass_stopping_power(material: str, database="SRIM") -> np.ndarray[np.ndarray[float, float]]:
    '''
    Get stopping power data from file
    @author: Adam Dearling (add525@york.ac.uk)

    Args:
        material: material of RCF layer
        database: source of data (only SRIM for now)

    Returns:
        list of all proton energies in MeV with stopping powers for particular material
    '''
    # PlasmaPy was setup to work with PSTAR files, which have units of MeV/(g/cm2)
    # so I will convert SRIM data (which is in MeV/(mg/cm2)) here for simplicity.
    # Additionally, SRIM outputs electronic and nuclear stopping power separtely.
    # PlasmaPy wants the total stopping power, so we will combine here.

    file = dose.ROOTDIR + "/Codes/Stopping_Power/" + database + "_" + material + ".txt"

    if database == "PSTAR":
        data = np.loadtxt(file, skiprows=8)

    elif database == "SRIM": # Have to manually handle formatting
        with open(file, mode = 'r', encoding = "utf-8") as f:
            data_raw = f.readlines()
            str_start = '-----------  ---------- ---------- ----------  ----------  ----------\n'
            str_end = '-----------------------------------------------------------\n'
            try:
                row_start = data_raw.index(str_start)
            except IndexError:
                row_start = data_raw.index('  ---' + str_start)
            row_end = data_raw.index(str_end)
            data_raw = data_raw[row_start+1:row_end]

            # Format data into array
            data = np.zeros((len(data_raw),2))
            i = 0
            for row in data_raw:
                row_split = row.split()
                # print(row_split)
                if row_split[1] == "keV":
                    data[i,0] = float(row_split[0])/1000
                elif row_split[1] == "MeV":
                    data[i,0] = float(row_split[0])
                data[i,1] = float(row_split[2]) + float(row_split[3])
                i+=1

        data[:,1] = data[:,1]*1000 # Convert to g/cm2

    return data

def get_target_density(material: str, database="SRIM") -> float:
    '''
    Get the target density in g/cm3.
    
    @author: Adam Dearling (add525@york.ac.uk)

    Args:
        material: material of RCF layer
        database: source of data (only SRIM for now)

    Returns:
        density of target layer
    '''

    file = dose.ROOTDIR + "/Codes/Stopping_Power/" + database + "_" + material + ".txt"

    if database == "PSTAR":
        raise FileNotFoundError("PSTAR files do not contain target density.")

    elif database == "SRIM": # Have to manually handle formatting
        with open(file, mode = 'r', encoding = 'utf-8') as f:
            for row in f.readlines():
                if "Target Density" in row:
                    target_density = float(row.split()[3])
                    break

    return target_density

def linear_stopping_power(material: str) -> tuple[np.ndarray[float], np.ndarray[float]]:
    '''
    Get the linear stopping power, as SRIM/PSTAR data is for mass stopping power
    
    @author: Adam Dearling (add525@york.ac.uk)

    Args:
        material: material of RCF layer

    Returns:
        stopping energy of layer
        stopping power of layer
    '''
    density = get_target_density(material) * units.g / units.cm**3
    stopping_mass = get_mass_stopping_power(material)
    stopping_power = stopping_mass[:,1] * units.MeV * units.cm**2 / units.g * density # Convert stopping power to astropy units
    stopping_energy = stopping_mass[:,0] * units.MeV # Convert corresponding energy to astropy units

    return stopping_energy, stopping_power

def layer_EBT3():
    '''
    Define the EBT3 layer
    Consists of "Polyester_Substrate" and "EBT2_Active_Layer"

    @author: Adam Dearling (add525@york.ac.uk)

    Returns:
        EBT3 layer
    '''

    active_layer = "EBT2_Active_Layer"
    substrate ="Polyester_Substrate"

    act_stopping_energy_power = linear_stopping_power(active_layer)
    sub_stopping_energy_power = linear_stopping_power(substrate)

    EBT3 = [Layer(125 * units.um, *sub_stopping_energy_power, active=False),
            Layer(28 * units.um, *act_stopping_energy_power, active=True),
            Layer(125 * units.um, *sub_stopping_energy_power, active=False),]

    return EBT3

def layer_HDV2():
    '''
    Define the HDV2 layer
    Consists of "Polyester_Substrate" and "HDV2_Active_Layer"

    @author: Adam Dearling (add525@york.ac.uk)

    Returns:
        HDV2 layer
    '''

    active_layer = "HDV2_Active_Layer"
    substrate ="Polyester_Substrate"

    act_stopping_energy_power = linear_stopping_power(active_layer)
    sub_stopping_energy_power = linear_stopping_power(substrate)

    HDV2 = [Layer(12 * units.um, *act_stopping_energy_power, active=True),
            Layer(97 * units.um, *sub_stopping_energy_power, active=False),]

    return HDV2
