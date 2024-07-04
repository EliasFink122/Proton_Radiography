"""
Created on Wed Jul 03 2024

@author: Elias Fink (elias.fink22@imperial.ac.uk)

Using stopping power data to match layers to energies of protons.

Methods:
    get_mass_stopping_power:
        imports stopping power data
        extracts information on proton energy to stopping power of particular material
    get_target_density:
        determines density of target layer
    linear_stopping_power:
        converts mass stopping power to linear stopping power
    layer_EBT3:
        builds stack layer with active EBT3
    layer_HDV2:
        builds stack layer with active HDV2
    build_layers:
        get all layers in stack (not just active)
        include filter layers
    build_stack:
        create plasmapy Stack object with all layers
"""

import RCF_Dose as dose
import RCF_Plotting as pm
import numpy as np
import pandas as pd
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

    # convert units
    stopping_power = stopping_mass[:,1] * units.MeV * units.cm**2 / units.g * density
    stopping_energy = stopping_mass[:,0] * units.MeV

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

def build_layers(project: str, shot: str) -> list[str]:
    '''
    Build up all layers of given stack design
    
    Args:
        project: project of interest
        shot: shot of interest

    Returns:
        all layers (not just active layers)
    '''
    path = dose.ROOTDIR + "/Data/" + project + "/Shot" + shot + "/RCF_Stack_Design.csv"
    stack_design = pd.read_csv(path, sep = ',')

    stack_material = stack_design[:][0]
    stack_filters = stack_design[:][2:4]

    layers = []
    for i, material in enumerate(stack_material):
        if material is np.nan:
            print(f"{i+1} RCF layers in stack.")
            break

        if stack_filters[0][i].count("/") > 0: # multiple filters
            n_filters = stack_filters[0][i].count("/") + 1
            layer_filters = stack_filters[0][i].split("/")
            layer_filters_thickness = stack_filters[1][i].split("/")
            for j in range(n_filters):
                layers.append([layer_filters[j], layer_filters_thickness[j]])
        else:
            layers.append([stack_filters[0][i], stack_filters[1][i]])
        layers.append([material])

    return layers

def build_stack(project: str, shot: str):
    '''
    Build and RCF stack with a given design.

    @author: Adam Dearling (add525@york.ac.uk)
    @edited: Elias Fink (elias.fink@imperial.ac.uk)

    Args:
        project: project of interest
        shot: shot of interest

    Returns:
        stack object
    '''
    layers = build_layers(project, shot)

    stack_layers = [] # reformatting for Stack object

    for layer in layers:
        if layer[0] == "EBT3":
            stack_layers.extend(layer_EBT3())
            continue
        elif layer[0] == "HDV2":
            stack_layers.extend(layer_HDV2())
            continue
        else:
            layer_stopping_energy_power = linear_stopping_power(layer[0])

            stack_layers.append(Layer(int(layer[1])*units.um, *layer_stopping_energy_power,
                                      active=False))

    stack = Stack(stack_layers)

    print(f"Number of layers: {stack.num_layers}")
    print(f"Number of active layers: {stack.num_active}")
    print(f"Total stack thickness: {stack.thickness:.2f}")

    return stack

def calc_energy_bands(energy, deposition, normalise: bool, mode="frac-max",
                      output=False, ret_espread=False, test=False):
    '''
    Determine energy bands of deposition

    @author: Adam Dearling (add525@york.ac.uk)
    @edited: Elias Fink (elias.fink22@imperial.ac.uk)
    
    Args:
        energy:
        deposition:
        normalise:
        mode: points at which energy bands are defined. "half-max" is at the points
                corresponding to the half-maximum of the deposition curve, "half-energy"
                corresponds to the region in which half of the energy is contained around 
                the maximum.
        output: bool whether to return full output
        ret_espread: bool whether to return spread
        test: activate test mode

    Returns:
        energy bands
        energy band spread
    '''

    energy_bands = np.zeros([deposition.shape[0], 2])

    frac=1/np.e

    bragg_peak = energy[np.argmax(deposition, axis = 1)]

    if mode == "half-energy" and normalise:
        print("Data must be un-normalised to obtain half-energy bands. Reverting to half-max.")
        mode = "half-max"

    for i in range(deposition.shape[0]):

        if mode == "frac-max":
            bragg_curve = deposition[i, :]

            # Find the indices corresponding to half the maximum value
            # on either side of the peak
            fracmax = np.max(bragg_curve) * frac

            assert fracmax > 0, "Energy range is not suitable to obtain high energy deposition curves."

            inds = np.argwhere(bragg_curve > fracmax)
            # Store those energies

            energy_bands[i, 0] = energy[inds[0][0]]
            energy_bands[i, 1] = energy[inds[-1][0]]

        elif mode == "frac-energy":
            deposition_tot = np.sum(deposition[i,:])
            deposition_cum = np.cumsum(deposition[i,:])
            deposition_frac = deposition_cum/deposition_tot

            energy_bands[i, 0] = energy[np.nonzero(deposition[i,:])[0][0]]
            energy_bands[i, 1] = energy[np.argmin(abs(deposition_frac-frac))]

            if test:
                fig, ax = pm.plot_figure_axis()
                ax.plot(energy, deposition_frac)
                ax.axvline(energy_bands[i, 0])
                ax.axvline(energy_bands[i, 1])
                fig.tight_layout()

    energy_spread = (energy_bands[i,1]-energy_bands[i,0])/bragg_peak

    if output:
        for i in range(deposition.shape[0]):
            print("------")
            print(f"Layer {i+1} ({mode}):")
            print(f"Bragg peak = {bragg_peak[i]:.1f} MeV")
            print(f"Energy band = {energy_bands[i,0]:.1f}-{energy_bands[i,1]:.1f} MeV")
            print(f"Energy spread = {energy_spread[i]:.1%} %")
        print("------")

    if ret_espread:
        return energy_bands, energy_spread
    return energy_bands