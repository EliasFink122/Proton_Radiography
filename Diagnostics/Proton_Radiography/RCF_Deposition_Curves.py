# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:53:22 2023

@author: add525

Utilises the PlasmaPy library to generate energy deposition curves.
"""

#%% Libraries

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# from tempfile import gettempdir

import astropy.units as u
from plasmapy.diagnostics.charged_particle_radiography.detector_stacks import (
    Layer,
    Stack,
)
# from plasmapy.utils.data.downloader import get_file

import RCF_Basic as rcf

rootdir = r"/Users/eliasfink/Desktop/Proton_Radiography"
sys.path.insert(1, rootdir + '/Codes/Python_Scripts/')

import Physics_Constants as pc
import Plot_Master as pm


#%% Functions

def get_mass_stopping_power(material, database="SRIM"):
    '''Get stopping power data from file'''
    # PlasmaPy was setup to work with PSTAR files, which have units of MeV/(g/cm2)
    # so I will convert SRIM data (which is in MeV/(mg/cm2)) here for simplicity.
    # Additionally, SRIM outputs electronic and nuclear stopping power separtely.
    # PlasmaPy wants the total stopping power, so we will combine here.

    owd = os.getcwd()

    os.chdir(rootdir)
    os.chdir("Diagnostics/Proton_Radiography/Stopping_Power")

    file = database + "_" + material + ".txt"
    if file not in os.listdir():
        raise FileNotFoundError(f"{file} is not in the Stopping_Power directory.")

    if database == "PSTAR":
        data = np.loadtxt(file, skiprows=8)

    elif database == "SRIM": # Have to manually handle formatting
        with open(file) as f:
            data_raw = f.readlines()
            str_start = '-----------  ---------- ---------- ----------  ----------  ----------\n'
            str_end = '-----------------------------------------------------------\n'
            try:
                row_start = data_raw.index(str_start)
            except:
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

    os.chdir(owd)

    return data


def get_target_density(material, database="SRIM"):
    '''
    Get the target density in g/cm3.
    
    Currently this is only available in SRIM files?
    '''

    owd = os.getcwd()

    os.chdir(rootdir)
    os.chdir("Diagnostics/Proton_Radiography/Stopping_Power")

    file = database + "_" + material + ".txt"
    if file not in os.listdir():
        raise FileNotFoundError(f"{file} is not in the Stopping_Power directory.")

    if database == "PSTAR":
        raise Exception("PSTAR files do not contain target density.")

    elif database == "SRIM": # Have to manually handle formatting
        with open(file) as f:
            for row in f.readlines():
                if "Target Density" in row:
                    target_density = float(row.split()[3])
                    break

    os.chdir(owd)

    return target_density


def linear_stopping_power(material):
    '''Get the linear stopping power, as SRIM/PSTAR data is for mass stopping power'''

    density = get_target_density(material) * u.g / u.cm**3
    stopping_mass = get_mass_stopping_power(material)
    stopping_power = stopping_mass[:,1] * u.MeV * u.cm**2 / u.g * density # Convert stopping power to astropy units
    stopping_energy = stopping_mass[:,0] * u.MeV # Convert corresponding energy to astropy units

    return stopping_energy, stopping_power


def layer_EBT3():
    '''
    Define the EBT3 layer
    
    Consists of "Polyester_Substrate" and "EBT2_Active_Layer"
    '''

    active_layer = "EBT2_Active_Layer"
    substrate ="Polyester_Substrate"

    act_stopping_energy_power = linear_stopping_power(active_layer)
    sub_stopping_energy_power = linear_stopping_power(substrate)

    EBT3 = [
        Layer(125 * u.um, *sub_stopping_energy_power, active=False),
        Layer(28 * u.um, *act_stopping_energy_power, active=True),
        Layer(125 * u.um, *sub_stopping_energy_power, active=False),
    ]

    return EBT3

def layer_HDV2():
    '''
    Define the HDV2 layer
    
    Consists of "Polyester_Substrate" and "HDV2_Active_Layer"
    '''

    active_layer = "HDV2_Active_Layer"
    substrate ="Polyester_Substrate"

    act_stopping_energy_power = linear_stopping_power(active_layer)
    sub_stopping_energy_power = linear_stopping_power(substrate)

    HDV2 = [
        Layer(12 * u.um, *act_stopping_energy_power, active=True),
        Layer(97 * u.um, *sub_stopping_energy_power, active=False),
    ]

    return HDV2


def build_layers(input_layers=None, project=None, shot=None, design=None):
    '''
    Build the layers of an RCF stack with a given design.
    
    input_layers: a tuple of two list where the first contains the stack materials
        (e.g. HDV2, EBT3) and the second contains a list of lists with the filter
        material and its thickness. Multiple filters can be listed for a single layer
        by placing a "/" between the material and its thickness.
    '''

    if input_layers is not None:
        print("Building stack with the following layers:")
        print(input_layers)
        stack_material = input_layers[0]
        stack_filters = input_layers[1]
    elif project is not None:
        stack_material = rcf.get_stack_design(project, shot=shot, design=design,
                                              info="material")
        stack_filters = rcf.get_stack_design(project, shot=shot, design=design,
                                             info="filters")
    else:
        raise RuntimeError("Either layers or project must be input.")

    layers = []

    # The first layer of the stack should be a filter
    for i, material in enumerate(stack_material):
        if material is np.nan:
            print(f"{i+1} RCF layers in stack.")
            break

        if stack_filters[0][i].count("/")>0: # Are multiple filters listed
            n_filters = stack_filters[0][i].count("/")+1
            layer_filters = stack_filters[0][i].split("/")
            layer_filters_thickness = stack_filters[1][i].split("/")
            for j in range(n_filters):
                layers.append([layer_filters[j], layer_filters_thickness[j]])
        else:
            layers.append([stack_filters[0][i], stack_filters[1][i]])
        layers.append([material])

    return layers


def build_stack(input_layers=None, project=None, shot=None, design=None):
    '''Build and RCF stack with a given design.'''

    layers = build_layers(input_layers=input_layers, project=project, shot=shot, 
                          design=design) # Returns a list containg the stack specs.
    # print(layers)

    stack_layers = [] # The "layers" list formatted so that the stack class can be constructed.

    for layer in layers:
        if layer[0] == "EBT3":
            stack_layers.extend(layer_EBT3())
            continue
        elif layer[0] == "HDV2":
            stack_layers.extend(layer_HDV2())
            continue
        else:
            layer_stopping_energy_power = linear_stopping_power(layer[0])

            stack_layers.append(Layer(int(layer[1])*u.um, *layer_stopping_energy_power,
                                      active=False))

    stack = Stack(stack_layers)

    print(f"Number of layers: {stack.num_layers}")
    print(f"Number of active layers: {stack.num_active}")
    print(f"Total stack thickness: {stack.thickness:.2f}")

    return stack


def calc_energy_bands(energy, deposition, normalise, mode="frac-max", frac=1/np.e,
                      output=False, ret_espread=False, test=False):
    '''
    "Improved" energy banding protocall.
    
    mode: points at which energy bands are defined. "half-max" is at the points
        corresponding to the half-maximum of the deposition curve, "half-energy"
        corresponds to the region in which half of the energy is contained around 
        the maximum.
    '''

    energy_bands = np.zeros([deposition.shape[0], 2])

    bragg_peak = energy[np.argmax(deposition, axis=1)]

    if mode == "half-energy" and normalise == True:
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
    else:
        return energy_bands


def get_deposition_curves(energy_range_MeV=[1,40], input_layers=None, project=None,
                          shot=None, design=None, dE = 0.00625, dx = 0.025,
                          normalise=False, normalise_type=None, return_active=True, 
                          output_eband=False, plot=True):
    '''
    Get energy deposition curves for an RCF stack
    I have added a "normalise" optional argument to deposition_curves function.
    This just requires adding an if statement surrounding the normalisition step
    in the plasmapy detector_stacks module - I have added an exception to handle
    this otherwise.
    
    dE: energy binning in MeV
    dx: spatial resolution in um (I believe)
    normalise: if True return the energy deposited in the layer as a fraction
        of the total energy deposited by that energy. if False return the amount
        of energy lost by that energy in that layer.
    '''

    stack = build_stack(input_layers=input_layers, project=project, shot=shot,
                        design=design)

    energy = np.arange(*energy_range_MeV, dE) * u.MeV

    # Try/except in case modified plasmapy version is not used.
    try:
        deposition_curves = stack.deposition_curves(energy, dx=dx * u.um, 
                                                    return_only_active=return_active,
                                                    normalise=normalise,
                                                    normalise_type=normalise_type)
    except:
        deposition_curves = stack.deposition_curves(energy, dx=dx * u.um, 
                                                    return_only_active=return_active)
        normalise = True

    energy = energy.value

    if plot:
        # Get energy bands - this was in Joules rather than eV (leads to plot bug) in plasmapy
        # ebands = stack.energy_bands(energy_range_MeV * u.MeV, dE * u.MeV, dx=dx * u.um, 
        #                             return_only_active=return_active)
        ebands = calc_energy_bands(energy, deposition_curves, normalise,
                                   frac=1/np.e, output=output_eband)

        if input_layers is None:
            rcf_material = rcf.get_stack_design(project, shot=shot, design=design,
                                                info="material")
        else:
            rcf_material = input_layers[0]

        plot_deposition_curves(stack, energy, deposition_curves, rcf_material=rcf_material,
                               ebands=ebands, normalise=normalise)

    return deposition_curves, energy


#%%  Plotting functions

def plot_stopping_power(materials, nplot=1):
    '''Plot the stopping power as a function of proton energy.'''

    if isinstance(materials, str):
        materials = [materials]
    elif not isinstance(materials, list):
        raise RuntimeError("Materials should be either desired string or list.")        

    fig, ax = pm.plot_figure_axis("small", nplot)
    if nplot == 1:
        ax_SP = ax
        ax = [ax]
    else:
        ax_SP = ax[0]
        ax_SPrho = ax[1]

    for material in materials:
        stopping_power = get_mass_stopping_power(material)
        target_density = get_target_density(material) * u.g / u.cm**3

        ax_SP.plot(stopping_power[:,0], stopping_power[:,1], label=material)
        if nplot != 1:
            ax_SPrho.plot(stopping_power[:,0], stopping_power[:,1] * u.MeV * u.cm**2 / u.g * target_density, label=material)

    ax_SP.set_ylabel(r"Stopping Power (MeV cm$^2$ g$^{-1}$)")
    if nplot != 1:
        ax_SPrho.set_ylabel(r"Stopping Power (MeV cm$^{-1}$)")
    for axi in ax:
        axi.set_xlabel(r"E$_\mathrm{k}$ (MeV)")
        axi.set_yscale("log")
        axi.set_xscale("log")
        axi.set_xlim(xmin=1e-2)
    ax[-1].legend()

    fig.tight_layout()

    return


def plot_deposition_curves(stack, energy, deposition_curves, rcf_material=None,
                           ebands=None, normalise=True):
    '''Plot deposition curves'''

    fig, ax = pm.plot_figure_axis("small", 1, ratio=[1,1])

    if not normalise:
        deposition_curves = (deposition_curves / pc.e) / 1e6

    plot_max = round(np.amax(deposition_curves),
                     -int(np.floor(np.log10(abs(np.amax(deposition_curves))))))*1.2

    for layer in range(stack.num_active):
        label = f"Layer {layer+1}"
        if rcf_material is not None:
            label = label + f" ({rcf_material[layer]})"

        ax.plot(energy, deposition_curves[layer, :], label=label)

        if ebands is not None:
            ax.fill_betweenx([0,plot_max], ebands[layer,0], ebands[layer,1], alpha=0.25)

    # ax.set_title("Energy deposition curves")
    ax.set_xlabel("$E_\mathrm{k}$ (MeV)")
    if normalise:
        ax.set_ylabel("Normalized energy deposition curve")
    else:
        ax.set_ylabel("Energy deposition curve (MeV)")
    ax.set_xlim(xmin=0, xmax=max(energy))
    ax.set_ylim(ymin=0, ymax=plot_max)
    ax.legend()

    fig.tight_layout()

    return

#%%  Main script
if __name__ == "__main__":

    project = "Carroll_2020"
    design = "A"

    # project = "Woolsey_2019"
    # design = "C"

    if 0:
        stack = build_stack(project=project, design=design)

    if 1:
        deposition_curves, energy = get_deposition_curves(energy_range_MeV=[1,120], project=project, design=design,
                                                          normalise=False, dE=0.625, dx=0.25, output_eband=True)

    if 0:
        plot_stopping_power(["Al", "Fe", "Mylar"], nplot=1)

plt.show()
