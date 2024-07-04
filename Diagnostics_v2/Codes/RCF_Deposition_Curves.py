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
    calc_energy_bands:
        determine energy bands from stack
    get_deposition_curves:
        calculate deposition curves for relevant stack
    plot_stopping_power:
        plot stopping power vs energy for material
    plot_deposition curves:
        plot possible propagation through material with deposition marked
"""

import RCF_Dose as dose
import scipy.constants as const
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

def get_deposition_curves(project: str, shot: str, energy_range_MeV=[1,40],
                          dE = 0.00625, dx = 0.025, return_active=True, output_eband=False, plot=True):
    '''
    Get energy deposition curves for an RCF stack

    @author: Adam Dearling (add525@york.ac.uk)
    @edited: Elias Fink (elias.fink22@imperial.ac.uk)
    
    Args:
        dE: energy binning in MeV
        dx: spatial resolution in um
    
    Returns:
        deposition curves
        respective energies
    '''

    stack = build_stack(project, shot)

    energy = np.arange(*energy_range_MeV, dE) * units.MeV

    # Try/except in case modified plasmapy version is not used.
    deposition_curves = stack.deposition_curves(energy, dx = dx * units.um,
                                                return_only_active=return_active)

    energy = energy.value

    if plot:
        # Get energy bands - this was in Joules rather than eV (leads to plot bug) in plasmapy
        # ebands = stack.energy_bands(energy_range_MeV * u.MeV, dE * u.MeV, dx=dx * u.um,
        #                             return_only_active=return_active)
        ebands = calc_energy_bands(energy, deposition_curves, normalise = False,
                                   output = output_eband)

        path = dose.ROOTDIR + "/Data/" + project + "/Shot" + shot + "/RCF_Stack_Design.csv"
        stack_design = pd.read_csv(path, sep = ',')

        rcf_material = stack_design[:][0]

        plot_deposition_curves(stack, energy, deposition_curves, rcf_material = rcf_material,
                               ebands = ebands, normalise = False)

    return deposition_curves, energy

def plot_stopping_power(materials, nplot=1):
    '''
    Plot the stopping power as a function of proton energy.

    @author: Adam Dearling (add525@york.ac.uk)
    @edited: Elias Fink (elias.fink22@imperial.ac.uk)
    
    Args:
        materials: materials in stack
        nplot: number of plots
    '''

    if isinstance(materials, str):
        materials = [materials]
    elif not isinstance(materials, list):
        raise TypeError("Materials should be either desired string or list.")

    fig, ax = pm.plot_figure_axis("small", nplot)
    if nplot == 1:
        ax_SP = ax
        ax = [ax]
    else:
        ax_SP = ax[0]
        ax_SPrho = ax[1]

    for material in materials:
        stopping_power = get_mass_stopping_power(material)
        target_density = get_target_density(material) * units.g / units.cm**3

        ax_SP.plot(stopping_power[:,0], stopping_power[:,1], label=material)
        if nplot != 1:
            ax_SPrho.plot(stopping_power[:,0], stopping_power[:,1] * units.MeV * unts.cm**2 / units.g * target_density,
                          label=material)

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

def plot_deposition_curves(stack, energy, deposition_curves, rcf_material=None,
                           ebands=None, normalise=True):
    '''
    Plot deposition curves

    @author: Adam Dearling (add525@york.ac.uk)
    @edited: Elias Fink (elias.fink22@imperial.ac.uk)

    Args:
        stack: stack object
        energy: proton energy bands
        deposition_curves: deposition curves from above method
    '''

    fig, ax = pm.plot_figure_axis("small", 1, ratio=[1,1])

    if not normalise:
        deposition_curves = (deposition_curves / const.elementary_charge) / 1e6

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
