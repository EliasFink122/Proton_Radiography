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
from scipy.constants import elementary_charge, proton_mass as e, m_p

def calc_active_layer_mass(dpi: int, material: str) -> float:
    '''
    Calculate the mass of the active layer in one pixel

    @author: Adam Dearling (add525@york.ac.uk)
    @edited: Elias Fink (elias.fink22@imperial.ac.uk)
    
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

def maxwellian_dist_1d(speed, n, T):
    '''
    Maxwellian distribution function in 1D

    Args:
        speed: kinetic energy in MeV
        n: number of protons
        T: temperature kB*T in MeV

    Returns:
        value of Maxwell-Boltzmann distribution
    '''
    dist = n * np.power(2*np.pi*T*e*1e6/m_p, -1/2) * np.exp(-speed/T)
    return dist

def maxwellian_prob_1d(speed, n, T):
    '''
    Maxwellian probability function in 1D
    
    Args:
        speed: kinetic energy in MeV
        n: number of protons
        T: temperature kB*T in MeV

    Returns:
        value of Maxwell-Boltzmann probability
    '''
    speed_in_joules = speed * 1e6 * e
    dist = maxwellian_dist_1d(speed, n, T)
    prob = (4 * np.pi * 1 / m_p) * np.sqrt(2 * speed_in_joules / m_p) * dist
    return prob

def log10_function(speed, n, T):
    '''
    Logarithm of Maxwellian distribution or probability

    Args:
        speed: kinetic energy in MeV
        n: number of protons
        T: temperature kB*T in MeV

    Returns:
        log10 of Maxwellian
    '''

    function = "probability"

    if function == "distribution":
        maxwellian = maxwellian_dist_1d(speed, n, T)
    elif function == "probability":
        maxwellian = maxwellian_prob_1d(speed, n, T)
    maxwellian_log10 = np.log10(maxwellian)

    return maxwellian_log10

def calc_dNdE_BPD(E, R, E1, E2, D):
    '''
    Calculate dNdE assuming energy deposition is dominated by the Bragg peak.

    Args:
        E, E1, E2 in MeV
        R, D (only current layer) in J
    
    D = int_0^inf(dNdE * R)dE is rearranged to give dNdE = D/int_E1^E2(S)dE
    with dNdE assumed to be constant in this area.
    
    Returns:
        dNdE value
    
    '''

    # Energy band over which integration should be performed
    E1_arg = np.argmin(abs(E-E1))
    E2_arg = np.argmin(abs(E-E2))

    # Simple Bragg peak dominated assumption
    int_R_x = E[E1_arg:E2_arg+1] * 1e6 * e
    int_R_y = R[E1_arg:E2_arg+1]
    int_R = np.trapz(int_R_y, x=int_R_x)

    dNdE = D / int_R

    return dNdE

def get_dNdE_spectrum_BPD(E, R_stack, dE_stack, D_stack, D_error=None):
    '''
    Get dNdE for a stack using the BPD assumption.

    Args:
        E: energy associated with response R
        R: response (deposition curve)
        dE: energy bands
        D: deposited energy

    Returns:
        dNdE values for stack
        dNdE error
    '''

    nlayers = len(D_stack)

    dNdE_stack = np.zeros(nlayers)

    # Iterate through layers
    for nlayer in range(nlayers):

        dNdE_stack[nlayer] = calc_dNdE_BPD(E, R_stack[nlayer], dE_stack[nlayer,0],
                                           dE_stack[nlayer,1], D_stack[nlayer])

    if D_error is not None:
        dNdE_error = dNdE_stack * (D_error/D_stack)
        return  dNdE_stack, dNdE_error
    return dNdE_stack

def interp_spectrum_deposition(E_in, R_in, bragg_in, dNdE_in, test=False):
    '''
    Find the energy deposited by the interpolated spectrum.
    
    E is the energy associated with R_i, the depostion curve for the ith layer.
    E_in are the energies associated with the portions of the spectrum in 
    dNdE_in, which is sliced between the ith and nth layers.
    
    E and E_in are in MeV, dNdE_in is in J.
    '''

    log10_dNdE_in = np.log10(dNdE_in) # J

    log10_dNdE_interp = np.interp(E_in, bragg_in, log10_dNdE_in)
    dNdE_interp = 10**(log10_dNdE_interp) # J

    if test:
        fig, ax = pm.plot_figure_axis()
        ax.scatter(bragg_in, dNdE_in)
        ax.plot(E_in, dNdE_interp)
        fig.tight_layout()

    E_dep_integ = dNdE_interp * R_in # J

    E_dep = np.trapz(E_dep_integ, x=E_in*1e6*e)

    return E_dep


if __name__ == "__main__":
    print(deposited_energy("Carroll_2023", "001"))
