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
import RCF_Deposition_Curves as dc
import RCF_Plotting as pm
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

def maxwellian_dist_1d(speed, n, T) -> float:
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

def maxwellian_prob_1d(speed, n, T) -> float:
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

def log10_function(speed, n, T) -> float:
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

def calc_dNdE_BPD(E, R, E1, E2, D) -> float:
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

def get_dNdE_spectrum_BPD(E, R_stack, dE_stack, D_stack, D_error=None) -> list[float]:
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

def interp_spectrum_deposition(E_in, R_in, bragg_in, dNdE_in, test=False) -> float:
    '''
    Find the energy deposited by the interpolated spectrum.

    Args:
        E_in: energies associated with portions of spectrum in dNdE in MeV
        dNdE_in: dNdE sliced between ith and nth layer in J
        R_i: deposition curve for ith layer
    '''

    log10_dNdE_in = np.log10(dNdE_in) # J

    log10_dNdE_interp = np.interp(E_in, bragg_in, log10_dNdE_in)
    dNdE_interp = np.power(10, log10_dNdE_interp) # J

    if test:
        fig, ax = pm.plot_figure_axis()
        ax.scatter(bragg_in, dNdE_in)
        ax.plot(E_in, dNdE_interp)
        fig.tight_layout()

    E_dep_integ = dNdE_interp * R_in # J

    E_dep = np.trapz(E_dep_integ, x=E_in*1e6*e)

    return E_dep

def get_dNdE_spectrum_iter(E, R_stack, dE_stack, D_stack, tol=0.05,
                           itmax=100, N_n=1e12, T_n=None, cutoff_n=None):
    '''
    Get dNdE for each stack layer iteratively
    Starts from the last layer and works forward to get the contribution of
    higher energy particles in the first layer. Makes the BPD assumption for
    the last layer in the stack.
    
    @author: Adam Dearling (add525@york.ac.uk)
    '''

    nlayers = len(D_stack)

    dNdE_stack = np.zeros(nlayers)

    bragg_stack = E[np.argmax(R_stack, axis=1)]

    # Calculate dN/dE for last layer using the BPD assumption, otherwise guess initial spectrum
    if T_n is None and cutoff_n is None:
        print("Staring with BPD assumption.")
        dNdE_stack[-1] = calc_dNdE_BPD(E, R_stack[-1], dE_stack[-1,0], dE_stack[-1,1],
                                       D_stack[-1])
    else:
        dNdE_guess = maxwellian_prob_1d(E, N_n, T_n)

        # Energy coordinates that integral will be calculated between
        En_arg = np.argmin(abs(bragg_stack[-1]-E)) # Position of final bragg peak
        Ec_arg = np.argmin(abs(cutoff_n-E)) # Position of cutoff

        E_nc = E[En_arg:Ec_arg+1]
        R_nc = R_stack[-1, En_arg:Ec_arg+1]
        dNdE_nc = dNdE_guess[En_arg:Ec_arg+1]

        D_nc = interp_spectrum_deposition(E_nc, R_nc, E_nc, dNdE_nc)

        D_ratio = D_nc / D_stack[-1]

        print('D_RATIO', D_ratio)

        if D_ratio == 0 :
            D_ratio = 1

        its = 0
        # Iterate until corrected spectrum is found
        while abs(D_ratio-1)>tol and its<itmax:

            dNdE_nc = dNdE_nc * 1/D_ratio

            D_nc = interp_spectrum_deposition(E_nc, R_nc, E_nc, dNdE_nc)

            D_ratio = D_nc / D_stack[-1]

            its +=1

        dNdE_stack[-1] = dNdE_nc[0]

        # Append inferred cutoff values
        dNdE_stack = np.append(dNdE_stack, dNdE_nc[-1])
        bragg_stack = np.append(bragg_stack, E_nc[-1])

    # Begin iteration, starting from the penultimate layer
    for i in np.arange(nlayers-2,-1,-1): # End should be -1

        # Initial prediction for layer spectrum at bragg peak
        dNdE_stack[i] = calc_dNdE_BPD(E, R_stack[i], dE_stack[i,0], dE_stack[i,1],
                                      D_stack[i])

        # Energy coordinates that integral will be calculated between
        Ei_arg = np.argmin(abs(bragg_stack[i]-E)) # Position of current bragg peak
        En_arg = np.argmin(abs(bragg_stack[-1]-E)) # Position of final bragg peak

        E_in = E[Ei_arg:En_arg+1] # MeV
        R_in = R_stack[i, Ei_arg:En_arg+1]
        bragg_in = bragg_stack[i:] # MeV
        dNdE_in = dNdE_stack[i:]

        # Initial energy deposition calculation
        D_iter = interp_spectrum_deposition(E_in, R_in, bragg_in, dNdE_in, test=False)

        D_ratio = D_iter / D_stack[i]

        its = 0
        # Iterate until corrected spectrum is found
        while abs(D_ratio-1)>tol and its<itmax:

            dNdE_in[0] = dNdE_in[0] * 1/D_ratio

            D_iter = interp_spectrum_deposition(E_in, R_in, bragg_in, dNdE_in)

            D_ratio = D_iter / D_stack[i]

            its +=1

        dNdE_stack[i] = dNdE_in[0]

    # Remove inferred value for last layer
    if not (T_n is None and cutoff_n is None):
        dNdE_stack = dNdE_stack[:-1]

    return dNdE_stack

def get_proton_spectrum(stack_energy, deposition_curves, deposition_energy_MeV,
                        stack_layers=None, stack_energy_error=None, method="BPD",
                        tol=0.05, T_iter= None , cutoff_iter=None, plot=False):
    '''
    Calculate the proton spectrum using the Bragg peak dominated method.

    @author: Adam Dearling (add525@york.ac.uk)
    
    Args:
        stack_energy: in J
        deposition_curves: in J
        deposition energy: in MeV

    Returns:
        proton spectrum data
    '''

    error = bool(stack_energy_error is not None)

    # Get the energy bands
    deposition_ebands_MeV = dc.calc_energy_bands(deposition_energy_MeV, deposition_curves,
                                              normalise=False, output=False)

    bragg_peak_MeV = deposition_energy_MeV[np.argmax(deposition_curves, axis=1)]

    # Get list of layer numbers, so correct curves can be extracted from the stack if required
    if stack_layers is not None:
        layer_numbers = [rcf.letter_to_num(layer)-1 for layer in stack_layers]
        stack_bragg_MeV = bragg_peak_MeV[layer_numbers]
        stack_deposition = deposition_curves[layer_numbers, :]
        stack_ebands_MeV = deposition_ebands_MeV[layer_numbers, :]
    else:
        stack_bragg_MeV = bragg_peak_MeV
        stack_deposition = deposition_curves
        stack_ebands_MeV = deposition_ebands_MeV

    if method == "BPD":
        stack_dNdE = get_dNdE_spectrum_BPD(deposition_energy_MeV, stack_deposition,
                                           stack_ebands_MeV, stack_energy,
                                           D_error=stack_energy_error)
    elif method == "iter":
        stack_dNdE = get_dNdE_spectrum_iter(deposition_energy_MeV, stack_deposition,
                                            stack_ebands_MeV, stack_energy,
                                            tol=tol, T_n=T_iter, cutoff_n=cutoff_iter)

    stack_fit = spectrum_fit(deposition_energy_MeV, stack_bragg_MeV, stack_dNdE,
                             error=error, plot=plot)

    # print(stack_fit)

    return stack_dNdE, stack_bragg_MeV, stack_fit


if __name__ == "__main__":
    print(deposited_energy("Carroll_2023", "001"))
