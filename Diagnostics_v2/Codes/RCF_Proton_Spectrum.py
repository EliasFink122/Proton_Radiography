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
    calc_dNdE_BPD:
        determine energy number product assuming all energy deposited in sharp Bragg peak
    get_dNdE_spectrum_BPD:
        use energy number product to determine spectrum of protons
        iterate through layers to get full spectrum
    interp_spectrum_deposition:
        interpolate spectrum for later fitting with Maxwellian
    get_dNdE_spectrum_iter:
        different approach to obtaining spectrum
        iteratively go from last layer to first to find contributions of different energies
    get_proton_spectrum:
        apply above mentioned methods to calculate the proton spectrum
    spectrum_fit:
        fit log10(Maxwellian) to log10(spectrum) to determine temperature and number of protons
    plot_spectrum:
        plot parameters of spectrum with fit on graph
        also plot respective deposition curves
"""

import RCF_Dose as dose
import RCF_Deposition_Curves as dc
import RCF_Plotting as pm
import scipy.optimize as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import elementary_charge as e

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
    imgs_converted = dose.convert_to_dose(project, shot)
    path = dose.ROOTDIR + "/Data/" + project + "/Shot" + shot + "/RCF_Stack_Design.csv"
    stack_design = pd.read_csv(path, sep = ',')

    deposited_energies = []
    for i, layer in enumerate(imgs_converted):
        dpi = 100
        mass = calc_active_layer_mass(dpi, stack_design[f"Layer {i+1}"][0])
        pixel_energies = layer * mass
        deposited_energies.append(np.sum(pixel_energies))

    return deposited_energies

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
        dNdE_guess = pm.maxwellian_prob_1d(E, N_n, T_n)

        # Energy coordinates that integral will be calculated between
        Ec_arg = np.argmin(abs(bragg_stack[-1]-E)) # Position of final bragg peak
        En_arg = np.argmin(abs(cutoff_n-E)) # Position of cutoff

        E_nc = E[En_arg:Ec_arg+1]
        R_nc = R_stack[-1, En_arg:Ec_arg+1]
        dNdE_nc = dNdE_guess[En_arg:Ec_arg+1]
        D_nc = interp_spectrum_deposition(E_nc, R_nc, E_nc, dNdE_nc)

        D_ratio = D_nc / D_stack[-1]

        print('D_RATIO', D_ratio)

        if D_ratio == 0:
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
                        tol=0.05, T_iter= None, cutoff_iter=None, plot=False):
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
        layer_numbers = [pm.letter_to_num(layer)-1 for layer in stack_layers]
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

    return stack_dNdE, stack_bragg_MeV, stack_fit

def spectrum_fit(energy_MeV, stack_bragg_MeV, stack_dNdE, error=False, plot=False,
                 test=False):
    '''
    Fit a spectrum to the dose data and returns the temperature.

    @author: Adam Dearling (add525@york.ac.uk)
    @edited: Elias Fink (elias.fink22@imperial.ac.uk)
    
    Args:
        log10_function: take inputs as MeV
    log10_function takes inputs in MeV, output should be in J units so stack_dNdE
    should be in J?
    '''

   # log10_function = np.nan_to_num(log10_function, nan=25)


    if error:
        stack_dNdE_error = stack_dNdE[1]
        stack_dNdE = stack_dNdE[0]

    guess = [1e12, 10]
    lower_bound = [-np.inf, -np.inf]
    upper_bound = [np.inf, np.inf]

    if test:
        fig, ax = pm.plot_figure_axis()
        ax.scatter(stack_bragg_MeV, stack_dNdE*e*1e6)
        ax.plot(energy_MeV, np.power(10, pm.log10_function(energy_MeV, *guess))*e*1e6)
        ax.set_xlabel("$E_\mathrm{k}$ (MeV)")
        ax.set_ylabel("$dN/dE$ (MeV$^{-1}$)")
        ax.set_xlim(xmin=0, xmax=np.round(stack_bragg_MeV[-1]+5,-1))
        ax.set_yscale("log")
        fig.tight_layout()

    popt, pcov = op.curve_fit(pm.log10_function, stack_bragg_MeV, np.log10(stack_dNdE*e*1e6),
                        p0=guess, bounds=(lower_bound, upper_bound), maxfev = 10000)
    pstd = np.diag(pcov)

    dNdE_fit = np.power(10, pm.log10_function(energy_MeV, *popt))

    # print(pcov)
    print(f"Temp = {popt[1]} +- {pstd[1]} MeV")
    print(f"Number = {popt[0]} +- {pstd[0]} particles")

    if plot:
        fig, ax = pm.plot_figure_axis()
        if error:
            ax.errorbar(stack_bragg_MeV, stack_dNdE*e*1e6,
                        yerr=stack_dNdE_error*e*1e6, fmt='x')
        else:
            ax.scatter(stack_bragg_MeV, stack_dNdE*e*1e6)
        ax.plot(energy_MeV, dNdE_fit*e*1e6)
        ax.set_xlabel("$E_\mathrm{k}$ (MeV)")
        ax.set_ylabel("$dN/dE$ (MeV$^{-1}$)")
        ax.set_xlim(xmin=0, xmax=np.round(stack_bragg_MeV[-1]+5,-1))
        ax.set_yscale("log")
        fig.tight_layout()

    return popt, pstd

def plot_spectrum(x, y, label=None, x_2=None, y_2=None, label_2=None,
                  x_line=None, y_line=None, y_line_fit=None, label_line=None,
                  y_line_2=None, y_line_fit_2=None, label_line_2=None,
                  y_unit="perMeV"):
    '''
    Plot the proton spectrum.
    By default x will be in MeV, while y will be in per J (laugh if not).

    @author: Adam Dearling (add525@york.ac.uk)
    @edited: Elias Fink (elias.fink22@imperial.ac.uk)
    
    Args:
        x, y, x_2, y_2: input data
        label, label_2: labels for data
        y_line, y_line_2, y_line_fit_2: lines for Maxwellian distribution
        y_unit: unit of y axis
    '''

    cmap = plt.get_cmap("tab10")

    if y_unit == "perMeV":
        scale = e*1e6
    elif y_unit == "perJ":
        scale = 1

    if y_line_fit is not None:
        y_line = pm.maxwellian_prob_1d(x_line, *y_line_fit[0])
        y_line_p = pm.maxwellian_prob_1d(x_line, y_line_fit[0][0],
                                          y_line_fit[0][1]+y_line_fit[1][1])
        y_line_m = pm.maxwellian_prob_1d(x_line, y_line_fit[0][0],
                                          y_line_fit[0][1]-y_line_fit[1][1])
    if y_line_fit_2 is not None:
        y_line_2 = pm.maxwellian_prob_1d(x_line, *y_line_fit_2[0])
        y_line_p_2 = pm.maxwellian_prob_1d(x_line, y_line_fit_2[0][0],
                                            y_line_fit_2[0][1]+y_line_fit_2[1][1])
        y_line_m_2 = pm.maxwellian_prob_1d(x_line, y_line_fit_2[0][0],
                                            y_line_fit_2[0][1]-y_line_fit_2[1][1])

    fig, ax = pm.plot_figure_axis()

    ax.scatter(x, y*1e6, label=label, s=50) # y*scale
    if y_2 is not None and x_2 is not None:
        ax.scatter(x_2, y_2*1e6, label=label_2, marker="x", s=50) # y*scale
    if x_line is not None and y_line is not None:
        ax.plot(x_line, y_line*scale, label=label_line, color="k", linestyle="--", zorder=0)
        if y_line_fit is not None:
            ax.fill_between(x_line, y_line_m*scale, y_line_p*scale, color=cmap(0), alpha=0.5)
    if x_line is not None and y_line_2 is not None:
        ax.plot(x_line, y_line_2*scale, label=label_line, color="k", linestyle="--", zorder=0)
        if y_line_fit is not None:
            ax.fill_between(x_line, y_line_m_2*scale, y_line_p_2*scale, color=cmap(1), alpha=0.5)

    ax.set_xlabel("$E_\mathrm{k}$ (MeV)")
    if y_unit == "perMeV":
        ax.set_ylabel("$dN/dE$ (MeV$^{-1}$)")
    elif y_unit == "perJ":
        ax.set_xlabel("$dN/dE$ (MeV$^{-1}$)")

    ax.set_xlim(xmin=0, xmax=np.round(x[-1]+5,-1))

    ax.set_yscale("log")

    ax.set_title("shot" + str(SHOT))
    if label is not None:
        ax.legend()

    fig.tight_layout()


if __name__ == "__main__":
    PROJECT = "Carroll_2023"

    if PROJECT == "Carroll_2023":
        e_range = [1,120]
        DESIGN = None
        SHOT = "001"
        STACK = "18"
        layers = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
                  "J", "K", "L", "M", "N", "O", "P", "Q"] # 17 layers
        SUFFIX = None
        edge = [100,20]
        SCANNER = "Epson_12000XL"
        MATERIAL_TYPE = None
        OD = False
        CLEAN = True
        channels = [1,1,1]
        DPI = 300

    deposition_curves, deposition_energy = dc.get_deposition_curves(energy_range_MeV=e_range,
                                                            project=PROJECT, shot=SHOT,
                                                            dE=0.00625, dx=0.25,
                                                            plot=True)

    stack_dose = dose.convert_to_dose(PROJECT, SHOT) # Dose in Grays

    stack_energy = deposited_energy(PROJECT, SHOT) # Energy in J, tuple including error

    stack_energy_total = np.array([np.sum(layer_energy) for layer_energy in stack_energy]) # Energy in J
    # stack_error_total = np.array([np.sum(layer_energy[1]) for layer_energy in stack_energy])

    BPD_dNdE, stack_bragg, BPD_fit = get_proton_spectrum(stack_energy_total, deposition_curves,
                                                        deposition_energy, stack_layers=layers, method = "BPD")#,
                                                        # stack_energy_error=stack_error_total)

    iter_dNdE, _, iter_fit = get_proton_spectrum(stack_energy_total, deposition_curves,
                                                deposition_energy, stack_layers=layers,
                                                method="iter", T_iter= BPD_fit[0][0], cutoff_iter = 59) #T_iter=6.754, cutoff_iter=23.887)
                                                #T_iter=11.779, cutoff_iter=53.7813)

    plot_spectrum(stack_bragg, BPD_dNdE, label="BPD", x_2=stack_bragg, y_2=iter_dNdE,
                label_2="iter.", x_line=deposition_energy, y_line_fit=BPD_fit,
                y_line_fit_2=iter_fit)

    plt.show()
