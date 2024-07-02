# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Adam Dearling (add525@york.ac.uk)

Calculates a proton spectrum from RCF.

The fitting 
"""

#%% Libraries

import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import RCF_Basic as rcf
from RCF_Dose import get_dose_data
from RCF_Deposition_Curves import get_deposition_curves, calc_energy_bands
# from RCF_Calibration import gaussian_fit, calc_average_circle

sys.path.insert(1, '../../Codes/Python Scripts/')
import scipy.constants as pc
import Plot_Master as pm


#%% Functions

def get_stack_materials(project, shot, layers):
    '''Get the materials in a stack'''

    stack_material_all = rcf.get_stack_design(project, shot, info="material")

    stack_layers = [rcf.letter_to_num(layer)-1 for layer in layers]

    stack_material = [stack_material_all[layer] for layer in stack_layers]

    return stack_material


def calc_active_layer_mass(dpi, material):
    '''Calculate the mass of the active layer in one pixel'''

    px_size_m = rcf.convert_dpi_to_m(dpi, units="m")
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


def get_deposited_energy(dose, dpi, material, dose_error=None):
    '''Convert dose (Gray) to energy (J) as Energy = Dose * Mass'''

    active_layer_mass = calc_active_layer_mass(dpi, material) # in kg
    energy = dose*active_layer_mass # in J

    if dose_error is not None:
        energy_error = energy * dose_error/dose

    if dose_error is not None:
        return energy, energy_error
    else:
        return energy


def maxwellian_distribution(velocity_MeV, n_perm3, T_MeV):
    '''
    Maxwellian distribution function.
    
    f_m(v) = (n / [pi^1.5 * v_T^3]) * exp(-v^2 / v_T^2)
    f_m(E) = (n / [pi^1.5 * (2 E_T/m)^1.5]) * exp(-E / E_T)
    '''

    exponent = - velocity_MeV / T_MeV

    amplitude = n_perm3 / ((2*np.pi * T_MeV * 1e6 * pc.e/ pc.m_p)**(1/2))

    oned_amplitude = (2*np.pi*T_MeV*pc.e*1e6/pc.m_p)**(-1/2)

    #amplitude = n_perm3 / np.power(2 * np.pi * T_MeV * 1e6 * pc.e / pc.m_p, 3/2) #3/2


    maxwellian_f = n_perm3* oned_amplitude * np.exp(exponent)

   # maxwellian_f = amplitude * np.exp(exponent)

    return maxwellian_f


def maxwellian_probability(velocity_MeV, N, T_MeV):
    '''
    Maxwellian probability function dN/dE.
    
    dN/dv = 4 * pi * V * f_m * v^2
    dN/dE = 4 * pi * V * f_m * sqrt(2 * E / m) / m
    
    f_m propto n, so if V = 1 n becomes N
    '''   

    velocity_J = velocity_MeV * 1e6 * pc.e

    maxwellian_f = maxwellian_distribution(velocity_MeV, N, T_MeV)

    maxwellian_p = (4 * np.pi * 1 / pc.m_p) * np.sqrt(2 * velocity_J / pc.m_p) * maxwellian_f

    return maxwellian_p


def log10_function(velocity_MeV, n_perm3, T_MeV):
    '''Function should be fit on a log plot.'''

    function = "probability"

    if function == "distribution":    
        maxwellian = maxwellian_distribution(velocity_MeV, n_perm3, T_MeV)

    elif function == "probability":
        V = 1 # n_perm3 then becomes N
        maxwellian = maxwellian_probability(velocity_MeV, n_perm3, T_MeV)

    maxwellian_log10 = np.log10(maxwellian)

    # check for nans within the log10 function - i think this is what is causing the problems with the 1D maxwellian.

    maxwellian_log10 = np.nan_to_num(maxwellian_log10, nan= 39.07232309)

    print(maxwellian_log10)

    return maxwellian_log10


def calc_dNdE_BPD(E, R, E1, E2, D):
    '''
    Calculate dNdE assuming energy deposition is dominated by the Bragg peak.
    
    D = int_0^inf(dNdE * R)dE is rearranged to give dNdE = D/int_E1^E2(S)dE
    with dNdE assumed to be constant in this area.
    
    E, E1, E2 in MeV. R, D in J. For D only the current layer should be provided.
    '''

    # Energy band over which integration should be performed
    E1_arg = np.argmin(abs(E-E1))
    E2_arg = np.argmin(abs(E-E2))

    # Simple Bragg peak dominated assumption
    int_R_x = E[E1_arg:E2_arg+1] * 1e6 * pc.e
    int_R_y = R[E1_arg:E2_arg+1]
    int_R = np.trapz(int_R_y, x=int_R_x)

    dNdE = D / int_R 

    return dNdE


def get_dNdE_spectrum_BPD(E, R_stack, dE_stack, D_stack, D_error=None):
    '''
    Get dNdE for a stack using the BPD assumption.
    
    E is the energy associated with R, the deposition curves. dE is the energy
    bands. D is the deposited energy.
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

    E_dep = np.trapz(E_dep_integ, x=E_in*1e6*pc.e)

    return E_dep


def get_dNdE_spectrum_iter(E, R_stack, dE_stack, D_stack, D_error=None, tol=0.05, 
                           itmax=100, N_n=1e12, T_n=None, cutoff_n=None):
    '''
    Get dNdE for a stack iteratively.
    
    Starts from the last layer and works forward to get the contribution of
    higher energy particles in the first layer. Makes the BPD assumption for
    the last layer in the stack.
    
    Code is insensitive to choice of N_n.
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
        dNdE_guess = maxwellian_probability(E, N_n, T_n)

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
        # bragg_in = np.array([bragg_stack[i], bragg_stack[-1]]) # MeV
        # dNdE_in = np.array([dNdE_stack[i], dNdE_stack[-1]])

        # Initial energy deposition calculation
        D_iter = interp_spectrum_deposition(E_in, R_in, bragg_in, dNdE_in, test=False)

        D_ratio = D_iter / D_stack[i]

        its = 0
        # Iterate until corrected spectrum is found
        while abs(D_ratio-1)>tol and its<itmax:

            dNdE_in[0] = dNdE_in[0] * 1/D_ratio

            D_iter = interp_spectrum_deposition(E_in, R_in, bragg_in, dNdE_in)

            D_ratio = D_iter / D_stack[i]

            # if abs(D_ratio-1)<tol:
            #     print("Energy deposition convergence reached.")

            its +=1

        dNdE_stack[i] = dNdE_in[0]

        # print("Ecalc/Etotal = {:.3f}.".format(D_ratio))
        # print("Iterations = {}.".format(its))

    # Remove inferred value for last layer
    if not (T_n is None and cutoff_n is None):
        dNdE_stack = dNdE_stack[:-1]

    return dNdE_stack


def get_proton_spectrum(stack_energy, deposition_curves, deposition_energy_MeV,
                        stack_layers=None, stack_energy_error=None, method="BPD",
                        tol=0.05, T_iter= None , cutoff_iter=None, plot=False):
    '''
    Calculate the proton spectrum using the Bragg peak dominated method.
    
    stack_energy should be provided in J, deposition_curves in J, and deposition
    energy in MeV...
    '''

    error = bool(stack_energy_error is not None)

    # Get the energy bands
    deposition_ebands_MeV = calc_energy_bands(deposition_energy_MeV, deposition_curves, 
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
                                            D_error=stack_energy_error, tol=tol,
                                            T_n=T_iter, cutoff_n=cutoff_iter)

    stack_fit = spectrum_fit(deposition_energy_MeV, stack_bragg_MeV, stack_dNdE,
                             error=error, plot=plot)

    # print(stack_fit)

    return stack_dNdE, stack_bragg_MeV, stack_fit


def spectrum_fit(energy_MeV, stack_bragg_MeV, stack_dNdE, error=False, plot=False,
                 test=False):
    '''
    Fit a spectrum to the dose data and returns the temperature.
    
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
        ax.scatter(stack_bragg_MeV, stack_dNdE*pc.e*1e6)
        ax.plot(energy_MeV, 10**log10_function(energy_MeV, *guess)*pc.e*1e6)
        ax.set_xlabel("$E_\mathrm{k}$ (MeV)")
        ax.set_ylabel("$dN/dE$ (MeV$^{-1}$)")
        ax.set_xlim(xmin=0, xmax=np.round(stack_bragg_MeV[-1]+5,-1))
        ax.set_yscale("log")
        fig.tight_layout()

    popt, pcov = curve_fit(log10_function, stack_bragg_MeV, np.log10(stack_dNdE),
                           p0=guess, bounds=(lower_bound, upper_bound))
    pstd = np.diag(pcov)

    dNdE_fit = 10**log10_function(energy_MeV, *popt)

    # print(pcov)
    print(f"Temp = {popt[1]} +- {pstd[1]} MeV")
    print(f"Number = {popt[0]} +- {pstd[0]} particles")

    if plot:
        fig, ax = pm.plot_figure_axis()
        if error:
            ax.errorbar(stack_bragg_MeV, stack_dNdE*pc.e*1e6,
                        yerr=stack_dNdE_error*pc.e*1e6, fmt='x')
        else:
            ax.scatter(stack_bragg_MeV, stack_dNdE*pc.e*1e6)
        ax.plot(energy_MeV, dNdE_fit*pc.e*1e6)
        ax.set_xlabel("$E_\mathrm{k}$ (MeV)")
        ax.set_ylabel("$dN/dE$ (MeV$^{-1}$)")
        ax.set_xlim(xmin=0, xmax=np.round(stack_bragg_MeV[-1]+5,-1))
        ax.set_yscale("log")
        fig.tight_layout()

    return popt, pstd


def test_get_proton_spectrum(E, R_stack, T_MeV=10, cutoff=None, tol=0.05,
                             T_iter=None, cutoff_iter=None, noise=None):
    '''
    Test the BPD and iterative methods for getting the proton spectrum.
    
    E is the energy corresponding to the response curves R. R is in J and E is 
    in MeV.
    
    cutoff: specify the distribution cutoff energy in MeV
    '''

    # Initalise test spectrum and delivered dose
    test_dNdE = maxwellian_probability(E, 1e12, T_MeV) # per J
    if noise is not None:
        test_dNdE += np.random.normal(0,noise,len(E))
    if cutoff is not None:
        cutoff_arg = np.argmin(abs(E-cutoff))
        test_dNdE[cutoff_arg:] = 0

    nlayers = R_stack.shape[0]

    test_D = np.zeros(nlayers)
    for nlayer in range(nlayers):    
        test_D_int = test_dNdE * R_stack[nlayer,:] # per J * J so unitless
        test_D[nlayer] = np.trapz(test_D_int, E)*1e6*pc.e # J

    # BPD solution
    BPD_dNdE, stack_bragg, BPD_fit = get_proton_spectrum(test_D, R_stack, E, method="BPD") 

    # Iterative solution
    iter_dNdE, __, iter_fit = get_proton_spectrum(test_D, R_stack, E, method="iter", tol=tol,
                                                  T_iter=BPD_fit[0][0], cutoff_iter= cutoff_iter[1])

    plot_spectrum(stack_bragg, iter_dNdE, label="iter.", x_2=stack_bragg, y_2=BPD_dNdE,
                  label_2="BPD", x_line=deposition_energy, y_line=test_dNdE, label_line="Input")

    return


# %% Plotting Functions

def plot_spectrum(x, y, label=None, x_2=None, y_2=None, label_2=None,
                  x_line=None, y_line=None, y_line_fit=None, label_line=None,
                  y_line_2=None, y_line_fit_2=None, label_line_2=None, 
                  y_unit="perMeV"):
    '''
    Plot the proton spectrum.
    
    By default x will be in MeV, while y will be in per J (laugh if not).
    '''

    cmap = plt.get_cmap("tab10")

    if y_unit == "perMeV":
        scale = pc.e*1e6
    elif y_unit == "perJ":
        scale = 1

    if y_line_fit is not None:
        y_line = maxwellian_probability(x_line, *y_line_fit[0])
        y_line_p = maxwellian_probability(x_line, y_line_fit[0][0], 
                                          y_line_fit[0][1]+y_line_fit[1][1])
        y_line_m = maxwellian_probability(x_line, y_line_fit[0][0], 
                                          y_line_fit[0][1]-y_line_fit[1][1])
    if y_line_fit_2 is not None:
        y_line_2 = maxwellian_probability(x_line, *y_line_fit_2[0])
        y_line_p_2 = maxwellian_probability(x_line, y_line_fit_2[0][0], 
                                            y_line_fit_2[0][1]+y_line_fit_2[1][1])
        y_line_m_2 = maxwellian_probability(x_line, y_line_fit_2[0][0], 
                                            y_line_fit_2[0][1]-y_line_fit_2[1][1])

    fig, ax = pm.plot_figure_axis()

    ax.scatter(x, y*scale, label=label, s=50)
    if y_2 is not None and x_2 is not None:
        ax.scatter(x_2, y_2*scale, label=label_2, marker="x", s=50)
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

    ax.set_title("shot" + str(shot))
    if label is not None:
        ax.legend()

    fig.tight_layout()

    return


#%% Main script

project = "Carroll_2023"

if project == "Carroll_2023":
    e_range = [1,120]
    design = None
    shot = "019"
    stack = "18"
    layers = ["B","C","D","E","F","G","H","I","J","K","L"] 
    suffix = None
    edge = [100,20]
    scanner = "Epson_12000XL"
    material_type = None
    OD = False
    clean = True
    channels = [1,1,1]
    dpi = 300

elif project == "Woolsey_2019":
    e_range = [1,80]
    design = "C"
    shot = "056"
    stack = "56"
    layers = ["C","D","E","F","G","H"]
    suffix = None
    edge = [0,0]
    scanner = "Nikon_CoolScan9000"
    material_type = "0"
    OD = True
    clean = False
    channels = [0,1,0]
    dpi = 1000

stack_materials = get_stack_materials(project, shot, layers)

deposition_curves, deposition_energy = get_deposition_curves(energy_range_MeV=e_range,
                                                              project=project, shot=shot,
                                                              design=design,
                                                              dE=0.00625, dx=0.25,
                                                              normalise=False, plot=True) # Curves in J, energy in MeV

if 1:
    stack_dose, stack_error = get_dose_data(project, shot, stack, layers, suffix=suffix,
                                            edge=edge, shape="rectangle", OD=OD,
                                            scanner=scanner, material_type=material_type,
                                            clean=clean, sigma=5, clean_chan=[1,1,1],
                                            channels=channels, plot=False, plot_output=True) # Dose in Grays

    stack_energy = [get_deposited_energy(layer_dose, dpi, stack_materials[i],
                                         dose_error=stack_error[i]) 
                    for i, layer_dose in enumerate(stack_dose)] # Energy in J, tuple including error

    stack_energy_total = np.array([np.sum(layer_energy[0]) for layer_energy in stack_energy]) # Energy in J
    stack_error_total = np.array([np.sum(layer_energy[1]) for layer_energy in stack_energy])

    BPD_dNdE, stack_bragg, BPD_fit = get_proton_spectrum(stack_energy_total, deposition_curves, 
                                                         deposition_energy, stack_layers=layers, method = "BPD")#,
                                                         # stack_energy_error=stack_error_total)


    iter_dNdE, __, iter_fit = get_proton_spectrum(stack_energy_total, deposition_curves, 
                                                  deposition_energy, stack_layers=layers, 
                                                  method="iter", T_iter= BPD_fit[0][0], cutoff_iter = 59) #T_iter=6.754, cutoff_iter=23.887)
                                                  #T_iter=11.779, cutoff_iter=53.7813)


    plot_spectrum(stack_bragg, BPD_dNdE, label="BPD", x_2=stack_bragg, y_2=iter_dNdE,
                  label_2="iter.", x_line=deposition_energy, y_line_fit=BPD_fit,
                  y_line_fit_2=iter_fit)

if 0: # Test solver
    test_get_proton_spectrum(deposition_energy, deposition_curves[:,:], T_MeV=20, cutoff=100,
                             tol=0.05, T_iter=20, cutoff_iter=100)



#plt.title('shot' + str(shot))
#plt.show()


# calibrated = True

# shape = "rectangle"

# yfit = False

# radius = 50

# semicircle = -1

# yoffset = 0

# centre_layer = False # Use centre position from this layer for all layers

# start = 0 # Point to start fitting from

# # Setup figures
# test = False

# colours = ["r","g","b"]

# fig, (ax_energy, ax_width) = pm.plot_figure_axis("small", 2)

# # Start analysis
# nlayers = []

# for layer in layers:

#     nlayers.append(ord(layer)-65) # Turn letter into number i.e. A == 0

# nlayers = np.array(nlayers)

# nshots = len(shots)

# for n in range(nshots):

#     if calibrated == False:
#         RCF_data = get_rotate_crop_data(project, shots[n], stack[n], layers)
#         centre = gaussian_fit(RCF_data, analysis="calib", plot=False)
#     else:
#         RCF_data = get_dose_data(project, shots[n], stack[n], layers, suffix=suffix,
#                                   edge=edge, shape=shape, calibration="valid", scanner=scanner,
#                                   clean=False, sigma=5, nan=True, plot=False)
#         centre, sigma = gaussian_fit(RCF_data, analysis="dose", yfit=yfit, plot=True)

#     if test:
#         for nlayer in range(len(layers)):
#             fig_centre, ax_centre = pm.plot_figure_axis("small", 1)
#             ax_centre.imshow(RCF_data[nlayer][:,:,0])#, vmin=0, vmax=10)
#             ax_centre.scatter(centre[0,0,nlayer], 0, color="r")
#             ax_centre.scatter(centre[0,1,nlayer], 0, color="g")
#             ax_centre.scatter(centre[0,2,nlayer], 0, color="b")
#             ax_centre.scatter(np.mean(centre[0,:,nlayer]), 0, color="w")
#             fig_centre.tight_layout()

#     average = calc_average_circle(RCF_data, centre, radius, fixed=centre_layer,
#                                   semicircle=semicircle, yoffset=yoffset,
#                                   plot=False)[0]

#     proton_energy = get_stack_design(project, shots[n], info="energy")
