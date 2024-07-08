## New code for RCF analysis

Preparation
- Specify full path of Diagnostics_v2 root directory in RCF_Plotting.py
- Put all raw data and stack details into Data folder (example given -> can run code as is only changing ROOTDIR).

General
- Use RCF_Plotting.py only as a library for all sorts of plotting algorithms and fitting functions.
- Use RCF_Image_Crop.py to just crop and rotate images and export them as needed.

Proton spectrum
- Process: raw data -> cropping and rotating -> converting to dose -> converting to spectrum
- Use RCF_Dose.py to convert all stack layers into dose pictures.
- Use RCF_Deposition_Curves.py to determine the protons of which energy deposit into which layer.
- Use RCF_Proton_Spectrum.py to find the energy spectrum and temperature of protons.

Beam divergence
- Process: raw data -> cropping and rotating -> converting to brightness -> find central blob -> integrated B field
- Use RCF_Beam_Divergence.py to determine how much the proton beam diverges from the centre of the stack 
and determine the integrated magnetic fields.