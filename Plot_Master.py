# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 14:12:47 2022

@author: add525
"""

'# Libraries'

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


'# Functions'

def setup_fontsize(size):
    '# Setup font sizes'
    
    if type(size) == int:
        
        mpl.rcParams['font.size'] = size
        
        mpl.rcParams["figure.titlesize"]=   'large' # default 'large'
        mpl.rcParams['axes.titlesize']  =   'large' # default 'large' = 'medium' * 1.2
        mpl.rcParams['axes.labelsize']  =   'medium' # default 'medium'
        mpl.rcParams['xtick.labelsize'] =   'small' # default 'medium'
        mpl.rcParams['ytick.labelsize'] =   'small' # default 'medium'
        mpl.rcParams['legend.fontsize'] =   'small' # default 'medium'
        
        return # print("Using {} fontsize.".format(size))
    
    elif size == "small":
        # XSmall = 12
        Small = 16
        Medium = 20
        Large = 24
        
    elif size == "large":
        # XSmall = 24
        Small = 32
        Medium = 40
        Large = 40
        
    mpl.rcParams['figure.titlesize']=   Large    
    mpl.rcParams['axes.titlesize']  =   Medium
    mpl.rcParams['axes.labelsize']  =   Medium
    mpl.rcParams['xtick.labelsize'] =   Small
    mpl.rcParams['ytick.labelsize'] =   Small
    mpl.rcParams['legend.fontsize'] =   Small
    
    mpl.rcParams["figure.dpi"] = 100
    
    return # print("Using {} fontsize.".format(size))


def add_legend(axis, label=None, variable="", species="", loc=None,
               line_include=None):
    '# Find what should be in the legend and add it'
    # If only one data set then no legend required

    # If label is not already provided
    if label is None:
        if isinstance(variable, list) and isinstance(species, list):
            label = [(spec,var) for spec in species for var in variable]
        elif isinstance(variable, list):
            label = [var for var in variable]
        elif isinstance(species, list):
            label = [spec for spec in species]
        else:
            label = [species + ", " + variable]
         
    # Modify label
    label = [s.replace("rhoNumber", "n") for s in label]
    label = [s.replace("Tsim", "T") for s in label]
    label = [s.replace("ion ", "") for s in label]
        
    # Enable legend if more than one label
    if len(label) > 1:
        # If there is only one axis all labels are associated with lines on that axis
        if not isinstance(axis, list):
            axis.legend(label)
        # If there are mulitple axis combine find all lines so labels can be combined
        else:
            print("Y")
            print(label)
            lines = []
            lines.append(axis[0].get_lines())
            lines.append(axis[1].get_lines())
            lines = [item for sublist in lines for item in sublist]
            print(lines)
            
            if line_include is not None:
                lines = [lines[i] for i in line_include]
            
            axis[-1].legend(lines, label, loc=loc)
    
    return


def plot_figure_axis(size="small", nplots=1, shape=1, ratio="square", output_ax=True):
    '''
    Function for making figures and axes
        nplot: number of plots, although this is overridden if shape is specified
            as a list.
        size: sets the text size and dimension of each subplot in the figure 
            with "small" = 6.
        shape: can be an integer specifying the number of rows, or a list for more
            complicated layouts.
        ratio: by default subplots are square, but if a list [a,b] is used the 
            subplot will be scaled by a in x and b in y.
    '''
    
    # Setup font size
    setup_fontsize(size)
    
    # Setup figure size
    subplotlen = 6 # Used to vary with "size" but this is unhelpful + can control with ratio.
        
    if type(shape) == int:
        rows = shape
        columns = int(nplots/rows)
        
    elif type(shape) == list:
        rows = len(shape)
        columns = max([sum(l) for l in shape])
    
    # Ratio controls the "ratio" of the x-axis to the y-axis length for the
    # individual subplots.
    if ratio == "square": # If the individual subplots should be square.
        fig = plt.figure(figsize=(subplotlen*columns,subplotlen*rows)) # First input controls xlen, second ylen.
        
    else: # If the subplots should have some arbitrary shape.
        fig = plt.figure(figsize=(ratio[0]*subplotlen*columns,ratio[1]*subplotlen*rows))
        
    '# Setup figure axis'
    # The variable "shape" controls the layout of the plots.
    
    # From the documentation:
    # Input: Three integers (nrows, ncols, index). The subplot will take the
    # index position on a grid with nrows rows and ncols columns. index starts
    # at 1 in the upper left corner and increases to the right. index can also
    # be a two-tuple specifying the (first, last) indices (1-based, and
    # including last) of the subplot, e.g., fig.add_subplot(3, 1, (1, 2))
    # makes a subplot that spans the upper 2/3 of the figure.
    
    if type(shape) == int:
        for n in range(nplots):        
            fig.add_subplot(shape,int(nplots/shape),n+1)
    
    elif type(shape) == list:
        nrow = 1
        for row in shape:
            ncol = 1
            for col in row:
                col_index = ((nrow-1)*sum(row) + ncol, (nrow-1)*sum(row) + ncol + col-1)
                # print(col_index)
                fig.add_subplot(len(shape),sum(row),col_index)
                ncol+=col
            nrow+=1
    # elif type(shape) == list:
    #     nrow = 1
    #     for row in shape:
    #         ncol = row
    #         for n in range(ncol):
    #             fig.add_subplot(len(shape),ncol,((nrow-1)*ncol)+n+1)
    #         nrow+=1
    
    if output_ax == False:
        return fig
    else:
        if nplots == 1:
            return fig, fig.axes[0]    
        else:
            return fig, fig.axes


def plot(data, data_x=None, size="small", nplots=1, shape=1, ratio="square", output_ax=False):
    '# Generic plotting function'
    
    # Create figure
    fig, ax = plot_figure_axis(size, nplots, shape, ratio)
    
    # Plot data
    if data.ndim == 2: # 2D data
        ax.imshow(data)
        
    elif data.ndim == 1: # 1D data
        if data_x is None:
            data_x = np.arange(len(data))
        ax.plot(data_x, data)
        
    fig.tight_layout()
    
    if output_ax:
        return ax
    else:
        return


if __name__ == "__main__":

    # fig, ax = plot_figure_axis("small", 16, 2, [1,1/1.5])
    fig, ax = plot_figure_axis("small", 3, [[1,1],[2]])
    # fig, ax = plot_figure_axis("small", 3, [[2,1]])
    # fig, ax = plot_figure_axis("small", 4, 2)