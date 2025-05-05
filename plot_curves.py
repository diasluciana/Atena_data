#!/usr/bin/env python
# coding: utf-8

# # WAID - plot image and basic logs
# 
# **Created by:** Rewbenio A. Frota
# 
# This notebook briefly shows how to load and plot the WAID data.

# ## Import dependencies



import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ## Important functions
# 
# * concat_IMG_data() loads image data from multiple files;
# * composite_plot() plots a composite log with image and basic logs. The present ```composite_plot()``` procedure to plot a composite log is addapted from [Petrophysics-Python-Series / 14 - Displaying Lithology Data.ipynb](https://github.com/andymcdgeo/Petrophysics-Python-Series/blob/master/14%20-%20Displaying%20Lithology%20Data.ipynb). More details [here](https://towardsdatascience.com/displaying-lithology-data-using-python-and-matplotlib-58b4d251ee7a).

# In[2]:


# To load image data from multiple files
def concat_IMG_data(well_id, data_path):
    # Due to file size limitations, the original AMP '.csv' file
    # has been split into several sub-files.
    # The concat_IMG_data() function aims to concatenate
    # them back into a single data object.
    #
    # concat_IMG_data() returns 'image_df', a Pandas dataframe
    # indexed by DEPTH information and whose columns are
    # the azimuthal coordinates of the AMP image log.
    
    # Name of the initial '00' file
    initial_file = well_id + "_AMP00.csv"

    # Read the the initial file to capture header information
    initial_file_path = os.path.join(data_path, initial_file)
    image_df = pd.read_csv(initial_file_path,sep = ';',
                           index_col=0,
                           na_values = -9999,na_filter = True,
                           decimal = ',',
                           skip_blank_lines = True).dropna()

    # Read and add data from the remaining files sequentially
    for file in os.listdir(data_path):
        if file.startswith(well_id) and file != initial_file:
            file_path = os.path.join(data_path, file)
            df_temp = pd.read_csv(file_path,sep = ';',
                                  header=None,index_col = 0,
                                  na_values = -9999, na_filter = True,
                                  decimal = ',', skip_blank_lines = True,
                                  dtype=np.float32
                                 ).dropna()
            
            # Adjust tem df's header to match image header
            df_temp.columns=image_df.columns
            
            # Concat dfs
            image_df = pd.concat([image_df, df_temp])
    return image_df

# To plot composite logs
def log_plot(depth, dfcurves, image_df, topdepth, bottomdepth):
    
    fig, ax = plt.subplots(figsize=(8,6), sharey = True)    #, layout="tight")

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    #Set up the plot axes
    ax1 = plt.subplot2grid((1,12), (0,0), rowspan = 1, colspan = 3)
    ax2 = plt.subplot2grid((1,12), (0,3), rowspan = 1, colspan = 3, sharey = ax1)
    ax3 = plt.subplot2grid((1,12), (0,6), rowspan = 1, colspan = 3, sharey = ax1)
    ax4 = ax3.twiny()    #Twins the y-axis for the density track with the neutron track
    ax5 = plt.subplot2grid((1,12), (0,9), rowspan = 3, colspan = 3, sharey = ax1)

    # As our curve scales will be detached from the top of the track,
    # this code adds the top border back in without dealing with splines
    ax10 = ax1.twiny()
    ax10.xaxis.set_visible(False)
    ax11 = ax2.twiny()
    ax11.xaxis.set_visible(False)
    ax12 = ax3.twiny()
    ax12.xaxis.set_visible(False)
    ax15 = ax5.twiny()
    ax15.xaxis.set_visible(False)
    
    parameters = []
    for i in dfcurves:
        parameters.append(i)
    # Gamma Ray track
    #print(parameters)
    
    ## Setting up the track and curve
    ax1.plot(dfcurves[parameters[0]], depth, color = "green", linewidth = 0.5)
    ax1.set_xlabel(parameters[0])
    ax1.xaxis.label.set_color("green")
    ax1.set_xlim(0, 60)
    ax1.set_ylabel("Depth (m)", fontweight='bold')
    ax1.tick_params(axis='x', colors="green")
    ax1.spines["top"].set_edgecolor("green")
    ax1.title.set_color('green')
    ax1.set_xticks([0, 60])
    ax1.text(0.02, 1.04, '0', color='green', 
             horizontalalignment='left', transform=ax1.transAxes)
    ax1.text(0.98, 1.04,'60', color='green', 
             horizontalalignment='right', transform=ax1.transAxes)
    ax1.set_xticklabels([])
    
    # Resistivity track
    ax2.plot(dfcurves[parameters[1]], depth, color = "tab:orange", linewidth = 0.5)
    ax2.set_xlabel(parameters[1])
    ax2.set_xlim(1, 10010)
    ax2.xaxis.label.set_color("tab:orange")
    ax2.tick_params(axis ='x', colors="tab:orange")
    ax2.spines["top"].set_edgecolor("tab:orange")
    ax2.set_xticks([1, 10, 100])
    ax2.semilogx()
    ax2.text(0.02, 1.04, '1', color = 'tab:orange', 
             horizontalalignment = 'left', transform = ax2.transAxes)
    ax2.text(0.98, 1.04, '10000', color = 'tab:orange', 
             horizontalalignment = 'right', transform = ax2.transAxes)
    ax2.set_xticklabels([])
    ax2.tick_params(left = False, labelleft = False)  # remove the ticks and axis label

    
    # Density track
    ax3.plot(dfcurves[parameters[2]], depth, color = "red", linewidth = 0.5)
    ax3.set_xlabel(parameters[2])
    ax3.set_xlim(2, 3)
    ax3.xaxis.label.set_color("red")
    ax3.tick_params(axis='x', colors="red")
    ax3.spines["top"].set_edgecolor("red")
    ax3.set_xticks([2, 2.5, 3])
    ax3.text(0.02, 1.04, '2', color='red', 
             horizontalalignment='left', transform=ax3.transAxes)
    ax3.text(0.98, 1.04, '3', color='red', 
             horizontalalignment='right', transform=ax3.transAxes)
    ax3.set_xticklabels([])
    ax3.tick_params(left = False, labelleft = False)  # remove the ticks and axis label


    # Neutron track placed ontop of density track
    ax4.plot(dfcurves[parameters[3]], depth, color = "blue", linewidth = 0.5)
    ax4.set_xlabel(parameters[3])
    ax4.xaxis.label.set_color("blue")
    ax4.set_xlim(0.45, -0.15)
    ax4.tick_params(axis='x', colors="blue")
    ax4.spines["top"].set_position(("axes", 1.09))
    ax4.spines["top"].set_visible(True)
    ax4.spines["top"].set_edgecolor("blue")
    ax4.set_xticks([0.45,  0.15, -0.15])
    ax4.text(0.02, 1.12, '0.45', color='blue', 
             horizontalalignment = 'left', transform = ax4.transAxes)
    ax4.text(0.98, 1.12, '-0.15', color = 'blue', 
              horizontalalignment = 'right', transform = ax4.transAxes)
    ax4.set_xticklabels([])
    ax4.tick_params(left = False, labelleft = False)  # remove the ticks and axis label
  
    # Adding in neutron density shading
    x1 = dfcurves[parameters[-2]]
    x2 = dfcurves[parameters[-1]]

    x = np.array(ax3.get_xlim())
    z = np.array(ax4.get_xlim())

    nz = ((x2-np.max(z))/(np.min(z)-np.max(z)))*(np.max(x)-np.min(x))+np.min(x)

    # To fill space between the porotisy curves
    ax3.fill_betweenx(depth, x1, nz, where=x1>=nz, interpolate=True, color='green')
    ax3.fill_betweenx(depth, x1, nz, where=x1<=nz, interpolate=True, color='yellow')
    
    
    # Common functions for setting up the plot can be extracted into
    # a for loop. This saves repeating code.
    for ax in [ax1, ax2, ax3, ax5]:
        ax.set_ylim(bottomdepth, topdepth)
        ax.grid(which = 'major', color = 'lightgrey', linestyle = '-')
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.spines["top"].set_position(("axes", 1.02))

    for ax in [ax2, ax3, ax4, ax5]:
        plt.setp(ax.get_yticklabels(), visible = True)
        
    ax5.imshow(image_df,
               extent = [0, len(image_df.columns), bottomdepth, topdepth],
               cmap ="afmhot",
               vmin = np.mean(image_df.values) - np.std(image_df.values),
               vmax = np.mean(image_df.values) + np.std(image_df.values),
               aspect = "auto"
              )
    
    ax5.grid(False)
    ax5.set_xlabel("AMP")
    ax5.set_xlim(0, 180)     #ax5.set_xlim(-1, 181.142)
    ax5.text(0.02, 1.04, '0°', color = 'black', 
             horizontalalignment='left', transform=ax5.transAxes)
    ax5.text(0.99, 1.04, '  360°', color = 'black', 
              horizontalalignment='right', transform=ax5.transAxes)
    ax5.set_xticklabels([])
    ax5.set_xticks([0,  90, 180])
    ax5.tick_params(left = False, labelleft = False)  # remove the ticks and axis label
    
    plt.ticklabel_format(useOffset=True)
    
    # Uncomment to save images
    #plt.savefig('Perfil_composto.eps', dpi=400, format='eps')
    plt.savefig('Perfil_composto2.png', dpi=400, format='png')
    #plt.savefig('Perfil_composto.svg', dpi=400, format='svg')
    
    fig.subplots_adjust(wspace = 0)
    #plt.show()


