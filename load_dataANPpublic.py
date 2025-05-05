#!/usr/bin/env python
# coding: utf-8


import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



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

# To plot an image log
def Image_plot(image_df):
    # Plots AMP image using appropriate colormap
    
    fig, ax = plt.subplots(figsize=(4, 9), dpi=100)  # Create a figure containing a single axes.
    ax.ticklabel_format(useOffset = False)
    ax.invert_yaxis()
    ax.set_xlabel('AMP Image')   
              
    ax.imshow(image_df,
              extent=[0, len(image_df.columns), min(image_df.index), max(image_df.index)],
              cmap='afmhot',
              vmin= np.mean(image_df.values) - np.std(image_df.values),
              vmax= np.mean(image_df.values) + np.std(image_df.values),
              aspect='auto'
              )
            
    plt.ticklabel_format(useOffset=False)
    #plt.show()


# ## Import image data from CSV files
def InterpWell(DEPTH_AMP, DFWELL):
    DFAMP = pd.DataFrame({"DEPTH":DEPTH_AMP, "IDLineAmp": np.arange(0,len(DEPTH_AMP),1)})
    #print(DFAMP)
    DFWELL.insert(0, column = 'IDLineLogs', value = np.arange(0,DFWELL.shape[0],1))
    DFWELL.set_index('IDLineLogs', inplace=True)
    #print(DFWELL)
    #print(list(DFWELL.columns.values))
    
    DF = pd.merge(DFAMP, DFWELL, on='DEPTH',  how='left')
    DF = DF.sort_values('DEPTH')
    DF['IDMerge'] = np.arange(0,DF.shape[0],1)
    
    dfclean = DF.dropna()
    DFWELLcol = DFWELL.columns.values.tolist()
    LineOK1 = dfclean['IDMerge'].values[0]
    LineOK2 = dfclean['IDMerge'].values[-1]
    DF = DF.iloc[int(LineOK1):int(LineOK2),:]
    DF = DF.drop(columns='IDMerge')
    #print(DFWELLcol[:-1])
    
    for i in DFWELLcol[:-1]:
        DF[i] = DF[i].interpolate(method='linear')
        DF[i] = DF[i].interpolate(method='cubic')
    #print(DF)
    
    return DF

def DataANPpublic__(well_identifier="antilope25"):
    # Location of the AMP image data files
    img_data_path = r"waid/dataset/img"
    bsc_data_path = r"waid/dataset/bsc"
    # Well identifier
    #well_identifier = "antilope25"    #antilope25  tatu22 botorosa47 coala88  antilope37

    # load basic log data
    bsc_filename =  well_identifier + '_BSC.csv'
    bsc_name = os.path.join(bsc_data_path, bsc_filename)
    well = pd.read_csv(bsc_name,
                        index_col = 0,
                        sep = ';', decimal = ',',
                        na_values = -9999, na_filter = True,
                        skip_blank_lines = True).dropna()

    well['DEPTH'] = well.index
    dfwell = well.iloc[well["DEPTH"].values > 2065.52]
    return dfwell

def Plot2(slice_img_df,img_data,slice_seg_df,threshold,custom_cmap,bins):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,6), dpi=100)

    # Plotting the original seg_df.
    ax[0].imshow(slice_img_df, cmap='afmhot',aspect='auto',vmin= np.mean(img_data.values) - np.std(img_data.values),
                                      vmax= np.mean(img_data.values) + np.std(img_data.values))
    ax[0].set_title('AMP image')
    ax[0].axis('off')

    # Plotting the histogram and the two thresholds obtained from
    # multi-Otsu.
    ax[1].hist(slice_img_df.values.flatten('F'), bins = bins)
    ax[1].set_title('AMP histogram')
    for thresh in threshold:
        ax[1].axvline(thresh, color='r')
        ax[1].set_ylabel("Frequency")

    # Plotting the Multi Otsu result.
    ax[2].imshow(slice_seg_df, cmap=custom_cmap, aspect='auto')
    ax[2].set_title('User defined segmentation')
    ax[2].axis('off')

    plt.subplots_adjust()
    plt.tight_layout()
    plt.savefig("Result2.png")

def Plot_Segm(IMGTH, img_data,IMG_SEGTH,MGPclsTH, DEPTH_TH_INV, custom_cmap):
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,6), dpi=100)

    # Plotting the original seg_df.

    ax[0].imshow(IMGTH, cmap='afmhot',aspect='auto',vmin= np.mean(img_data.values) - np.std(img_data.values),
                                                                              vmax= np.mean(img_data.values) + np.std(img_data.values))
    ax[0].set_title('AMP image')
    ax[0].axis('off')

    sizebar = 4
    for n,i in enumerate(MGPclsTH):
        if float(MGPclsTH[n]) == float(0):
            ax[1].plot(np.arange(0,sizebar,1), DEPTH_TH_INV[n]*np.ones((sizebar,)), "b")
        if float(MGPclsTH[n]) == float(1):
            ax[1].plot(np.arange(0,sizebar,1), DEPTH_TH_INV[n]*np.ones((sizebar,)), "r")
    
    ax[2].imshow(IMG_SEGTH, cmap=custom_cmap, aspect='auto')
    ax[2].set_title('User defined segmentation')
    ax[2].axis('off')
    plt.subplots_adjust()
    plt.tight_layout()
    plt.savefig("Result_seg_candida.png")

def DataANPpublic(well="antilope25"):
    # Location of the AMP image data files
    img_data_path = r"waid/dataset/img"
    bsc_data_path = r"waid/dataset/bsc"
    
    if not (os.path.exists(img_data_path) and os.path.exists(img_data_path)):
        Print("You need clone waid github repository, see in read me on current repository")

    # Well identifier
    well_identifier = "antilope25"    #antilope25  tatu22 botorosa47 coala88  antilope37


    # Load full image data
    img_data = concat_IMG_data(well_identifier,img_data_path)
    print(img_data.shape)
    
    # load basic log data
    bsc_filename =  well_identifier + '_BSC.csv'
    bsc_name = os.path.join(bsc_data_path, bsc_filename)
    well = pd.read_csv(bsc_name,
                    index_col = 0,
                    sep = ';', decimal = ',',
                    na_values = -9999, na_filter = True,
                    skip_blank_lines = True).dropna()

    well['DEPTH'] = well.index
    print(img_data.index.values)
    
    well = InterpWell(img_data.index.values, well)

    img_data = img_data.iloc[img_data.index > 2065.52]  # Deleting spurious values at the top of the image file
    #img_data = img_data.iloc[img_data.index <= 2277.10]     # Interest limit
    well = well.iloc[well["DEPTH"].values > 2065.52]
    print(well.shape)
    print(img_data.shape)
    
    # ## Plot an image slice

    slice_img_df = img_data.iloc[600:1200]   #User defined data slice


    # Flatten amplitude values
    data = img_data.values.flatten('F')

    # User definded thresholds
    threshold = [26, 34, 37]

    # User definded range
    range_dB = np.array((-20, 40))

    # User definded number of bins in histogram
    n_bins = 50

    # range for every bin
    delta_bin = np.abs((range_dB[1] - range_dB[0]))/n_bins

    # Compute the number of bins per class based on 'delta_bin'
    range_bins = np.array((int((threshold[0] - range_dB[0])//delta_bin),
                       int((threshold[1] - range_dB[0])//delta_bin),
                       int((threshold[2] - range_dB[0])//delta_bin)))


    N, bins, patches = plt.hist(data, bins=n_bins, ec="k", range=(range_dB[0], range_dB[1]))

    # ## Segment after histogram categories
    amp_array = img_data.values
    amp_segment = np.zeros(amp_array.shape)

    for i, img_row in enumerate(amp_array):
        for j, amp_value in enumerate(img_row):   
            if amp_value <= threshold[0]:
                amp_segment[i][j] = 3
            elif amp_value <= threshold[1]:
                amp_segment[i][j] = 2
            elif amp_value <= threshold[2]:
                amp_segment[i][j] = 1
            else:
                amp_segment[i][j] = 0



    seg_df = pd.DataFrame(amp_segment)

    seg_df.insert(0, column = 'DEPTH', value = img_data.index)
    seg_df.set_index('DEPTH', inplace=True)



    # Creat custom colormaps to plot segmentations
    from matplotlib.colors import ListedColormap

    custom_cmap = ListedColormap(['yellow', 'red', 'blue', 'black'])
    custom_cmap_r = ListedColormap(['black', 'blue', 'red', 'yellow'])


    slice_seg_df = seg_df.iloc[600:1200]
    slice_seg_dfLINE = np.array(slice_seg_df.values)


    ###############################################################################################################
    Plot2(slice_img_df,img_data,slice_seg_df,threshold,custom_cmap,bins)


    DF_AMP = img_data
    DF_AMP_SEG = seg_df
    print(DF_AMP.shape)
    print(DF_AMP_SEG.shape)
    

    MGPcls = np.zeros((DF_AMP_SEG.shape[0],), dtype=np.uint8)
    HPMcls = np.zeros((DF_AMP_SEG.shape[0],), dtype=np.uint8)
    MPMcls = np.zeros((DF_AMP_SEG.shape[0],), dtype=np.uint8)
    LPMcls = np.zeros((DF_AMP_SEG.shape[0],), dtype=np.uint8)

    ARR_DF_AMP_SEG = DF_AMP_SEG.values
    ARR_DF_AMP = DF_AMP.values
    print(ARR_DF_AMP_SEG)
   
    #cont = 0
    for i in range(DF_AMP_SEG.shape[0]):
        Line = ARR_DF_AMP_SEG[i,:]
        #TTTT = DF_AMP.values[i,:]
        #print(Line)
        #print(TTTT)
        
        if 3.0 in Line: # and cont==i:
            MGPcls[i] = 1

    DF_CLS = pd.DataFrame({'DEPTH':DF_AMP_SEG.index, 'Mega-Giga Pore':MGPcls, 'High permeability matrix':HPMcls, 'Medium permeability matrix':MPMcls, 'Low permeability matrix':LPMcls})
    print(set(MGPcls))

    ################################################################################################################
    # Define percentual de area por profundidade
    AbinSeg = ARR_DF_AMP_SEG.copy()
    AbinSeg[AbinSeg < 3.0] = 0.0
    AbinSeg[AbinSeg == 3.0] = 1.0
    LineAreaPerc = np.zeros(MGPcls.shape)
    for i in range(AbinSeg.shape[0]):
        Line = AbinSeg[i,:]
        if MGPcls[i] == 1:
            LineAreaPerc[i] =  np.sum(Line) / len(Line)

    print(np.histogram(MGPcls))
    print(np.histogram(LineAreaPerc))
        
    ################################################################################################################
    DEPTHTT = DF_AMP_SEG.index
    IMGTT = np.array(ARR_DF_AMP)
    IMG_SEGTT = np.array(ARR_DF_AMP_SEG)  #************************8 
  
    DEPTH_TH = DEPTHTT[600:1200]
    DEPTH_TH_INV = np.sort(DEPTH_TH)[::-1]
    
    MGPclsTH = MGPcls[600:1200]   #User defined data slice
    IMGTH = IMGTT[600:1200]
    IMG_SEGTH = IMG_SEGTT[600:1200]
    print(np.min(IMG_SEGTH), np.max(IMG_SEGTH))
    print(np.min(MGPclsTH), np.max(MGPclsTH))
    
    Plot_Segm(IMGTH, img_data,IMG_SEGTH,MGPclsTH, DEPTH_TH_INV, custom_cmap)

    return IMGTT, IMG_SEGTT, MGPcls, LineAreaPerc, DEPTHTT, well

#AMP, AMP_SEG, MGP, DEPTH, welllogs = DataANPpublic()
#print(AMP.shape, AMP_SEG.shape, MGP.shape, DEPTH)
#print(AMP)
#print(AMP_SEG)
#print(MGP)
#print(DEPTH)

