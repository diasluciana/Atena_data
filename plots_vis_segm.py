#!/usr/bin/env python
# coding: utf-8

#### used docker image diasluciana/qcnnd:latest-gpu

import os
import numpy as np
import pandas as pd
#from load_dataANPpublic import DataANPpublic 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
import keras
import time
from utils import * #MinMaxScaler
from dataanpsegm import anp_seg
from matplotlib.colors import ListedColormap

def plotIMGandCurves_SegmResult(depth, AMP,xtestM, PRED, ytest,pred,Qtrain=True,range=[600,1200], pathReport=os.getcwd()):
    ######################## Plot Image segment PREDICT
    custom_cmap = ListedColormap(['yellow', 'red', 'blue', 'black'])

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,7), dpi=100)
    # Plotting the original seg_df.
    ax[0].imshow(AMP[range[0]:range[1]], cmap='afmhot',aspect='auto',vmin= np.mean(AMP) - np.std(AMP),
                              vmax= np.mean(AMP) + np.std(AMP))
    ax[0].set_title('AMP image')
    ax[0].axis('off')

    ax[1].imshow(xtestM[range[0]:range[1]], cmap="gray", aspect='auto')
    if Qtrain:
        title = 'QUNET'
    else:
        title = 'UNET'
    ax[1].set_title('True Mask')
    ax[1].axis('off')

    ax[2].plot(ytest[range[0]:range[1]],depth[range[0]:range[1]], "b", label="True")
    ax[2].plot(pred[range[0]:range[1]], depth[range[0]:range[1]], "g", label="Pred")
    ax[2].legend()
    ax[2].set_title(title +' Area True and Predict') # mega giga poros e hight Permeability matrix
    ax[2].axis('off')

    diff = dice_coef(xtestM[range[0]:range[1]], PRED[range[0]:range[1]])
    diff = diff.numpy()
    diff = np.around(diff,decimals=2)
    print(diff)

    ax[3].imshow(PRED[range[0]:range[1]],cmap='gray',aspect='auto')
    ax[3].set_title('PREDICT/Dice coef = ' + str(diff),  fontdict={'fontsize':8}) # mega giga poros e hight Permeability matrix
    ax[3].axis('off')

    plt.subplots_adjust()
    plt.tight_layout()
    #plt.show()
    NFILE = "ResultSEGM_IMGandCURVES_CANDIDAandMODEL.png"
    plt.savefig(os.path.join(pathReport,NFILE))
    print("IMAGE COMPARE Result TESTSET saved in: " + os.path.join(pathReport,NFILE))
    


def plotIMG_SegmResult(depth, AMP,xtestM,predALL,Qtrain=True,range=[600,1200], pathReport=os.getcwd()):
    ######################## Plot Image segment PREDICT
    custom_cmap = ListedColormap(['yellow', 'red', 'blue', 'black'])

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10,7), dpi=100)
    # Plotting the original seg_df.
    ax[0].imshow(AMP[range[0]:range[1]], cmap='afmhot',aspect='auto',vmin= np.mean(AMP) - np.std(AMP),
                              vmax= np.mean(AMP) + np.std(AMP))
    ax[0].set_title('AMP image')
    ax[0].axis('off')

    AMP_SEG_MGP = xtestM[range[0]:range[1]].copy()
    #AMP_SEG_MGP[AMP_SEG_MGP < 2.0] = 0.0
    #AMP_SEG_MGP[AMP_SEG_MGP > 0.0] = 1.0

    ax[1].imshow(predALL[range[0]:range[1]], cmap="gray", aspect='auto')
    if Qtrain:
        ax[1].set_title('QUNET segmentation')
    else:
        ax[1].set_title('UNET segmentation')
    ax[1].axis('off')

    ax[2].imshow(AMP_SEG_MGP, cmap="gray", aspect='auto')
    ax[2].set_title('Candida segmentation') # mega giga poros e hight Permeability matrix
    ax[2].axis('off')
    
    diff = dice_coef(xtestM[range[0]:range[1]], predALL[range[0]:range[1]])
    diff = diff.numpy()
    diff = np.around(diff,decimals=2)
    print(diff)

    ax[3].imshow(AMP_SEG_MGP-predALL[range[0]:range[1]], cmap="seismic", aspect='auto')
    ax[3].set_title('Diff segmentation/Dice coef = ' + str(diff),  fontdict={'fontsize':8}) # mega giga poros e hight Permeability matrix
    ax[3].axis('off')
    
    plt.subplots_adjust()
    plt.tight_layout()   
    #plt.show()
    NFILE = "ResultSEGM_IMG_CANDIDAandMODEL.png"
    plt.savefig(os.path.join(pathReport,NFILE))
    print("IMAGE COMPARE Result TESTSET saved in: " + os.path.join(pathReport,NFILE))

