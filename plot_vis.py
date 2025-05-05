#!/usr/bin/env python
# coding: utf-8



#Importing tensorflow
import time
import pandas as pd
#For data manipulation
import numpy as np
from sklearn import metrics
import pickle
#to plot stuff
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from random import sample

def save_resultsPKL(results, path = None):
    with open(path,'wb') as f:
        pickle.dump(results,f)

""" function to load results """
def load_resultsPKL(path=None):
    with open(path,'rb') as f:
        results = pickle.load(f)
    return results


def report_Visual(xtrue,ytrue,ypred,depth,pathsave,cut=[0,255], typeplot="MGP"):
    if typeplot=="MGP":
        if len(np.unique(np.uint8(ypred))) != 2:
            ypredA = [1.0 if x >= np.mean(ypred) else 0.0 for x in ypred]
        else:
            ypredA = [1.0 if x >= np.mean(ypred) else 0.0 for x in ypred]
    DEPTH_TH = depth
    DEPTH_TH_INV = np.sort(DEPTH_TH)[::-1]
    img_data = xtrue
    
    sizebar = 4
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,6), dpi=100)
    #for n,i in enumerate(ytrue):
    #    if float(ytrue[n]) == float(0):
    #        ax[0].plot(np.arange(0,sizebar,1), DEPTH_TH_INV[n]*np.ones((sizebar,)), "b")
    #    if float(ytrue[n]) == float(1):
    #        ax[1].plot(np.arange(0,sizebar,1), DEPTH_TH_INV[n]*np.ones((sizebar,)), "r")

    ax[0].imshow(img_data, cmap='afmhot',aspect='auto',vmin=cut[0],
                                                  vmax=cut[1])
    ax[0].set_title('AMP image')
    print(np.histogram(ytrue)) 
    #ax[0].axis('off')
    if len(set(ytrue)) == 2: 
        for n,i in enumerate(ytrue):
            if float(ytrue[n]) == float(0):
                ax[1].plot(np.arange(0,sizebar,1), DEPTH_TH_INV[n]*np.ones((sizebar,)), "b")
            if float(ytrue[n]) == float(1):
                ax[1].plot(np.arange(0,sizebar,1), DEPTH_TH_INV[n]*np.ones((sizebar,)), "r")

        # Plotting the result model.
        for n,i in enumerate(ypredA):
            if float(ypredA[n]) == float(0):
                ax[2].plot(np.arange(0,sizebar,1), DEPTH_TH_INV[n]*np.ones((sizebar,)), "b")
            if float(ypredA[n]) == float(1):
                ax[2].plot(np.arange(0,sizebar,1), DEPTH_TH_INV[n]*np.ones((sizebar,)), "r")

    else:
        ax[1].plot(ytrue, DEPTH_TH_INV, "b")
        ax[1].plot(ypred, DEPTH_TH_INV, "r")
        ax[2].plot(ytrue, DEPTH_TH_INV, "b")
        ax[2].plot(ypred, DEPTH_TH_INV, "r")

    ax[1].set_title('True by Candida')
    ax[2].set_title('Pred by Model')
    #ax[1].axis('off')
    #ax[2].axis('off')

    plt.subplots_adjust()
    plt.tight_layout()
    pathplt = os.path.join(pathsave[:-4] + "Plot_VisualPredTrue.png")
    plt.savefig(pathplt)
    print("Plot Visual Prediction  saved in: " + pathplt)

