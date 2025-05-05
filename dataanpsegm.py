#!/usr/bin/env python
# coding: utf-8

#### used docker image diasluciana/qcnnd:latest-gpu

import os
import numpy as np
import pandas as pd
from load_dataANPpublic import DataANPpublic 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow as tf
import keras
import time
from utils import * #MinMaxScaler, plotMosaic
from matplotlib.colors import ListedColormap


def anp_seg(SW=32,well="antilope25",pathResult=os.getcwd(),pathReport=os.getcwd(),NsamplesTrain=60000,ObjInt="MGP"):
    SW = 32 # size image input model train
    if ObjInt == "MGP":
        ctr = 3.0
    else:
        ctr = 2.0
    AMP, AMP_SEG, MGP, LineAreaPerc, DEPTH, df_welllogs = DataANPpublic(well="antilope25")
    print(AMP.shape, AMP_SEG.shape, MGP.shape, DEPTH.shape)

    ######################## Plot Image segment
    custom_cmap = ListedColormap(['yellow', 'red', 'blue', 'black'])

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(9,6), dpi=100)
    # Plotting the original seg_df.
    ax[0].imshow(AMP[600:1200], cmap='afmhot',aspect='auto',vmin= np.mean(AMP) - np.std(AMP),
                      vmax= np.mean(AMP) + np.std(AMP))
    ax[0].set_title('AMP image')
    ax[0].axis('off')

    AMP_SEG_MGP = AMP_SEG[600:1200].copy()
    AMP_SEG_MGP[AMP_SEG_MGP < ctr] = 0.0
    AMP_SEG_MGP[AMP_SEG_MGP > 0.0] = 1.0

    ax[1].imshow(AMP_SEG[600:1200], cmap=custom_cmap, aspect='auto')
    ax[1].set_title('4 class Candida segmentation')
    ax[1].axis('off')

    ax[2].imshow(AMP_SEG_MGP, cmap="gray", aspect='auto')
    ax[2].set_title('MGP and HP Candida segmentation') # mega giga poros e hight Permeability matrix
    ax[2].axis('off')

    plt.subplots_adjust()
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(pathResult,"ResultSEGM_CANDIDA.png"))

    ########## Define Binary Masks 
    AMP_SEGB = AMP_SEG.copy()
    AMP_SEGB[AMP_SEGB < ctr] = 0.0
    AMP_SEGB[AMP_SEGB > 0.0] = 1.0
    print(np.histogram(AMP_SEGB))
    #print(AMP_SEGB)

    ########## standard and convert array to dataframe sets
    x = AMP #returns a numpy array
    if 0: 
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        x_scaled = min_max_scaler.fit_transform(x)
    else:
        x_scaled = x

    print("min original x: " + str(np.min(x)))
    print("max original x: " + str(np.max(x)))
    minPerc = -(np.percentile(-x, 100.0))
    maxPerc = np.percentile(x,99.9)
    #x = np.clip(x,minPerc,maxPerc)
    print("min original x cut percentile : " + str(np.min(x)))
    print("max original x cut percentile : " + str(np.max(x)))
    #x_scaled = MinMaxScaler(x, scale=[255.0,0.0])
    #x_scaled = x_scaled * 3.0
    df_AMP = pd.DataFrame(x_scaled,columns=list(map(str,np.arange(0,180,1))))
    df_AMP["IDline"] = np.arange(0,df_AMP.shape[0],1)
    df_AMP['DEPTH'] = DEPTH
    #print(list(df_AMP.columns.values))
    #print(df_AMP)
    print("Dataframe AMP shape")
    print(df_AMP.shape)


    filedfData = "DFsets.csv"
    if not os.path.exists(os.path.join(pathResult,filedfData)):
        idtest = np.arange(0,2000,1)
        idtrain  = np.arange(2000,df_AMP.shape[0],1)
        df_AMPtrain, df_AMPtest = \
            df_AMP.loc[idtrain,:], df_AMP.loc[idtest,:]
        df_AMPtrain["trainset"] = np.ones((df_AMPtrain.shape[0],1))
        df_AMPtest["testset"] = np.ones((df_AMPtest.shape[0],1))


        #df_AMPsplit = df_AMP.copy()
        df_AMPsplit = pd.concat([df_AMPtrain,df_AMPtest], axis=0)
        df_AMPsplit = df_AMPsplit.sort_index() # sort by index 
        #df_AMPsplit = df_AMPsplit.fillna(0)
        df_AMPsplit.to_csv(os.path.join(pathResult,filedfData))
    else:
        df_AMPsplit = pd.read_csv(os.path.join(pathResult,filedfData))
        df_AMPtrain = df_AMPsplit[df_AMPsplit["trainset"] > 0.0]
        df_AMPtest = df_AMPsplit[df_AMPsplit["testset"] > 0.0]

    print("Dataframe AMP and others shape")
    print(df_AMPsplit.shape)
    print("Dataframe TRAIN AMP and others shape")
    print(df_AMPtrain.shape)
    print("Dataframe TEST AMP and others shape")
    print(df_AMPtest.shape)

    ##################### Define split data sets IMAGE and MASKS
    coll = list(map(str,np.arange(0,180,1)))
    #print(coll)
    #print(df_AMPtrain)

    DFtrain = df_AMPtrain[coll]
    DFtest = df_AMPtest[coll]
    xtrain = np.array(DFtrain.values,dtype=np.float32)
    xtest = np.array(DFtest.values,dtype=np.float32)

    print("Shapes xtrain,xtest")
    print(xtrain.shape, xtest.shape)


    idxtrain = np.array(df_AMPtrain["IDline"].values, dtype=np.uint32)
    #print(idxtrain)

    idxtest = np.array(df_AMPtest["IDline"].values, dtype=np.uint32)
    #print(idxtest)

    xtrainM = AMP_SEGB[idxtrain, :]
    xtestM = AMP_SEGB[idxtest, :]

    print("Shapes MASKS xtrainM,xtestM")
    print(xtrainM.shape, xtestM.shape)

    ####################### Define inputs to training model using sliding windows from size window 
    X=[]
    Y=[]
    xtrainW = []
    for (x, y, window) in sliding_window(xtrain, stepSize=1, windowSize=(SW, SW)):
        if window.shape == (SW,SW):
            xtrainW.append(window)
            X.append(x)
            Y.append(y)
            #print(x,y)

    xtrainW = np.array(xtrainW, dtype=np.float32)

    XM = []
    YM = []
    xtrainMW = []
    for (x, y, window) in sliding_window(xtrainM, stepSize=1, windowSize=(SW, SW)):
        if window.shape == (SW,SW):
            xtrainMW.append(window)
            XM.append(x)
            YM.append(y)

    xtrainMW = np.array(xtrainMW, dtype=np.float32)

    Xteste = np.sum(np.array(X)-np.array(XM))
    Yteste = np.sum(np.array(Y)-np.array(YM))
    print(Xteste)
    print(Yteste)
    #print(X[:50],XM[:50])
    #print(Y[:50],YM[:50])


    print("Shapes xtrain windows input")
    print(xtrainW.shape)
    print("Shapes MASKS xtrainM windows input")
    print(xtrainMW.shape)

    ####################### centralizando objetos bordas (adicionando valores 1.0 nas bordas do input e 0.0 nas bordas do output
    CenterObj = False
    if CenterObj:
        PixMin = np.min(xtrainW)
        xtrainW[:,:2,:] = PixMin
        xtrainW[:,:,:2] = PixMin
        xtrainW[:,-2:,:] = PixMin
        xtrainW[:,:,-2:] = PixMin

        PixMin = np.min(xtrainMW)
        xtrainMW[:,:2,:] = PixMin
        xtrainMW[:,:,:2] = PixMin
        xtrainMW[:,-2:,:] = PixMin
        xtrainMW[:,:,-2:] = PixMin


    ########################## CLEANING windows input TRAIN none information, when np.sum() == 0.0
    IDX01 = []
    IDXFULLONES = []
    IDXFULLZEROS = []
    for i in range(xtrainMW.shape[0]):
        if not (float(np.sum(xtrainMW[i])) == float(0.0) or float(np.sum(xtrainMW[i])) == float(SW*SW*1.0)):
            IDX01.append(i)
        else:
            if float(np.sum(xtrainMW[i])) == float(0.0):
                IDXFULLZEROS.append(i)
            else:
                IDXFULLONES.append(i)

    #else:
    #    print(float(np.sum(xtrainMW[i])))
    print(np.histogram(xtrainMW))  
    print("Number and Percentage of IDX01 images input network")
    print(len(IDX01), len(IDX01)/xtrainMW.shape[0])
    print("Number and Percentage of IDXFULLZEROS images input network")
    print(len(IDXFULLZEROS), len(IDXFULLZEROS)/xtrainMW.shape[0])
    print("Number and Percentage of IDXFULLONES images input network")
    print(len(IDXFULLONES), len(IDXFULLONES)/xtrainMW.shape[0])


    IDXUSE = IDX01 #+ IDXFULLONES #+ IDXFULLZEROS ### USEI TDS
    xtrainW = xtrainW[IDXUSE,:,:]
    xtrainMW = xtrainMW[IDXUSE,:,:]

    print("Shapes xtrain windows input not sum == 0")
    print(xtrainW.shape)
    print("Shapes MASKS xtrainM windows input not sum == 0")
    print(xtrainMW.shape)

    ########################## SUFFLE and CUT number samples windows input TRAINSET 
    N = np.random.permutation(np.arange(0,xtrainW.shape[0],1))
    if NsamplesTrain is not None:
        N = N[:NsamplesTrain]
    xtrainW = xtrainW[N]
    xtrainMW = xtrainMW[N]

    print("Shapes xtrain windows input after SUFFLE and select number of samples")
    print(xtrainW.shape)
    print("Min Max xtrain windows input after SUFFLE and select number of samples")
    print(np.min(xtrainW),np.max(xtrainW))
    print("Shapes MASKS xtrainM windows input after SUFFLE and select number of samples")
    print(xtrainMW.shape)
    print("Min Max MASKS xtrainM windows input after SUFFLE and select number of samples")
    print(np.min(xtrainMW),np.max(xtrainMW))

    ###################### Adjust axis

    if len(xtrainW.shape) < 4:
        xtrainW = xtrainW[..., np.newaxis]

    if len(xtrainMW.shape) < 4:
        xtrainMW = xtrainMW[..., np.newaxis]

    print("Shapes xtrain windows input after SUFFLE and select number of samples")
    print(xtrainW.shape)
    print("Min Max xtrain windows input after SUFFLE and select number of samples")
    print(np.min(xtrainW),np.max(xtrainW))
    print("Shapes MASKS xtrainM windows input after SUFFLE and select number of samples")
    print(xtrainMW.shape)
    print("Min Max MASKS xtrainM windows input after SUFFLE and select number of samples")
    print(np.min(xtrainMW),np.max(xtrainMW))
    ##################### Show the inputs windows train input network
    img_arrs = []
    N = np.random.permutation(np.arange(0,xtrainW.shape[0],1))
    for i in N[:4]:
        img_arrs.append(xtrainW[i])
        img_arrs.append(xtrainMW[i])
        img_arrs.append(xtrainMW[i])

    print("PLOTTING MOSAIC IMAGES INPUT NETWORK")
    plotMosaic(img_arrs,path=os.path.join(pathReport,"PlotMosaicInputNetwork.png"))


    ###################################### CONVERT IMAGE WELL TO IMAGE WELL WINDOWS SW
    xtestW = []
    Xtt = []
    Ytt = []
    for (x, y, window) in sliding_window(xtest, stepSize=6, windowSize=(SW, SW)):
        if window.shape == (SW,SW):
            xtestW.append(window)
            Xtt.append(x)
            Ytt.append(y)

    xtestW = np.array(xtestW, dtype=np.float32) ##### convert IMAGE xtest size None,180 to xtestW None,SW,SW
    print("XTESTW SHAPE (IMAGE WINDOWS)...")
    print(xtestW.shape)

    ###################################### CONVERT IMAGE MASKS WELL TO IMAGE WELL WINDOWS SW
    xtestMW = []
    XM = []
    YM = []
    for (x, y, window) in sliding_window(xtestM, stepSize=6, windowSize=(SW, SW)):
        if window.shape == (SW,SW):
            xtestMW.append(window)
            XM.append(x)
            YM.append(y)

    xtestMW = np.array(xtestMW, dtype=np.float32) ##### convert MASKS xtest size None,180 to xtestW None,SW,SW
    print("XTESTMW SHAPE (MASKS WINDOWS)...")
    print(xtestMW.shape)

    if 0:
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        xtrainW = min_max_scaler.fit_transform(xtrainW)
        xtestW = min_max_scaler.fit_transform(xtestW)

    ###################### Adjust axis

    if len(xtestW.shape) < 4:
        xtestW = xtestW[..., np.newaxis]

    if len(xtrainMW.shape) < 4:
        xtestMW = xtestMW[..., np.newaxis]

    ##################### Show the inputs windows train input network
    img_arrs = []
    NRDM = np.random.permutation(np.arange(0,xtestW.shape[0],1))
    for i in NRDM[:4]:
        img_arrs.append(xtestW[i])
        img_arrs.append(xtestMW[i])
        img_arrs.append(xtestMW[i])

    print("PLOTTING MOSAIC IMAGES INPUT NETWORK")
    plotMosaic(img_arrs,path=os.path.join(pathReport,"PlotMosaicTESTSETwindows.png"))

    return DEPTH, AMP, xtrain, xtrainM, xtrainW, xtrainMW, xtest, xtestM, xtestW, xtestMW, df_AMPsplit, df_AMPtrain, df_AMPtest, Xtt, Ytt

