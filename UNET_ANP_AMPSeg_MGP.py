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


################# Hiper parameters model
Train = True
ANPdata = True
MNISTdata = False
epochs=300
batch_size=5
Qtrain = False
SW = 32 # size image input model train
NsamplesTrain = None #60000 # number of samples to training -> set int or None
makeMosaic = False
ObjInt = "MGP" # "MGPandHP"
########## create fold to result EXP
dirResult = os.getcwd()
if Qtrain:
    foldResult = "QUNET_AMP_ANP_"+ObjInt+"_"+ str(epochs)+"OrigFiltersStandInW"
    #foldResult = "QUNET_AMP_ANP_MPGandHP_" + str(epochs) + "Epochs"
else:
    foldResult = "UNET_AMP_ANP_"+ObjInt+"_" + str(epochs) + "OrigFiltersStandInW"
pathResult = os.path.join(dirResult,foldResult)
if not os.path.exists(pathResult):
    os.makedirs(pathResult)
pathReport = os.path.join(pathResult,"Report")
if not os.path.exists(pathReport):
    os.makedirs(pathReport)


################## save codes in path results
filecorr = os.path.basename(__file__)
print(filecorr)
import shutil

fileadjS = ["load_dataANPpublic.py", "model.py", "utils.py", "dataanpsegm.py", "unet_model.py", "qunet_model.py"] 
if Train:
    pathcode = pathResult
else:
    pathcode = pathReport

if not (os.path.exists(os.path.join(pathcode,filecorr)) and Train):
    shutil.copyfile(os.path.join(os.getcwd(),filecorr), os.path.join(pathcode,filecorr))

for i in fileadjS:
    if not (os.path.exists(os.path.join(pathcode,i)) and Train):
        shutil.copyfile(os.path.join(os.getcwd(),i), os.path.join(pathcode,i))

##################### Load Data
if ANPdata:
    DEPTH, AMP, xtrain, xtrainM, xtrainW, xtrainMW, xtest, xtestM, xtestW, xtestMW, df_AMPsplit, df_AMPtrain, df_AMPtest, Xtt, Ytt = anp_seg(SW=SW,well="antilope25", pathResult=pathResult,pathReport=pathReport,NsamplesTrain=NsamplesTrain,ObjInt=ObjInt)
print(list(df_AMPtest.columns))
print(df_AMPsplit.columns)


if MNISTdata:
    from load_data_mnist_segm import *
    xtrainW, xtrainMW, xtestW, xtestMW = load_mnist_seg(D=SW)

##################### AK model
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import * #ModelQ, Model0
from unet_model import *
from qunet_model import *

nGPU = [0] 

if len(nGPU) > 1:
    strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
else:
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

print(strategy)

fileCSV = os.path.join(pathResult,'history.csv') 
CSVL = tf.keras.callbacks.CSVLogger(
                    fileCSV, separator=',', append=False)
earlystopper = EarlyStopping(patience=5, verbose=1)
fileBM = os.path.join(pathResult,'bestmodel.keras')
checkpointer = ModelCheckpoint(fileBM, verbose=1, save_best_only=True) #, monitor='val_loss', mode='min')
callback = [CSVL,checkpointer]

if Qtrain: 
    with strategy.scope():
        #model = ModelQ(SW) # 9 hs epoca 1 na 228 gpu 0 com 60k samples train e 0.1 de valset batch 5
        model = build_qunet_model() # 425 h [64,128,256,512]
else:
    with strategy.scope():
        #model = Model0(SW)
        model = build_unet_model()

report_model(model, os.path.join(pathResult,"modelsummary.txt"))
filemodelfinal = os.path.join(pathResult, "final_model")
if Train:
    if os.path.exists(fileCSV):
        hist = pd.read_csv(fileCSV)
        epFin = int(np.array(hist["epoch"])[-1])
        epInit = epFin + 1
        EI = epFin+1
        print("Loading weights model and Init epoch " + str(epInit))
        with strategy.scope():
             model.load_weights(fileBM)
    else:
        EI = 0

    valsplit = 0.3
    model.fit(xtrainW, xtrainMW, validation_split=valsplit, batch_size=batch_size, shuffle=True,
                validation_batch_size=batch_size, epochs=epochs,callbacks=callback,
                initial_epoch=EI,
                #steps_per_epoch=int(np.ceil(xtrainW.shape[0]/batch_size)),validation_steps=int(np.ceil((xtrainW.shape[0]*valsplit)/batch_size)))
                steps_per_epoch=500,validation_steps=500)
    
    model.save(filemodelfinal)

##################### Load best model and predict from windows data
#from utils import *
#from model import *
print("@@@@@@@@@ PREDICTING IN TESTSET @@@@@@@@@@@@@@@")
with strategy.scope():
    model.load_weights(fileBM)



###########################################
IntSave = np.arange(0,xtestW.shape[0],50)
StepsP = None
batch_size_pred = 5
filepred = os.path.join(pathResult, str(epochs) + '_Epochs_Predicts_Batch'+str(batch_size_pred)+'_Steps'+str(StepsP)+'.npy')
fileIDpred = os.path.join(pathResult, 'IDpred'+'.npy')
if not os.path.exists(filepred):
    pred = np.zeros(xtestMW.shape)
    idpred = np.zeros((xtestMW.shape[0],),dtype=np.uint8)
    print("Pred no exist, creating predict windows test set...")
else:
    pred = np.load(filepred)
    idpred = np.load(fileIDpred)
    print("Pred exist, loading...")
    print(np.min(pred), np.max(pred))


NFallPred = np.where(idpred < 1.0)[0]


if len(NFallPred) > 1: #not Train:
    print("Pred exist, predicting windows test set from exist idpred " + str(NFallPred[0])+"/"+str(xtestW.shape[0]))
    st = time.time()
    with strategy.scope():
        for i in NFallPred: # np.arange(0,xtestW.shape[0],1):
            PP = model.predict(xtestW[i:i+batch_size_pred,:,:,:],verbose=0,batch_size=batch_size_pred) # ~4h em 1 gpu batch_size=5 steps=500
            print(np.min(PP),np.max(PP))
            #if np.min(PP) < 0.5:
            #    print(np.min(PP),np.max(PP))
            
            ed = time.time() - st
            print("Time Predict "+str(ed)+ " in Window: "+str(i)+" / "+str(len(NFallPred)))
            pred[i:i+PP.shape[0],:,:] = PP[:,:,:,0]
            idpred[i:i+PP.shape[0]] = np.ones((PP.shape[0],))
            if i in IntSave:
                print("Saving all predict during Window: "+str(i)+" / "+str(len(NFallPred)))
                np.save(filepred, pred)
                np.save(fileIDpred, idpred)
        print("Saving FINAL windows test set preds...")
        #print(pred.shape)
        
        np.save(filepred, pred)
        np.save(fileIDpred, idpred)
    ed = time.time() - st
    np.savetxt(os.path.join(pathResult, "ReportTimeStartFinalNumberSamplesPredTestSet.txt"),np.array([st,ed,xtestW.shape[0]]))


pathReportIMG = os.path.join(pathResult,"ReportIMG")
if not os.path.exists(pathReportIMG):
    os.makedirs(pathReportIMG)

filesplotsResultsFinal = os.path.join(pathReportIMG, "PLOTmosaic_"+str(xtestMW.shape[0]-1)+".png")

if makeMosaic:
    if not os.path.exists(filesplotsResultsFinal):
        print("CREATING PLOTS MOSAICS RESULTS WINDOWS ................")
        for i in range(xtestMW.shape[0]):
            TPLT = []
            xMk = xtestW[i,:,:]
            TPLT.append(xMk)
            PRED = pred[i,:,:]
            TPLT.append(PRED)
            DIFF = xtestMW[i,:,:] #- pred[i,:,:]
            TPLT.append(DIFF)
            plotMosaic(TPLT,path=os.path.join(pathReportIMG, "PLOTmosaic_"+str(i)+".png"))

#pred = pred[:,:,:,0]
print("TEST SET PREDICT SHAPE was loaded")
print(pred.shape)
print(np.min(pred), np.max(pred))
print("XTESTW SET SHAPE")
print(xtestW.shape)
print("XTESTMW SET (MASKS) SHAPE")
print(xtestMW.shape)


################################### Analysis pred from all well
predALL = np.zeros(xtest.shape)
#print(np.max(Xtt))
#print(np.max(Ytt))

filepredALLTest = os.path.join(pathResult, str(epochs) + '_Epochs_TESTSETNone180_Predicts_Batch'+str(batch_size_pred)+'_Steps'+str(StepsP)+'.npy')
if not os.path.exists(filepredALLTest):
    cont = 0
    for i,j in zip(Xtt,Ytt):
        print("Predicting X: " + str(i)+"/"+str(np.max(Xtt)) + " and Y: " + str(j)+"/"+str(np.max(Ytt)))
        inp = pred[cont,:,:]
        #print(inp.shape)
        predALL[j:j+SW,i:i+SW] = inp

        cont += 1
    np.save(filepredALLTest, predALL)
else:
    predALL = np.load(filepredALLTest)


print(predALL.shape)
print("PredALL Min Max Values")
print(np.min(predALL), np.max(predALL))

######################## Binarize IMAGE PREDICTs
PREDALLB = predALL.copy()
THRB = 0.5 #np.median(PREDALLB)
PREDALLB[PREDALLB >= THRB] = 1.0
PREDALLB[PREDALLB < THRB] = 0.0

######################## Plot Image segment PREDICT
from plots_vis_segm import *

idtest = np.array(df_AMPtest["IDline"])
DEPTHtest = DEPTH[idtest]
plotIMG_SegmResult(DEPTHtest, AMP, xtestM, PREDALLB,Qtrain=Qtrain,range=[600,1200], pathReport=pathReport)

###########
## Calculating Percentual Area Line MGP and HPM

PLtest = np.zeros((xtestM.shape[0],))
PLpred = np.zeros((xtestM.shape[0],))
for i in range(xtestM.shape[0]):
    LINE = xtestM[i,:]
    PLtest[i] = np.sum(LINE) / len(LINE)
    LINEP = PREDALLB[i,:]
    PLpred[i] = np.sum(LINEP) / len(LINEP)

plotIMGandCurves_SegmResult(DEPTHtest, AMP, xtestM, PREDALLB, PLtest,PLpred,Qtrain=Qtrain,range=[600,1200], pathReport=pathReport)
# depth, AMP,xtestM, ytest,pred,Qtrain=True,range=[600,1200], pathReport=os.getcwd()


