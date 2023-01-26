# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os
import json
import re

import numpy as np
import tensorflow as tf
import keras
from keras import layers

import keras_tuner as kt
# custom imports
# from build_model import build_model

# %%
#TODO find a way to read in these. Separate files?
# the class works best if we can just have it included. It shouldn't change or shouldn't change much so that's not a problem. 

# Based on https://github.com/keras-team/keras-tuner/issues/122
class TunerGroupwiseCV(kt.engine.tuner.Tuner):
    def run_trial(self, trial, x, y,
                  nTestGroups,# = len(set(testGroups)),
                  kfolds,# = 2,     # <- We specify the number of desired folds because
                                  # the number of groups in the training set my not be evenly divsible by
                                  # the number of groups in the true test set.
                  cvSeed,# = 646843,
                  trainGroups,# = trainGroups,

#                   batch_size,#=32,
                  epochs,#=1,
                  *fit_args, **fit_kwargs
                 ):
        # Fold setup goes here ----
        cvGroup = list(set(trainGroups))
        rng = np.random.default_rng(cvSeed)
        valHoldoutDict = {}
        ## make the validation fold group membership for each fold. ====
        for i in range(kfolds):
            rng.shuffle(cvGroup)
            valHoldoutDict.update({str(i):cvGroup[0:nTestGroups]})
        ## This is what will be used for the test/train evaluation ====
        cvFoldIdxDict = {}
        for i in valHoldoutDict.keys():
            foldTrainIdx = [idx for idx in range(len(trainIndex)) if trainGroups[idx] not in valHoldoutDict[i]]
            foldTestIdx =  [idx for idx in range(len(trainIndex)) if trainGroups[idx]     in valHoldoutDict[i]]
            cvFoldIdxDict.update({i:{'Train':foldTrainIdx,'Test':foldTestIdx}})
        val_losses = []

        # Training loop ----
        for i in cvFoldIdxDict.keys():
            # TODO I suppose we should be rescaling all data here instead of outside
            #      to reduce information leaking from validataion -> training.
            #      If we don't it'll potentially cause underperformance instead of
            #      overperformance so this isn't a critical todo.
            x_train, x_test = x[cvFoldIdxDict[i]['Train']], x[cvFoldIdxDict[i]['Test']]
            y_train, y_test = y[cvFoldIdxDict[i]['Train']], y[cvFoldIdxDict[i]['Test']]

            model = self.hypermodel.build(trial.hyperparameters)
            
            fit_kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32) # <---------- 
            model.fit(x_train, y_train,
#                       batch_size= batch_size,
                      epochs=epochs
                     )
            val_losses.append(model.evaluate(x_test, y_test))
        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
        self.save_model(trial.trial_id, model)


# %%
def get_cv_mae(y,
                     x,
                     nTestGroups
):
    # This is adapted from the training code
    cvGroup = list(set(trainGroups))
    rng = np.random.default_rng(cvSeed)
    valHoldoutDict = {}
    ## make the validation fold group membership for each fold. ====
    for i in range(kfolds):
        rng.shuffle(cvGroup)
        valHoldoutDict.update({str(i):cvGroup[0:nTestGroups]})
    ## This is what will be used for the test/train evaluation ====
    cvFoldIdxDict = {}
    for i in valHoldoutDict.keys():
        foldTrainIdx = [idx for idx in range(len(trainIndex)) if trainGroups[idx] not in valHoldoutDict[i]]
        foldTestIdx =  [idx for idx in range(len(trainIndex)) if trainGroups[idx]     in valHoldoutDict[i]]
        cvFoldIdxDict.update({i:{'Train':foldTrainIdx,'Test':foldTestIdx}})
    val_maes = []
    naive_maes = []

    # Evaluation loop ----
    for i in cvFoldIdxDict.keys():
        x_train, x_test = x[cvFoldIdxDict[i]['Train']], x[cvFoldIdxDict[i]['Test']]
        y_train, y_test = y[cvFoldIdxDict[i]['Train']], y[cvFoldIdxDict[i]['Test']]
        
        naive_mae = np.mean(abs(y_test - np.mean(y_train)))
        naive_maes = naive_maes  + [naive_mae]      

        yhat = best_model.predict(x_test)
        mean_yhat = np.mean(abs(yhat - y_test))
        val_maes = val_maes + [mean_yhat]
        
    resDict = {"cv_maes": val_maes,
               # don't calculate mean here so that we can account for training fold overperformance
               # "mean_cv_mae": np.mean(val_maes), 
               "naive_maes": naive_maes,
               # "mean_naive_mae": np.mean(naive_maes)
              }
    return(resDict)


# %%

# %%
# # Load Tuner
# path = '../data/atlas/'
# # path = '../../'         #atlas version
# # Specified Options -----------------------------------------------------------
# projectName = 'kt' # this is the name of the folder which will be made to hold the tuner's progress.
# maxTrials = 2

# hp = kt.HyperParameters()

# model = build_model(hp)
# model.summary()

# tuner = TunerGroupwiseCV(
#     hypermodel=build_model,
#     oracle=kt.oracles.BayesianOptimization(
#         objective='val_loss',
#         max_trials= maxTrials),    
#     directory=".",
#     project_name=projectName,
# )

# %%
splitIndex = 0

kfolds =     10

cvSeed = 646843

needGNum = True
needGPCA = True
needS    = True
needW    = True

# load in data for evaluation
path     = '../data/atlas/' #local version
pathData = path+'data/processed/'

indexDictList = json.load(open(pathData+'indexDictList.txt'))
trainIndex =  indexDictList[splitIndex]['Train']
testIndex =   indexDictList[splitIndex]['Test']
trainGroups = indexDictList[splitIndex]['TrainGroups']
testGroups =  indexDictList[splitIndex]['TestGroups'] # Don't actually need this until we evaluate the tuned model.



# indexDictList = json.load(open(pathData+'indexDictList_PCA.txt'))
# trainIndex =  indexDictList[splitIndex]['Train']
# testIndex =   indexDictList[splitIndex]['Test']
# trainGroups = indexDictList[splitIndex]['TrainGroups']
# testGroups =  indexDictList[splitIndex]['TestGroups'] # Don't actually need this until we evaluate the tuned model.



# Y ===========================================================================
Y = np.load(pathData+'Y.npy')
YStd = Y[trainIndex].std()
YMean = Y[trainIndex].mean()
YNorm = ((Y - YMean) / YStd)


# G ===========================================================================
# if needGNum:
#     G = np.load(pathData+'G.npy')
#     # No work up here because they're already ready.
#     GNorm = G

#     GNorm = GNorm.astype('int32')
#     GNormNum = GNorm.copy()
    
# G ===========================================================================
# if needGPCA:
#     G = np.load(pathData+'G_PCA_1.npy') # <--- Changed to PCA
#     GStd = np.apply_along_axis(np.nanstd, 0, G[trainIndex])
#     GMean = np.apply_along_axis(np.nanmean, 0, G[trainIndex])
#     GNorm = ((G - GMean) / GStd)    

#     GNorm = GNorm.astype('float32')
#     GNormPCA = GNorm.copy()
        
# S ===========================================================================
if needS:
    S = np.load(pathData+'S.npy')
    SStd = np.apply_along_axis(np.std,  0, S[trainIndex])
    SMean = np.apply_along_axis(np.mean, 0, S[trainIndex])
    SNorm = ((S - SMean) / SStd)

    SNorm = SNorm.astype('float32')

# W ===========================================================================
# if needW:
#     W = np.load(pathData+'W.npy')
#     # we want to center and scale _but_ some time x feature combinations have an sd of 0. (e.g. nitrogen application)
#     # to get around this we'll make it so that if sd == 0, sd = 1.
#     # Then we'll be able to proceed as normal.
#     WStd = np.apply_along_axis(np.std,  0, W[trainIndex])
#     WStd[WStd == 0] = 1
#     WMean = np.apply_along_axis(np.mean, 0, W[trainIndex])
#     WNorm = ((W - WMean) / WStd)
# #     WNorm = WNorm.astype('float32')
#     WNorm = WNorm.astype('float64')

# %% [markdown]
# # Test generated sets
#
#
# If the outputs are equivalent, evidence would suggest that the rng is correctly set.

# %%
nTestGroups = 3

# This is adapted from the training code
cvGroup = list(set(trainGroups))
rng = np.random.default_rng(cvSeed)
valHoldoutDict = {}
## make the validation fold group membership for each fold. ====
for i in range(kfolds):
    rng.shuffle(cvGroup)
    valHoldoutDict.update({str(i):cvGroup[0:nTestGroups]})
## This is what will be used for the test/train evaluation ====
cvFoldIdxDict = {}
for i in valHoldoutDict.keys():
    foldTrainIdx = [idx for idx in range(len(trainIndex)) if trainGroups[idx] not in valHoldoutDict[i]]
    foldTestIdx =  [idx for idx in range(len(trainIndex)) if trainGroups[idx]     in valHoldoutDict[i]]
    cvFoldIdxDict.update({i:{'Train':foldTrainIdx,'Test':foldTestIdx}})


# %%
# This is adapted from the training code
cvGroup = list(set(trainGroups))
rng = np.random.default_rng(cvSeed)
valHoldoutDict = {}
## make the validation fold group membership for each fold. ====
for i in range(kfolds):
    rng.shuffle(cvGroup)
    valHoldoutDict.update({str(i):cvGroup[0:nTestGroups]})
## This is what will be used for the test/train evaluation ====
cvFoldIdxDict2 = {}
for i in valHoldoutDict.keys():
    foldTrainIdx = [idx for idx in range(len(trainIndex)) if trainGroups[idx] not in valHoldoutDict[i]]
    foldTestIdx =  [idx for idx in range(len(trainIndex)) if trainGroups[idx]     in valHoldoutDict[i]]
    cvFoldIdxDict2.update({i:{'Train':foldTrainIdx,'Test':foldTestIdx}})


# %%
print('| k | TrainEq | TestEq |')
print('|---|---------|--------|')
for i in range(kfolds):
    trainEq = cvFoldIdxDict[str(i)]['Train'] == cvFoldIdxDict2[str(i)]['Train'] 
    testEq = cvFoldIdxDict[str(i)]['Test'] == cvFoldIdxDict2[str(i)]['Test']
    print('| '+str(i)+' |  '+str(trainEq)+'   |  '+str(testEq)+'  | ')

# %%

# %%

# %%

# %%

# %%

# %%

# %%
