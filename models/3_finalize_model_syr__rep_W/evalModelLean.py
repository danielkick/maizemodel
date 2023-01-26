# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import shutil 
import os
import re
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

## -------------------------------------------------------------------------- ##
# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# -
# ## Cuda settings:

# ## Data Prep:

# +
# Specified Options -----------------------------------------------------------
## General ====
projectName = 'kt' # this is the name of the folder which will be made to hold the tuner's progress.
hp = kt.HyperParameters()
## Tuning ====
splitIndex = 0
maxTrials  = 40  
numEpochs  = 1000 
# kfolds     = 10 
cvSeed     = 646843
max_pseudorep = 10 # <-- New. How many times should the model be run to account for random initialization? 
## Model ====
# set needed data
needGpca = True
needS    = True
needW    = True

# I don't foresee a benefit to constraining all HP optimizations to follow the same path of folds.
# To make the traning more repeatable while still allowing for a pseudorandom walk over the folds 
# we define the rng up front and will write it out in case it is useful in the future. 
random_cv_starting_seed = int(round(np.random.uniform()*1000000))
k_cv_rng = np.random.default_rng(random_cv_starting_seed)
with open('random_cv_starting_seed.json', 'w') as f:
    json.dump({'random_cv_starting_seed':random_cv_starting_seed}, f)


# +
# Index prep -------------------------------------------------------------------
path     = '../../' #atlas version
pathData = path+'data/processed/'

indexDictList = json.load(open(pathData+'indexDictList_syr.txt')) 

trainIndex    = indexDictList[splitIndex]['Train']
trainGroups   = indexDictList[splitIndex]['TrainGroups']
testGroups    = indexDictList[splitIndex]['TestGroups'] 
testIndex     = indexDictList[splitIndex]['Test']
# -

# Data prep -------------------------------------------------------------------
# Y ===========================================================================
Y = np.load(pathData+'Y.npy')
# G ===========================================================================
if needGpca:
    G = np.load(pathData+'G_PCA_1.npy') 
# S ===========================================================================
if needS:
    S = np.load(pathData+'S.npy')
# W =========================================================================== 
if needW:
    W = np.load(pathData+'W.npy')
data_list = [G, S, W]

"""
Testing:
1. Are there NaN in the genomic data that should not be so?
"""
all_idx = trainIndex+testIndex
print("Total indices ---", len(all_idx))
print("+ctrl NaNs in G -", np.isnan(G).sum())
print("   In selection -", np.isnan(G[all_idx]).sum())
print("")
assert 0 == np.isnan(G[all_idx]).sum(), 'Halted if there were nans found.'


nTestGroups = len(set(testGroups))
x = data_list
y = Y

# +
i = 0
cvFoldCenterScaleDict = {}

# Calculate values for center/scaling each fold
ith_tr_idx = trainIndex

YStd  = y[ith_tr_idx].std()
YMean = y[ith_tr_idx].mean()

GStd  = np.apply_along_axis(np.nanstd,  0, G[ith_tr_idx])
GMean = np.apply_along_axis(np.nanmean, 0, G[ith_tr_idx])

SStd  = np.apply_along_axis(np.std,  0, S[ith_tr_idx])
SMean = np.apply_along_axis(np.mean, 0, S[ith_tr_idx])

# we want to center and scale _but_ some time x feature combinations have an sd of 0. (e.g. nitrogen application)
# to get around this we'll make it so that if sd == 0, sd = 1.
WStd  = np.apply_along_axis(np.std,  0, W[ith_tr_idx])
WStd[WStd == 0] = 1
WMean = np.apply_along_axis(np.mean, 0, W[ith_tr_idx])

cvFoldCenterScaleDict.update({i:{'YStd' : YStd,
                                 'YMean': YMean,
                                 'GStd' : GStd,
                                 'GMean': GMean,
                                 'SStd' : SStd,
                                 'SMean': SMean,
                                 'WStd' : WStd,
                                 'WMean': WMean}})

# +
# Training loop ----

foldwise_data_list = []

"""
This is now a list of tensors instead of one tensor (to prep for the big model)
We'll do this a little different by iterating over the inputs. Then we can be agnostic as to 
what inputs are going in -- just going off position.

This will need to happen for the k fold evaluation below
"""
x_train = [x_tensor[trainIndex] for x_tensor in x] 
x_test  = [x_tensor[testIndex]  for x_tensor in x] 
y_train = y[trainIndex] 
y_test  = y[testIndex]

# Center and scale testing and training data: ---------------------------------
## Training data: =============================================================
y_train = (y_train - cvFoldCenterScaleDict[i]['YMean']) / cvFoldCenterScaleDict[i]['YStd']

x_train[0] = ((x_train[0] - cvFoldCenterScaleDict[i]['GMean']) / cvFoldCenterScaleDict[i]['GStd'])
x_train[0] =   x_train[0].astype('float32')

x_train[1] = ((x_train[1] - cvFoldCenterScaleDict[i]['SMean']) / cvFoldCenterScaleDict[i]['SStd'])
x_train[1] =   x_train[1].astype('float32')

x_train[2] = ((x_train[2] - cvFoldCenterScaleDict[i]['WMean']) / cvFoldCenterScaleDict[i]['WStd'])
x_train[2] =   x_train[2].astype('float32')

## Test | Validation data: ====================================================
y_test = (y_test - cvFoldCenterScaleDict[i]['YMean']) / cvFoldCenterScaleDict[i]['YStd']

x_test[0] = ((x_test[0] - cvFoldCenterScaleDict[i]['GMean']) / cvFoldCenterScaleDict[i]['GStd'])
x_test[0] =   x_test[0].astype('float32')

x_test[1] = ((x_test[1] - cvFoldCenterScaleDict[i]['SMean']) / cvFoldCenterScaleDict[i]['SStd'])
x_test[1] =   x_test[1].astype('float32')

x_test[2] = ((x_test[2] - cvFoldCenterScaleDict[i]['WMean']) / cvFoldCenterScaleDict[i]['WStd'])
x_test[2] =   x_test[2].astype('float32')

foldwise_data_list.append({'train':{'y':y_train,
                                    'x':x_train},
                            'test':{'y':y_test,
                                    'x':x_test} 
                      })
# -

# ## Evaluate Model

x_train = foldwise_data_list[0]['train']['x']
y_train = foldwise_data_list[0]['train']['y']
x_test  = foldwise_data_list[0]['test' ]['x']
y_test  = foldwise_data_list[0]['test' ]['y']

# +
model_files = [e for e in os.listdir('eval') if not re.findall('checkpoint|history|predictions' , e)]

# model_file = model_files[0] # fixme
for model_file in model_files:
    path = './eval/'+model_file
    model = keras.models.load_model(path)

    yHat_train = model.predict(x_train)
    yHat_test = model.predict(x_test)
    print('./eval/'+model_file.replace('.h5', '_predictions.json'))
    ## Save out Settings & Predicitons ==================================================
    with open('./eval/'+model_file.replace('.h5', '_predictions.json'), 'w') as f:
        json.dump({
        'use_tf_model_rep': model_file,
        'y_train': y_train,
        'yHat_train': yHat_train,
        'y_test': y_test,
        'yHat_test': yHat_test
        }, f, cls = NpEncoder)
