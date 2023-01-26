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

# This notebook is a quick reshaping of the tensors used for fitting DNNs into something that R can use.
# This is modified from the data prep used in DNN fitting.

# +
import shutil 
import os
import re
import json

import numpy as np
import pandas as pd
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import keras_tuner as kt
# -


# ## Cuda settings:

# +
# Set Training options, these will differ for lambda and Atlas:
# https://stackoverflow.com/questions/53533974/how-do-i-get-keras-to-train-a-model-on-a-specific-gpu
# os.environ["CUDA_VISIBLE_DEVICES"]="0" # first gpu
# os.environ["CUDA_VISIBLE_DEVICES"]="1" # second gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # runs in cpu

# Other possible settings:
#https://github.com/tensorflow/tensorflow/issues/14475
#gpu_fraction = 0.1
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Directory prep -------------------------------------------------------------- 
## Automatic file manipulaiton logic goes here ================================       
# clean out logs
# if os.path.exists('./logs/training_log.csv'): 
#     os.remove('./logs/training_log.csv')
# if os.path.exists('./logs/tb_logs'):
#     shutil.rmtree('./logs/tb_logs') # likely has files in it which is why we don't use os.rmdir().

## Load additional options files here =========================================       
# check that everthing is okay. otherwise write out an error file with exception -- naming inconsistency. 
# modelDir = os.getcwd().split(os.path.sep)[-1]
# use_cvFold_idx = int(modelDir.split("_")[-1].strip("k"))
# -


# ## Data Prep:

# +
# Directory prep -------------------------------------------------------------- 
## Automatic file manipulaiton logic goes here ================================       
# clean out logs
# if os.path.exists('./logs/training_log.csv'): 
#     os.remove('./logs/training_log.csv')
# if os.path.exists('./logs/tb_logs'):
#     shutil.rmtree('./logs/tb_logs') # likely has files in it which is why we don't use os.rmdir().

# +
## Load additional options files here =========================================       
# check that everthing is okay. otherwise write out an error file with exception -- naming inconsistency. 
# modelDir = os.getcwd().split(os.path.sep)[-1]
# use_cvFold_idx = int(modelDir.split("_")[-1].strip("k"))

# +
# Specified Options -----------------------------------------------------------
## General ====
# projectName = 'kt' # this is the name of the folder which will be made to hold the tuner's progress.
# hp = kt.HyperParameters()
## Tuning ====
splitIndex = 0
# maxTrials  = 40  
# numEpochs  = 12 
# kfolds     = 10 
cvSeed     = 646843
# max_pseudorep = 10 # <-- New. How many times should the model be run to account for random initialization? 
## Model ====
# set needed data
needGpca = True
needS    = True
needW    = True

# I don't foresee a benefit to constraining all HP optimizations to follow the same path of folds.
# To make the traning more repeatable while still allowing for a pseudorandom walk over the folds 
# we define the rng up front and will write it out in case it is useful in the future. 
# random_cv_starting_seed = int(round(np.random.uniform()*1000000))
# k_cv_rng = np.random.default_rng(random_cv_starting_seed)
# with open('random_cv_starting_seed.json', 'w') as f:
#     json.dump({'random_cv_starting_seed':random_cv_starting_seed}, f)


# +
# Index prep -------------------------------------------------------------------
# path     = '../../' #atlas version
path     = '../' #local version
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
# data_list = [G, S, W]

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




# Reduce dataset size with rows being train,test
usedDataIndex = trainIndex+testIndex
G = G[usedDataIndex, ]
S = S[usedDataIndex, ]
W = W[usedDataIndex, ]
Y = Y[usedDataIndex]

# +
# get the names of each covariate
soilCovNames = [e for e in list(pd.read_csv(pathData+'tensor_ref_soil.csv')
                               ) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]

weatherCovNames = [e for e in list(pd.read_csv(pathData+'tensor_ref_weather.csv')
                                  ) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]
# -



# +
## Easy tensors to convert
G_for_R = pd.DataFrame(G, columns=["PC"+str(1+e) for e in range(G.shape[1])])

assert S.shape[1] == len(soilCovNames)
S_for_R = pd.DataFrame(S, columns=soilCovNames)

## Slightly more involved
### Weather
covs_in_tensor = W.shape[2]
days_in_tensor = W.shape[1]

assert covs_in_tensor == len(weatherCovNames)
W_slices = [pd.DataFrame(W[:, :, i], columns = [weatherCovNames[i]+"_DPP"+str(-75+j) for j in range(days_in_tensor)]) for i in range(covs_in_tensor)]
W_for_R = pd.concat(W_slices, axis=1)

### Indexing
Idx_for_R = pd.DataFrame(
    zip(['Train' for i in trainIndex]+['Test' for i in testIndex],
        trainIndex+testIndex#,
        #[i+1 for i in trainIndex]+[i+1 for i in testIndex]
       ),
    columns = ["Set", "PythonIndex"#, "RIndex"
              ]
)

### Test that the indexes are correct
# G_Check_Idx_for_R = pd.DataFrame(G[testIndex, 0:3], columns=["PC"+str(1+e) for e in range(3)])
# G_Check_Idx_for_R['Set'] = 'Test'



# +
# "../data/processed/"
# S_for_R
# covariatesForR
# -



G_for_R.to_csv(pathData+"ForRG.csv")

S_for_R.to_csv(pathData+"ForRS.csv") 

W_for_R.to_csv(pathData+"ForRW.csv")

Idx_for_R.to_csv(pathData+"ForRIdx.csv")

pd.DataFrame(Y, columns = ['Y']).to_csv(pathData+"ForRY.csv")

# +
# G_Check_Idx_for_R.to_csv(pathData+"ForRGIdxCheck.csv")
# -




