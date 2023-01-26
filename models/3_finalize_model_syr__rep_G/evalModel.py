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
import re
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %% code_folding=[]
# Data prep adapted from model.py
# Specified Options -----------------------------------------------------------
## Tuning ====
splitIndex = 0
numEpochs  = 500
kfolds     = 10
cvSeed     = 646843
max_pseudorep = 10 # <-- New. How many times should the model be run to account for random initialization?

## Model ====
readyModelGPCA = True
readyModelGNUM = False
readyModelS    = False
readyModelWmlp = False
readyModelWc1d = False
readyModelWc2d = False

# set needed data
needGpca = False
needGnum = False
needS    = False
needW    = False

if readyModelGPCA:
    needGpca = True
if readyModelGNUM:
    needGnum = True
if readyModelS:
    needS    = True
if readyModelWmlp:
    needW    = True
if readyModelWc1d:
    needW    = True
if readyModelWc2d:
    needW    = True


# %% code_folding=[]
# Index prep -------------------------------------------------------------------
path     = '../../' #atlas version
pathData = path+'data/processed/'


# get the right index set automatically.
if needGpca == True:
    indexDictList = json.load(open(pathData+'indexDictList_PCA.txt')) # <--- Changed to PCA
else:
    indexDictList = json.load(open(pathData+'indexDictList.txt'))

trainIndex    = indexDictList[splitIndex]['Train']
trainGroups   = indexDictList[splitIndex]['TrainGroups']
testGroups    = indexDictList[splitIndex]['TestGroups'] 
testIndex     = indexDictList[splitIndex]['Test']


# Data prep -------------------------------------------------------------------
# Y ===========================================================================
Y = np.load(pathData+'Y.npy')
YStd = Y[trainIndex].std()
YMean = Y[trainIndex].mean()
YNorm = ((Y - YMean) / YStd)

# G ===========================================================================
if needGpca:
    G = np.load(pathData+'G_PCA_1.npy') # <--- Changed to PCA
    GStd = np.apply_along_axis(np.nanstd, 0, G[trainIndex])
    GMean = np.apply_along_axis(np.nanmean, 0, G[trainIndex])
    GNorm = ((G - GMean) / GStd)
#     GNorm = G
    GNormPCA = GNorm.astype('float32') 
    
## G NUM
if needGnum:
    G = np.load(pathData+'G.npy')
    # No work up here because they're already ready.
    GNorm = G

    GNormNUM = GNorm.astype('int32')

# S ===========================================================================
if needS:
    S = np.load(pathData+'S.npy')
    SStd = np.apply_along_axis(np.std,  0, S[trainIndex])
    SMean = np.apply_along_axis(np.mean, 0, S[trainIndex])
    SNorm = ((S - SMean) / SStd)

    SNorm = SNorm.astype('float32')

# W =========================================================================== 
if needW:
    W = np.load(pathData+'W.npy')
    # we want to center and scale _but_ some time x feature combinations have an sd of 0. (e.g. nitrogen application)
    # to get around this we'll make it so that if sd == 0, sd = 1.
    # Then we'll be able to proceed as normal.
    WStd = np.apply_along_axis(np.std,  0, W[trainIndex])
    WStd[WStd == 0] = 1
    WMean = np.apply_along_axis(np.mean, 0, W[trainIndex])
    WNorm = ((W - WMean) / WStd)

    WNorm = WNorm.astype('float32')
    #WNorm = WNorm.astype('float64') # <- this was set, unclear if it was functional. 

# %%
# mimicing model.py We set up the model to be run (or the text for it in this case) and the data for it together.    
# Model defination ------------------------------------------------------------    
if   readyModelGPCA:
    ## G PCA
    data_list = [GNormPCA[trainIndex]]
    data_list2 = [GNormPCA[testIndex]]    
elif readyModelGNUM:
    ## G NUM   
    data_list = [GNormNUM[trainIndex]]
    data_list2 = [GNormNUM[testIndex]]    
elif readyModelS:
    ## S
    data_list = [SNorm[trainIndex]]
    data_list2 = [SNorm[testIndex]]    
elif readyModelWmlp:
    ## W MLP
    data_list = [WNorm[trainIndex]]
    data_list2 = [WNorm[testIndex]]
elif readyModelWc1d:
    ## W C1D 
    data_list = [WNorm[trainIndex]]
    data_list2 = [WNorm[testIndex]]    
elif readyModelWc2d:
    ## W C2d       
    data_list = [WNorm[trainIndex]]
    data_list2 = [WNorm[testIndex]]    
else: 
    raise NameError("No model has been selected to load.")

# %%
# dir_contents = os.listdir()
# models_to_load = [entry for entry in dir_contents if re.match('trained_model_*', entry)]

# out_dict = {}
# for model_to_load in models_to_load:
#     rep_id = model_to_load.split('_')[-1]

#     model = keras.models.load_model('./'+model_to_load)

#     yhat_train = model.predict(data_list)
#     y_train = YNorm[trainIndex]
#     yhat_test = model.predict(data_list2)
#     y_test = YNorm[testIndex]

#     temp_dict = {
#         rep_id: {
#         "y_train"   : list(y_train),
#         "yhat_train": [entry[0] for entry in yhat_train.tolist()],
#         "y_test"    : list(y_test),
#         "yhat_test" : [entry[0] for entry in yhat_test.tolist()]
#         }}
    
#     out_dict.update(temp_dict)
    
# with open('all_model_res.json', 'w') as f:
#     json.dump(out_dict, f)

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%













































































