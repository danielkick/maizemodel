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
# from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
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
path     = '../data/atlas/' #atlas version 
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
# """
# Additional Testing:
# Has the fold definition code messed up these indices? 
# """
# for key in cvFoldIdxDict.keys():
#     nan_Train = np.isnan(G[cvFoldIdxDict[key]['Train']]).sum()
#     nan_Val   = np.isnan(G[cvFoldIdxDict[key]['Test']]).sum()
#     print('key', key, ': NaN in Train: ', nan_Train, ', NaN in Test: ', nan_Val, sep = '')
#     assert 0 == nan_Train == nan_Val, 'Nans found in dataset'

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






x_train = foldwise_data_list[0]['train']['x']
y_train = foldwise_data_list[0]['train']['y']
x_test  = foldwise_data_list[0]['test' ]['x']
y_test  = foldwise_data_list[0]['test' ]['y']

# ### Load Models, Rebuild if needed.

# +
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tf_keras_vis
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

def score_function(output):
    return output[:, 0]


# -

# #### Genome Only
#

# +
ith_tensor = 0

model_g = tf.keras.models.load_model('../data/atlas/models/3_finalize_model_syr__rep_G/eval/hps_rep_1.h5')

model_g.summary()

# +
model_3_inputs = model_g

remade_model =  keras.Sequential()
remade_model.add(model_3_inputs.get_layer('gIn'))
for i in [2, 3]:
        remade_model.add(model_3_inputs.get_layer('dense'+'_'+str(i)))
        remade_model.add(model_3_inputs.get_layer('batch_normalization'+'_'+str(i)))
        remade_model.add(model_3_inputs.get_layer('dropout'+'_'+str(i)))

remade_model.add(model_3_inputs.get_layer('yHat'))
# -

# Regression Test
model_yHat = model_g.predict(x_train)
remade_model_yHat = remade_model.predict(x_train[ith_tensor])
assert not False in model_yHat == remade_model_yHat, 'Regression test failed! \
                                                      Loaded and recreated model are not equivalent!'

# +
saliency = Saliency(remade_model_yHat, clone=True)


def score_function(output):
    return output[:, 0]

saliency_map = saliency(score_function, (x_test[ith_tensor]), keepdims=True)
# -










# #### Soil Only

# +
ith_tensor = 1

model_s = tf.keras.models.load_model('../data/atlas/models/3_finalize_model_syr__rep_S/eval/hps_rep_0checkpoint00000100.h5')

model_s.summary()

# +
model_3_inputs = model_s

remade_model =  keras.Sequential()
remade_model.add(model_3_inputs.get_layer('sIn'))
for i in range(7):
    if i == 0:
        remade_model.add(model_3_inputs.get_layer('dense'))
        remade_model.add(model_3_inputs.get_layer('batch_normalization'))
        remade_model.add(model_3_inputs.get_layer('dropout'))
    else:
        remade_model.add(model_3_inputs.get_layer('dense'+'_'+str(i)))
        remade_model.add(model_3_inputs.get_layer('batch_normalization'+'_'+str(i)))
        remade_model.add(model_3_inputs.get_layer('dropout'+'_'+str(i)))

remade_model.add(model_3_inputs.get_layer('yHat'))
# -

# Regression Test
model_yHat = model_s.predict(x_train)
remade_model_yHat = remade_model.predict(x_train[ith_tensor])
assert not False in model_yHat == remade_model_yHat, 'Regression test failed! \
                                                      Loaded and recreated model are not equivalent!'

saliency = Saliency(remade_model, clone=False)
saliency_map = saliency(score_function, (x_test[ith_tensor]), keepdims=True)











# #### Weather Only

# +
ith_tensor = 2

model_w = tf.keras.models.load_model('../data/atlas/models/3_finalize_model_syr__rep_W/eval/hps_rep_0.h5')

model_w.summary()
# -



# +
model_3_inputs = model_w

ith_conv = 0 # there are 2 per block. Using a separate index is an easy way to accomplish this.

remade_model =  keras.Sequential()
remade_model.add(model_3_inputs.get_layer('wIn'))

for block in range(6):
    if block == 0:
        if ith_conv == 0:
            remade_model.add(model_3_inputs.get_layer('conv1d'))
            ith_conv += 1
        remade_model.add(model_3_inputs.get_layer('conv1d'+'_'+str(ith_conv)))
        ith_conv += 1        
        remade_model.add(model_3_inputs.get_layer('batch_normalization'))
        remade_model.add(model_3_inputs.get_layer('max_pooling1d'))
    else:
        remade_model.add(model_3_inputs.get_layer('conv1d'+'_'+str(ith_conv)))
        ith_conv += 1        
        remade_model.add(model_3_inputs.get_layer('conv1d'+'_'+str(ith_conv)))
        ith_conv += 1        
        remade_model.add(model_3_inputs.get_layer('batch_normalization'+'_'+str(block)))
        remade_model.add(model_3_inputs.get_layer('max_pooling1d'+'_'+str(block)))       

remade_model.add(model_3_inputs.get_layer('flatten'))
 
# -

# Regression Test
model_yHat = model_w.predict(x_train)
remade_model_yHat = remade_model.predict(x_train[ith_tensor])
assert not False in model_yHat == remade_model_yHat, 'Regression test failed! \
                                                      Loaded and recreated model are not equivalent!'

saliency = Saliency(remade_model, clone=False)
saliency_map = saliency(score_function, (x_test[ith_tensor]), keepdims=True)



# +
smap_mean = saliency_map.numpy().mean(axis = 0)

smap_std  = saliency_map.numpy().std(axis = 0)

# +
import matplotlib.pyplot as plt

plt.imshow(smap_std)
# -

saliency_map



# +
import matplotlib.pyplot as plt

# plt.imshow(saliency_map[1])

# +
import pandas as pd

df = pd.DataFrame(saliency_map[0].numpy())#.reset_index()
plt.imshow(pd.DataFrame(df.mean()).T.loc[:,0:50])

# -

df = pd.DataFrame(saliency_map[1].numpy())#.reset_index()
plt.imshow(pd.DataFrame(df.mean()).T)

plt.hist(pd.DataFrame(df.mean()))

# +

plt.imshow(saliency_map[2].numpy().mean(axis = 0).T)



# +



import plotly.express as px

fig = px.imshow(saliency_map[2].numpy().mean(axis = 0).T)
fig.show()
# -














# #### Concatenated Model

# +
# Placeholder

# +
import pandas as pd

phenotypeGBS = pd.read_pickle('../data/processed/phenotype_with_GBS.pkl')
phenotypeGBS.shape
# -

(Y.shape, G.shape, S.shape, W.shape, )

phenotype = pd.read_csv('../data/processed/tensor_ref_phenotype.csv')
soil = pd.read_csv('../data/processed/tensor_ref_soil.csv')
weather = pd.read_csv('../data/processed/tensor_ref_weather.csv')

phenotype.shape

# #### Full Model

model = tf.keras.models.load_model('../data/atlas/models/3_finalize_model_syr__rep_full/eval/hps_rep_0checkpoint00000100.h5')

saliency = Saliency(model, clone=False)
saliency_map = saliency(score_function, (x_test[0], x_test[1], x_test[2]), keepdims=True)

saliency_map

# ### Generate Predictions



# +
# model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)
# -





# model_checkpoint = 'hps_rep_0checkpoint00001000.h5'
path = '../3_finalize_model_syr__rep_full/eval/hps_rep_0checkpoint00000100.h5'#+model_checkpoint # fixme!
path = './eval/hps_rep_0checkpoint00000100.h5'#+model_checkpoint # fixme!
model = tf.keras.models.load_model(path)



# +
saliency = Saliency(model, clone=False)


# -

[   sum(sum(x_test[0] == np.nan)),
    sum(sum(x_test[1] == np.nan)),
sum(sum(sum(x_test[2] == np.nan)))
]

sum(model.predict(x_test) == np.nan)

model.summary()

# +
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

remade_model =  keras.Sequential()
remade_model.add(model.get_layer('sIn'))
for i in range(7):
    if i == 0:
        remade_model.add(model.get_layer('dense'))
        remade_model.add(model.get_layer('batch_normalization'))
        remade_model.add(model.get_layer('dropout'))
    else:
        remade_model.add(model.get_layer('dense'+'_'+str(i)))
        remade_model.add(model.get_layer('batch_normalization'+'_'+str(i)))
        remade_model.add(model.get_layer('dropout'+'_'+str(i)))

remade_model.add(model.get_layer('yHat'))

# +
# remade_model.predict(x_train[1])


saliency = Saliency(remade_model, clone=False)
saliency_map = saliency(score_function, (x_test[1]), keepdims=True)



# model.get_layer('dense_6').get_weights()
# model.get_layer('dense_6')

# +
# todo TRY rebuilding the keras model using the trained weights to edit the inputs. Then pass in the right tensor. 
# my suspicision is that haveing inputs that aren't used is what's causing problems -- changes in those tensors has no effect on the gradient.

# +
# saliency_map = saliency(score_function, (x_test[0], x_test[1], x_test[2]), keepdims=True) # Worked for full network

saliency_map = saliency(score_function, (x_test[0], x_test[1], x_test[2]), keepdims=True)
# -

len(saliency_map)




