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

# ## Evaluate Model

# ### Look at histories



# +
import pandas as pd
import plotly.express as px

dir_contents = os.listdir('./eval')
models_to_load = [entry for entry in dir_contents if re.match('hps_rep_.*_history*', entry)]


def _prep_history(model_to_load):
    rep_i = re.findall('rep_\d+', model_to_load)[0]
    rep_i = re.findall('\d+$', rep_i)[0]
    rep_i = int(rep_i)
    
    with open('./eval/'+model_to_load, 'r') as f:        
        dat = json.load(f)
    dat = pd.DataFrame().from_dict(dat)
    dat = dat.reset_index().rename(columns = {'index':'epoch'})
    dat['rep'] = rep_i
    return(dat)

dat = pd.concat([_prep_history(i) for i in models_to_load])
dat = dat.melt(id_vars = ['rep', 'epoch'], value_vars = ['loss', 'val_loss'])

fig = px.line(dat, x = 'epoch', y = 'value', color = 'variable', facet_col="rep", template = 'plotly_white')
fig.show()
fig.write_image("./loss.svg")
# -

# ### Generate Predictions



# +
# model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)
# -

x_train = foldwise_data_list[0]['train']['x']
y_train = foldwise_data_list[0]['train']['y']
x_test  = foldwise_data_list[0]['test' ]['x']
y_test  = foldwise_data_list[0]['test' ]['y']


def get_model_errors(model_checkpoint = 'hps_rep_0checkpoint00001000.h5'):
    path = './eval/'+model_checkpoint
    model = keras.models.load_model(path)

    yhat_train = model.predict(x_train)
    yhat_test = model.predict(x_test)
    # y_test/train have a dimension of (#, ?). 
    # We set all to be , 1 so that (#, ) doesn't become(#,#) when we subtract it.
    error_train = y_train.reshape((y_train.shape[0], 1)) - yhat_train
    error_test = y_test.reshape((y_test.shape[0], 1)) - yhat_test
    # naive guess
    error_naive_train = y_train.reshape((y_train.shape[0], 1)) - np.mean(y_train)
    error_naive_test = y_test.reshape((y_test.shape[0], 1)) - np.mean(y_train)


    def _error_to_df(in_npy, # = error_train,
                     eval_data = 'Train',
                     model_type =  'dnn'
    ): 
        temp = pd.DataFrame(
                in_npy,
            columns = ['Error'])
        temp['Set'] = np.repeat(eval_data, in_npy.shape[0])
        temp['Type'] = np.repeat(model_type, in_npy.shape[0])
        return(temp)

    # combine into a single df
    errors = [
        error_train, error_test, error_naive_train, error_naive_test]
    split = ['Train',     'Test',           'Train',           'Test']
    key =   ['DNN',        'DNN',       'Intercept',      'Intercept']

    res = [_error_to_df(
        in_npy = errors[i],
        eval_data = split[i],
        model_type = key[i] ) for i in range(len(errors))]

    res = pd.concat(res)
    res['File'] = model_checkpoint
    return({'res':res,
            'model':model})

# +
# final models

# checkpoints
os.listdir('./eval')

# +
temp = get_model_errors(model_checkpoint = 'hps_rep_0.h5')
model = temp['model']
temp = temp['res']
temp =temp.reset_index()
# temp['Type'] = temp['Type']+"_"+temp['Set']  

# todo renaming goes here

# -

model.summary()

import seaborn as sns
sns.displot(data=temp.loc[temp.Set == 'Test'], kind="ecdf", x="Error", #col="Set", 
            hue="Type", rug=True
           )

sns.catplot(data=temp, kind="violin", x="Set", y="Error", hue="Type", split=True)

# +
import matplotlib.pyplot as plt
sns.set(rc={"figure.figsize":(6, 4)})
sns.set_style("whitegrid")

sns.kdeplot(data=temp.loc[temp.Set == 'Test'], x="Error", hue = 'Type', shade = True)


plt.savefig('./G_kde.svg')
