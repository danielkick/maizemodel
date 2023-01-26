# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
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
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import keras_tuner as kt

import pickle as pkl
# useful pattern for speeding up runs ----v
# path = "./data_intermediates/cvFoldCenterScaleDict.p"
# if os.path.exists(path):
#     cvFoldCenterScaleDict = pkl.load(open(path, 'rb'))
# else:
#     pkl.dump(cvFoldCenterScaleDict, open(path, 'wb'))

import sklearn
from sklearn.preprocessing import OneHotEncoder

# Reference:
# https://github.com/hyperopt/hyperopt/blob/master/tutorial/02.MultipleParameterTutorial.ipynb
# Import HyperOpt Library
# from hyperopt import tpe, hp, fmin

import pickle


# + code_folding=[19, 41, 63]
## Settings ----------------------------------------------------------------- ##
# data_list_index = 0   # Genotype
# data_list_index = 1   # Soil
# data_list_index = 2   # DON'T USE. This is the W 3d tensor. It will not work.
data_list_index = 3   # weather catagorical, one hot encoded
# data_list_index = 4   # <- this does not exist but caused 0,1,3 to be concatenated.

n_folds = 10
max_evals = 250
n_cores = 1




use_sklearn_model = 'hps_rf'

n_jobs= 1


if use_sklearn_model == 'hps_knn':
    if data_list_index == 0:   # G
        params = {
            'current_weight_metric':'uniform',
            'current_k': 237
        }
    elif data_list_index == 1: # S
        params = {
            'current_weight_metric':'distance',
            'current_k': 248
        }              
    elif data_list_index == 3: # W1
        params = {
            'current_weight_metric':'uniform',
            'current_k': 248
        }        
    elif data_list_index == 4: # Cat
        params = {
            'current_weight_metric':'distance',
            'current_k': 49
        }              
        
if use_sklearn_model == 'hps_rf':
    if data_list_index == 0:   # G
        params = {
            'current_max_depth':64,
            'current_min_samples_leaf':171 
        }    
    elif data_list_index == 1: # S
        params = {
            'current_max_depth':10,
            'current_min_samples_leaf':163 
        }          
    elif data_list_index == 3: # W1
        params = {
            'current_max_depth':102,
            'current_min_samples_leaf': 100
        }          
    elif data_list_index == 4: # Cat
        params = {
            'current_max_depth':7,
            'current_min_samples_leaf':149 
        }                  
        
if use_sklearn_model == 'hps_rnr':
    if data_list_index == 0:   # G
        params = {
            'current_weight_metric':'distance',
            'current_radius': 39.759518
        }
    elif data_list_index == 1: # S
        params = {
            'current_weight_metric':'distance',
            'current_radius': 3.406197
        }
        
    elif data_list_index == 3: # W1
        params = {
            'current_weight_metric':'uniform',
            'current_radius': 5.986679
        }        
    elif data_list_index == 4: # Cat
        params = {
            'current_weight_metric':'distance',
            'current_radius': 40.375418 
        }


if use_sklearn_model == 'hps_svrl':
    if data_list_index == 0:   # G
        params = {
            'current_loss': 'epsilon_insensitive',
            'current_C': 2.772318
        }
    elif data_list_index == 1: # S
        params = {
            'current_loss': 'epsilon_insensitive',
            'current_C': 5.613996 
        }
    elif data_list_index == 3: # W1
        params = {
            'current_loss': 'epsilon_insensitive',
            'current_C': 4.623351
        }
    elif data_list_index == 4: # Cat

        params = {
            'current_loss': 'squared_epsilon_insensitive',
            'current_C': 2.787589
        }



# This is bad practice, but hey it works.
# sub_dir is not passed into fmin via opt_this_fcn, but is found in a higher 
# scope. It is the output directory to be used and it should contain files only 
# for a certain data processing.
index_to_name_suffix = { 0:'G', 1:'S', 3:'WOneHot', 4:'All' }
sub_dir =  'hps_res_'+index_to_name_suffix[data_list_index] 
if not os.path.exists('./'+sub_dir):
    os.mkdir('./'+sub_dir)
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



# + [markdown] heading_collapsed=true
# ## Cuda settings:

# + hidden=true



# + hidden=true
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
# Specified Options -----------------------------------------------------------
## General ====
projectName = 'kt' # this is the name of the folder which will be made to hold the tuner's progress.
# hp = kt.HyperParameters()
## Tuning ====
splitIndex = 0
maxTrials  = 40  
numEpochs  = 2000 
kfolds     = 10
cvSeed     = 646843

## Model ====
# set needed data
needGpca = True
needS    = True
needW    = True


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

# ## Cluster weather 

import tslearn
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

# ## Additional Data Prep -- Representing Weather

# Use `tslearn` to cluster weather time series into categories

w_all = data_list[2]
w_all.shape
# w_train == w_all

# +
# Functions for data setup -----------------------------------------------------

# Produces a one to many dictionary where 
# every key is the index of a unique weather observation
# every value is a list of indices with the same weather observation (since there are multiple obs/site )
def mk_weather_dupe_dict(w_all):
    # find all matching indices:
    weather_dupe_dict = {}
    for i in range(w_all.shape[0]):
        # if the dict is empty
        if ( [] == list(weather_dupe_dict.keys()) ):
            weather_dupe_dict.update({i:[i]})
        else:
            equivalent_found = False
            for key in list(weather_dupe_dict.keys()):
    #             print(key)
                equivalent = np.mean(w_all[i, :, :] == w_all[key, :, :])
                if equivalent == 1.0:
                    # if there's an equivalent observation, add the index to the list and break.
                    equivalent_found = True
                    weather_dupe_dict[key].extend([i])
                    break
            # if no equivalent was found, add to dict
            if equivalent_found == False:
                weather_dupe_dict.update({i:[i]})
    return(weather_dupe_dict)



# using the duplicates dictionary and input data create a deduplicated ndarray 
# where the ith index is the ith's dupe_dict key's data. 
def mk_weather_distinct(w_all,
                       weather_dupe_dict):
    # create a duplicate free dataset to work with.
    w_all_distinct = np.zeros((
        len(list(weather_dupe_dict.keys())), 
        w_all.shape[1], 
        w_all.shape[2]  ))

    # fill in the distinct data where key order matches index 0
    for i in range(len(list(weather_dupe_dict.keys()))):
        keys = list(weather_dupe_dict.keys())
        w_all_distinct[i, :, :] = w_all[keys[i], :, :]

    return(w_all_distinct)

# Functions for time series kmeans clustering ----------------------------------

# get time series kmeans model
# model ------ The fitted model. Useful for applying to test data
# clusters --- Predictions of clusters for training data
# silhouette - Silhouette score for choosing a reasonable k
#
# this function hides the mess of fitting and evaluating a model. 
# Now we can get what we need for prediction and evaluation in one easy call.
def _get_ts_kmeans(
    in_data, # = w_train_distinct[:, :, 0], # last index should be a number or :
    in_k, #= 2,
    randSeed
):
    km = TimeSeriesKMeans(n_clusters=in_k,
                          n_init= 4,
                          metric="dtw",
                          # verbose=True,
                          n_jobs = 4, 
                          max_iter_barycenter=10,
                          random_state=randSeed)
    y_pred = km.fit_predict(in_data)
    silhouette = tslearn.clustering.silhouette_score(in_data, y_pred, metric = 'dtw')
    return({"model":km,
            "clusters":y_pred,
            "silhouette":silhouette})


# find the "right" k for one column
# overview - df of k, silhouette
# res_list - list of _get_ts_kmeans output
#
# take a maximum k and input data. Return an overview with the silhouette score for each k 
# and a list of the results from `_get_ts_kmeans()` for each. 
def _test_ks_ts_kmeans(in_data, # = w_train_distinct[:, :, 0],
                       max_k, # = 3
                       randSeed# = randSeed 
                      ):
    possible_ks = [i+2 for i in range(max_k) if i+1 < max_k]
    res_list = [_get_ts_kmeans(in_data = in_data, in_k = k, randSeed = randSeed) for k in possible_ks]
    overview = pd.DataFrame(
        zip(possible_ks,
            [res['silhouette'] for res in res_list]), columns = ['k', 'Silhouette'])
    return({'overview':overview,
           'res_list':res_list})


# try to find the best k for the set. 
def find_best_k(overview): # = res['overview']): # <--- output from _test_ks_ts_kmeans
    silhouetteComparison = overview
    # find the lowest value before the silhouette starts to increase. 
    # If it only decreases we'll use the k for the lowest silhouette score (i.e. the biggest k).

    silhouetteComparison['NextSil'] = list(silhouetteComparison['Silhouette'][1:]) + [0]
    silhouetteComparison['Delta'] = silhouetteComparison['Silhouette'] - silhouetteComparison['NextSil']
    silhouetteComparison['Allow'] = True 

    # find the first k where the silhouette score degrades.
    # disallow all values of k greater than that.
    stopK = silhouetteComparison.loc[silhouetteComparison.Delta < 0, 'k']
    if stopK.shape[0] > 0:
        silhouetteComparison.loc[silhouetteComparison.k > min(stopK), 'Allow'] = False
    # use the largest k before it regresses
    bestK = max(silhouetteComparison.loc[silhouetteComparison.Allow, 'k'])

    print("The best k found is", bestK)

    # plot as a sanity check
    plt.plot(silhouetteComparison.k, silhouetteComparison.Silhouette)
    plt.scatter(x=bestK, 
                y = list(silhouetteComparison.loc[silhouetteComparison.k == bestK, 'Silhouette'])[0], 
                color = 'red')

    return(bestK)


# wrapper function, just to improve readability.
def _get_cluster_for_k(res, # = res,
                      desired_k # = 2
                      ):
    k_idx = (desired_k-2) # convert desired_k to index (k=0,1 disallowed)
    k_cluster = res['res_list'][k_idx]['clusters']
    return(k_cluster)

# for efficiency we deduplicated the weather observations. 
# Now we need to use the dictionary defined earlier (weather_dupe_dict, which matches the 
# a unique site's index in the data to all the indices with identical data) to get cluster
# definitions for each index in the original data.
def _expand_cluster_to_data(
    weather_dupe_dict, # = weather_dupe_dict,
    cluster # = cluster
):

    data_cluster_list = []
    data_idx_list     = []
    data_uniq_group   = [] # for sanity check 

    dict_key_list = list(weather_dupe_dict.keys())
    for i in range(len(dict_key_list)):
        ith_key = dict_key_list[i]
        ith_cluster = cluster[i]

        data_idx_gets_cluster = weather_dupe_dict[ith_key]

        data_uniq_group.extend(np.repeat(ith_key, len(data_idx_gets_cluster)))
        data_idx_list.extend(data_idx_gets_cluster)
        data_cluster_list.extend(np.repeat(ith_cluster, len(data_idx_gets_cluster)))

    res = pd.DataFrame(zip(data_uniq_group, data_idx_list, data_cluster_list), 
                       columns = ['uniq_obs_index', 'data_index', 'cluster'])
    # sort so the cluster column is ready to go. 
    res = res.sort_values('data_index')
    # quick sanity check
    sanity_check = res.groupby('uniq_obs_index'
                              ).agg(cluster_dev = ('cluster', 'std')
                              ).reset_index()
    sanity_check = list(sanity_check.cluster_dev.drop_duplicates())
    # ensure that all unique observations are in one group. 
    # if they are then the standard deviation of the cluster will be 0 
    # unless there is only one obsevation in which case it will be nan.
    sanity_check = [True if (e == 0) | np.isnan(e) else False for e in sanity_check]
    assert False not in sanity_check, 'One unique weather onbservation is in multiple groups'

    return({"res":res,
            "cluster":list(res.cluster)})



# -

# ### Data Setup

# +
# only needs to be run once
path = "./data_intermediates/weather_dupe_dict.p"
if os.path.exists(path):
    weather_dupe_dict = pkl.load(open(path, 'rb'))
else:
    weather_dupe_dict = mk_weather_dupe_dict(w_all)
    pkl.dump(weather_dupe_dict, open(path, 'wb'))
    
w_all_distinct = mk_weather_distinct(w_all, weather_dupe_dict)

# + [markdown] heading_collapsed=true
# ### Demo

# + hidden=true



# + hidden=true
# demo goes here
demo_tslearn = True

if demo_tslearn:
    ith_var = 5
    temp = w_all_distinct[:, :, ith_var]

    plt.imshow(temp)

# + hidden=true
if demo_tslearn:
    path = "./data_intermediates/demo_weather_cluster_res.p"
    if os.path.exists(path):
        res = pkl.load(open(path, 'rb'))
    else:
        # take a selected k, 
        # retrieve the predictions, 
        # expand back to match the orginal observations in the non-distinct data.
        res = _test_ks_ts_kmeans(
            in_data = temp,
            max_k = 5,
            randSeed = 68874)

        pkl.dump(res, open(path, 'wb'))

# + hidden=true
if demo_tslearn:
    best_k = find_best_k(
        overview = res['overview'])

# + hidden=true
if demo_tslearn:
    best_cluster = _get_cluster_for_k(
        res = res, 
        desired_k = best_k)                 

    obs_cluster = _expand_cluster_to_data(
        weather_dupe_dict = weather_dupe_dict,
        cluster = best_cluster)

    # obs_cluster['cluster']

    temp = pd.DataFrame(temp)
    temp['cluster'] = best_cluster
    temp = temp.sort_values('cluster')
    plt.imshow(temp)
# -

# ### Apply workflow to each variable.



path_cluster = "./data_intermediates/w_all_clusters.p"
if os.path.exists(path_cluster):
    w_all_clusters = pkl.load(open(path_cluster, 'rb'))
else:
    max_k = 40 # There are 160 total unique weather observations. 
               # This assumes that at least 4 should make a cluster (on average)

    # Will hold all the cluster results
    w_all_clusters = np.zeros((w_all.shape[0], w_all.shape[2]))   

    for ith_var in tqdm.tqdm( range( w_all_distinct.shape[2]) ):
        temp = w_all_distinct[:, :, ith_var]

        # this is computationally expensive, so if the results already exist don't recalculate them.
        path = "./data_intermediates/weather_cluster_res_col"+str(ith_var)+".p"
        if os.path.exists(path):
            res = pkl.load(open(path, 'rb'))
        else:
            res = _test_ks_ts_kmeans(
                in_data = temp,
                max_k = max_k,
                randSeed = 68874)
            pkl.dump(res, open(path, 'wb'))    

        best_k = find_best_k(
            overview = res['overview'])

        best_cluster = _get_cluster_for_k(
            res = res, 
            desired_k = best_k)                 

        obs_cluster = _expand_cluster_to_data(
            weather_dupe_dict = weather_dupe_dict,
            cluster = best_cluster)

        obs_cluster = _expand_cluster_to_data(
            weather_dupe_dict = weather_dupe_dict,
            cluster = best_cluster)

        # Write results into full set object
        w_all_clusters[:, ith_var] = obs_cluster['cluster'] 
    
    pkl.dump(w_all_clusters, open(path_cluster, 'wb'))    

# +
# One Hot encode weather clusters 
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(w_all_clusters)
w_all_clusters_oneHot = enc.transform(w_all_clusters).toarray()

# Add to data list
data_list.extend([w_all_clusters_oneHot])
# -

w_all_clusters

# +

# Reversing the operation
# enc.inverse_transform(trans_vals)

# -












# ## Run with finalized parameters

# +
# adapted from other 3_
# -

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
# -



# +
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


# For small scale testing/debugging. See default values in _data_prep() within opt_this_fcn()
# for the input list, define a random subset containing n observations
# and filter each of xs and y ndarry/arrays
def _downsample_train_data(
    x_train,
    y_train,
    nTrainObs = 100,
    randSeed = 4683 # for reproducibility -- some draws may contain only one enviroment.
):
    rng = np.random.default_rng(randSeed)
    # randomly downsample the observations to get a new training set.
    train_downsample = rng.choice(
        [i for i in range(y_train.shape[0])], 
        size = nTrainObs, 
        replace = False)

    x_train = [x[train_downsample] for x in x_train]
    y_train = y_train[train_downsample]
    return({"x":x_train,
            "y":y_train})


# + code_folding=[97, 123, 137]
"""
Note! Selecting jth_matrix == 4 will provide a concatenated dataset of 
0, 1, 3 (G, S, WOneHot)
"""
def _data_prep(
    foldwise_data_list = foldwise_data_list,
    ith_fold = 0,
    jth_matrix = 0, # [G, ]
    downsample = False # <- When ready to run change the default to not downsample. 
    # ... other changes here 
    ):
    # foldwise_data_list = foldwise_data_list
    # ith_fold = 0
    # jth_matrix = 0 # [G, ]
    # downsample = True

    x_train = foldwise_data_list[ith_fold]['train']['x']
    y_train = foldwise_data_list[ith_fold]['train']['y']
    x_test  = foldwise_data_list[ith_fold]['test' ]['x']
    y_test  = foldwise_data_list[ith_fold]['test' ]['y']

    if jth_matrix != 4:            
        if downsample:
            res = _downsample_train_data(
                x_train,
                y_train,
                nTrainObs = 500,
                randSeed = 9834273475)
            x_train = res['x'][jth_matrix]
            y_train = res['y']
        else:
            x_train = x_train[jth_matrix]

        x_test = x_test[jth_matrix]
    else:
        if downsample:
            res = _downsample_train_data(
                x_train,
                y_train,
                nTrainObs = 100,
                randSeed = 9834273475)

            data_G       = res['x'][0]
            data_S       = res['x'][1]
            data_WOneHot = res['x'][3]               
            data_All = np.concatenate([data_G, data_S, data_WOneHot], axis = 1)

            x_train = data_All 
            y_train = res['y']
        else:
            data_G       = x_train[0]
            data_S       = x_train[1]
            data_WOneHot = x_train[3]               
            data_All = np.concatenate([data_G, data_S, data_WOneHot], axis = 1)

            x_train = data_All

        data_test_G       = x_test[0]
        data_test_S       = x_test[1]
        data_test_WOneHot = x_test[3]               
        data_test_All = np.concatenate([data_test_G, data_test_S, data_test_WOneHot], axis = 1)

        x_test = data_test_All


    return({
        'x_train':x_train,
         'x_test':x_test,
        'y_train':y_train,
         'y_test':y_test
    })

hps_trial_dict = {}
# Setup #############################################################################


prepped_data = _data_prep(
    foldwise_data_list = foldwise_data_list,
    ith_fold = 0,
    jth_matrix = data_list_index
)
x_train = prepped_data['x_train']
x_test  = prepped_data['x_test']
y_train = prepped_data['y_train']
y_test  = prepped_data['y_test']


## Support Vector Machines ==========================================================   
### SVM Linear Reg. -----------------------------------------------------------------

# svmKernel cache size: For SVC, SVR, NuSVC and NuSVR, the size of the kernel cache has a strong impact on run times for larger problems. If you have enough RAM available, it is recommended to set cache_size to a higher value than the default of 200(MB), such as 500(MB) or 1000(MB).

if use_sklearn_model == 'hps_svrl':
    current_loss = params['current_loss']
    current_C = params['current_C']

    hps_trial_dict.update({'current_loss':current_loss})
    hps_trial_dict.update({'current_C':current_C})

    mod = sklearn.svm.LinearSVR(
        C=current_C,         # Reguralization              
        loss=current_loss,    # loss type
        max_iter = 5000,      # increased because it failed to converge with the default value. 1000
        tol = 1*10**-3        # default 1e-4
    )

## Nearest Neighbors ================================================================
### K Nearest Neighbors -------------------------------------------------------------
if use_sklearn_model == 'hps_knn':
    current_weight_metric = params['current_weight_metric']
    current_k = params['current_k']

    mod = sklearn.neighbors.KNeighborsRegressor(
        n_neighbors= current_k ,
        weights=current_weight_metric,
        n_jobs= n_jobs )

### Radius Neighbors Regressor ------------------------------------------------------
if use_sklearn_model == 'hps_rnr':
    current_weight_metric = params['current_weight_metric']
    current_radius = params['current_radius']

    hps_trial_dict.update({'current_weight_metric':current_weight_metric})
    hps_trial_dict.update({'current_radius':current_radius})

    mod = sklearn.neighbors.RadiusNeighborsRegressor(
        radius= current_radius ,
        weights=current_weight_metric,
        n_jobs= n_jobs )

## Decision Trees ===================================================================
### Random Forest -------------------------------------------------------------------
if use_sklearn_model == 'hps_rf':
    current_max_depth = params['current_max_depth']
    current_min_samples_leaf = params['current_min_samples_leaf']

    hps_trial_dict.update({'current_max_depth':current_max_depth})
    hps_trial_dict.update({'current_min_samples_leaf':current_min_samples_leaf})

    if current_min_samples_leaf > 1:
        current_min_samples_leaf = int(current_min_samples_leaf)

    mod = sklearn.tree.DecisionTreeRegressor(
        criterion='squared_error', 
        splitter='best',                                                  
        max_depth= current_max_depth,               #  <- can be used to reduce the size of the tree for large datasets
        min_samples_split=2, 
        min_samples_leaf= current_min_samples_leaf, # <- can be used to reduce the size of the tree for large datasets
                                                    # has a smoothing effect in regression 
        max_leaf_nodes=None)  # This is left to be unlimited 

# Return to main processing #########################################################    
mod = mod.fit(x_train, y_train)

yHat_train = mod.predict(x_train)
yHat_test = mod.predict(x_test)


# Save out results ##################################################################  
# using selected variables piece together the right name
model_name = 'Best_'+use_sklearn_model+'_'+str(data_list_index
                )+['G', '', 'S', 'WOneHot', 'All'][data_list_index]
pkl_filename = model_name+'.pkl'

## Save out Model ===================================================================
# Saving Model 
with open('./'+sub_dir+'/'+pkl_filename, 'wb') as file:
    pickle.dump(mod, file)
# Loading
# with open(pkl_filename, 'rb') as file:
#     pkl_mod = pickle.load(file)

## Save out Settings & Predicitons ==================================================
model_in_out_json_name = model_name+'.json'
with open('./'+sub_dir+'/'+model_in_out_json_name, 'w') as f:
    json.dump({
        'use_sklearn_model': use_sklearn_model,
        'params': params,
        'n_jobs': n_jobs,
        'y_train': y_train,
        'yHat_train': yHat_train,
        'y_test': y_test,
        'yHat_test': yHat_test
        }, f, cls = NpEncoder)
