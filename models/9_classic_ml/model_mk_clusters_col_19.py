#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# ## Cuda settings:

# In[ ]:





# In[2]:


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


# ## Data Prep:

# In[3]:


# Directory prep -------------------------------------------------------------- 
## Automatic file manipulaiton logic goes here ================================       
# clean out logs
# if os.path.exists('./logs/training_log.csv'): 
#     os.remove('./logs/training_log.csv')
# if os.path.exists('./logs/tb_logs'):
#     shutil.rmtree('./logs/tb_logs') # likely has files in it which is why we don't use os.rmdir().


# In[4]:


## Load additional options files here =========================================       
# check that everthing is okay. otherwise write out an error file with exception -- naming inconsistency. 
# modelDir = os.getcwd().split(os.path.sep)[-1]
# use_cvFold_idx = int(modelDir.split("_")[-1].strip("k"))


# In[5]:


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

# I don't foresee a benefit to constraining all HP optimizations to follow the same path of folds.
# To make the traning more repeatable while still allowing for a pseudorandom walk over the folds 
# we define the rng up front and will write it out in case it is useful in the future. 
random_cv_starting_seed = int(round(np.random.uniform()*1000000))
k_cv_rng = np.random.default_rng(random_cv_starting_seed)
with open('random_cv_starting_seed.json', 'w') as f:
    json.dump({'random_cv_starting_seed':random_cv_starting_seed}, f)


# In[6]:


# Index prep -------------------------------------------------------------------
path     = '../../' #atlas version
pathData = path+'data/processed/'

indexDictList = json.load(open(pathData+'indexDictList_syr.txt')) 

trainIndex    = indexDictList[splitIndex]['Train']
trainGroups   = indexDictList[splitIndex]['TrainGroups']
testGroups    = indexDictList[splitIndex]['TestGroups'] 
testIndex     = indexDictList[splitIndex]['Test']


# In[ ]:





# In[7]:


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

# In[8]:


import tslearn
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
import tqdm
import matplotlib.pyplot as plt


# ## Additional Data Prep -- Representing Weather

# Use `tslearn` to cluster weather time series into categories

# In[9]:


w_all = data_list[2]
w_all.shape
# w_train == w_all


# In[10]:


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
    # TODO wrap this in a function.

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


# ### Data Setup

# In[11]:


# only needs to be run once
path = "./data_intermediates/weather_dupe_dict.p"
if os.path.exists(path):
    weather_dupe_dict = pkl.load(open(path, 'rb'))
else:
    weather_dupe_dict = mk_weather_dupe_dict(w_all)
    pkl.dump(weather_dupe_dict, open(path, 'wb'))
    
w_all_distinct = mk_weather_distinct(w_all, weather_dupe_dict)


# ### Demo

# In[ ]:





# ### Apply workflow to each variable.

# In[4]:


import multiprocess


# In[7]:


from multiprocess import Pool


import time


# In[10]:


from multiprocess import Pool
def f(x): 
    time.sleep(2)
    print('ding')

p = Pool(4)
result = p.map_async(f, range(10))
# print (result.get(timeout=1))


print("ran", ith_var, 'nd col')


# In[ ]:



max_k = 40 # There are 160 total unique weather observations. 
           # This assumes that at least 4 should make a cluster (on average)

# Will hold all the cluster results
w_all_clusters = np.zeros((w_all.shape[0], w_all.shape[2]))   

# removed to parallelize'



def parallelize_me(ith_var):

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
        print("ran", ith_var, 'col')

# Write results into full set object
# w_all_clusters[:, ith_var] = obs_cluster['cluster'] 


p = Pool(4)
result = p.map_async(f, range(10))

