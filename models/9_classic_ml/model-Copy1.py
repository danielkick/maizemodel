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





# In[12]:


# demo goes here
demo_tslearn = True

if demo_tslearn:
    ith_var = 5
    temp = w_all_distinct[:, :, ith_var]

    plt.imshow(temp)


# In[15]:


# TODO add logic to retrieve instead of retrun data artifacts if they exist.


# In[14]:


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


# In[ ]:


if demo_tslearn:
    best_k = find_best_k(
        overview = res['overview'])


# In[ ]:


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


# ### Apply workflow to each variable.

# In[ ]:





# In[ ]:


path_cluster = "./data_intermediates/w_all_clusters.p"
if os.path.exists(path):
    w_all_clusters = pkl.load(open(path, 'rb'))
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


# In[18]:


# Add to data list
data_list.extend([w_all_clusters])


# In[ ]:





# In[19]:


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


# In[20]:


nTestGroups = len(set(testGroups))
x = data_list
y = Y


# In[21]:


# Fold setup goes here ----
path = "./data_intermediates/cvFoldIdxDict.p"
path2 = "./data_intermediates/cvFoldIdxDict.p"
if (os.path.exists(path) & os.path.exists(path2)):
    cvFoldIdxDict = pkl.load(open(path, 'rb'))
    cvFoldCenterScaleDict  = pkl.load(open(path2, 'rb'))
else:
    cvGroup = list(set(trainGroups))
    rng = np.random.default_rng(cvSeed)
    valHoldoutDict = {}
    ## make the validation fold group membership for each fold. ====
    for i in range(kfolds):
        rng.shuffle(cvGroup)
        valHoldoutDict.update({str(i):cvGroup[0:nTestGroups]})
    ## This is what will be used for the test/train evaluation ====
    cvFoldIdxDict = {}
    cvFoldCenterScaleDict = {}
    for i in valHoldoutDict.keys():
        # i = '0'
        # these are positional indices not data indices.
        foldTrainIdx = [idx for idx in range(len(trainIndex)) if trainGroups[idx] not in valHoldoutDict[i]]
        foldTestIdx =  [idx for idx in range(len(trainIndex)) if trainGroups[idx]     in valHoldoutDict[i]]
        # convert to data indices.
        foldTrainDataIdx = [trainIndex[data_idx] for data_idx in foldTrainIdx]
        foldTestDataIdx = [trainIndex[data_idx] for data_idx in foldTestIdx]

        cvFoldIdxDict.update({i:{'Train':foldTrainDataIdx,'Test':foldTestDataIdx}})

        # Calculate values for center/scaling each fold
        ith_tr_idx = cvFoldIdxDict[i]['Train']

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
    
    pkl.dump(cvFoldCenterScaleDict, open(path, 'wb'))
    pkl.dump(cvFoldIdxDict, open(path, 'wb'))


# In[22]:


"""
Additional Testing:
Has the fold definition code messed up these indices? 
"""
for key in cvFoldIdxDict.keys():
    nan_Train = np.isnan(G[cvFoldIdxDict[key]['Train']]).sum()
    nan_Val   = np.isnan(G[cvFoldIdxDict[key]['Test']]).sum()
    print('key', key, ': NaN in Train: ', nan_Train, ', NaN in Test: ', nan_Val, sep = '')
    assert 0 == nan_Train == nan_Val, 'Nans found in dataset'


# In[ ]:





# ## Set up CV splits

# In[23]:


path = "./data_intermediates/foldwise_data_list.p"
if os.path.exists(path):
    foldwise_data_list = pkl.load(open(path, 'rb'))
else:
    # Training loop ----
    # i = str(int(use_cvFold_idx))
    foldwise_data_list = []
    for i in range(kfolds): 
        i = str(i)
        """
        This is now a list of tensors instead of one tensor (to prep for the big model)
        We'll do this a little different by iterating over the inputs. Then we can be agnostic as to 
        what inputs are going in -- just going off position.

        This will need to happen for the k fold evaluation below
        """
        x_train = [x_tensor[cvFoldIdxDict[i]['Train']] for x_tensor in x] 
        x_test  = [x_tensor[cvFoldIdxDict[i]['Test']]  for x_tensor in x] 
        y_train = y[cvFoldIdxDict[i]['Train']] 
        y_test  = y[cvFoldIdxDict[i]['Test']]

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
    
    
    pkl.dump(foldwise_data_list, open(path, 'wb'))


# ## Additional (temporary) prep -- Reducing dataset size

# In[24]:


i = 0 # which fold to be used?

x_train = foldwise_data_list[i]['train']['x']
y_train = foldwise_data_list[i]['train']['y']
x_test  = foldwise_data_list[i]['test' ]['x']
y_test  = foldwise_data_list[i]['test' ]['y']


# FIXME Reduce the train/test data so the below is fast. 


# In[25]:


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

res = _downsample_train_data(
    x_train,
    y_train,
    nTrainObs = 100,
    randSeed = 9834273475)

x_train = res['x']
y_train = res['y']


# In[ ]:





# ### Example Workflow -- KNN

# In[26]:


i = 0 # which fold to be used?


#TODO should W clusters be centered and scaled? Is sklearn good enough to interpret as catagories?


# In[27]:


# input should be a single array. concatenating any arrays should happen here.
# for ith_fold in range(len(foldwise_data_list)):

# Data set up -------------------------------------------------------------------------
ith_fold = 0

x_train = foldwise_data_list[ith_fold]['train']['x']
y_train = foldwise_data_list[ith_fold]['train']['y']
x_test  = foldwise_data_list[ith_fold]['test' ]['x']
y_test  = foldwise_data_list[ith_fold]['test' ]['y']

# FIXME -- this should come out to use the full data set. 
res = _downsample_train_data(
    x_train,
    y_train,
    nTrainObs = 100,
    randSeed = 9834273475)

x_train = res['x']
y_train = res['y']
    
# TODO combine all x data into one matrix
x_train = x_train[0] # FIXME set up a way to select the data to be used (G, S, Wc, All)
    
# HP tuning -------------------------------------------------------------------------    
HPs = {n_neighbors = [1,2,3,4]}




# In[ ]:





# #### Nearest Neighbor

# In[ ]:


# x_train_G = x_train[0]
# x_test_G = x_test[0]



# # K Nearest Neighbor
# knn_r = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, 
#                                       weights='uniform', 
#                                       algorithm='auto', 
#                                       leaf_size=30,
#                                       p=2, 
#                                       metric='minkowski', 
#                                       metric_params=None, 
#                                       n_jobs=None)

# knn_r = knn_r.fit(x_train_G, y_train)


# # MAE
# # in sample
# yHat_train = knn_r.predict(x_train_G)
# knn_r_train_mae = np.mean(abs(yHat_train - y_train))

# # out of sample
# yHat_test = knn_r.predict(x_test_G)
# knn_r_test_mae = np.mean(abs(yHat_test - y_test))

# print(knn_r_train_mae, knn_r_test_mae)


# In[ ]:


import skopt # scikit-optimize


# In[ ]:


## optimizing with skopt



i = 0 # which fold to be used?
i_data_list = 0

x_train = foldwise_data_list[i]['train']['x']
y_train = foldwise_data_list[i]['train']['y']
x_test  = foldwise_data_list[i]['test' ]['x']
y_test  = foldwise_data_list[i]['test' ]['y']

res = _downsample_train_data(
    x_train,
    y_train,
    nTrainObs = 100,
    randSeed = 9834273475)



x_train = res['x'][i_data_list]
y_train = res['y']
x_test = x_test[i_data_list]
y_test = y_test


# In[ ]:


def f(x):
    # K Nearest Neighbor
    knn_r = sklearn.neighbors.KNeighborsRegressor(n_neighbors= x , 
                                          weights='uniform', 
                                          algorithm='auto', 
                                          leaf_size=30,
                                          p=2, 
                                          metric='minkowski', 
                                          metric_params=None, 
                                          n_jobs=None)

    knn_r = knn_r.fit(x_train, y_train)


    # MAE
    # in sample
    yHat_train = knn_r.predict(x_train)
    knn_r_train_mae = np.mean(abs(yHat_train - y_train))

    # out of sample
    yHat_test = knn_r.predict(x_test)
    knn_r_test_mae = np.mean(abs(yHat_test - y_test))
    return(knn_r_test_mae)


# In[ ]:


print(f(2), f(3))


# In[ ]:


opt = skopt.BayesSearchCV(
    f(x),
    {'x':[i+1 for i in range(20)]},
    n_iter = 10,
    cv = 0
)
# pip install scikit-optimize==0.8.1


opt.fit(x_train, x_test)


# In[62]:


skopt. gp_minimize(f,                  # the function to minimize
            [(0, 20)],      # the bounds on each dimension of x
            acq_func="EI",      # the acquisition function
            n_calls=15,         # the number of evaluations of f
            n_random_starts=5,  # the number of random initialization points
            #noise=0.1**2,       # the noise level (optional)
            random_state=1234)


# In[60]:


opt_res = skopt.gp_minimize(f,  [(1, 20)], n_calls=10)


# In[ ]:





# In[66]:


from skopt import gp_minimize

res = gp_minimize(f,                  # the function to minimize
                  [(-2.0, 2.0)],      # the bounds on each dimension of x
                  acq_func="EI",      # the acquisition function
                  n_calls=15,         # the number of evaluations of f
                  n_random_starts=5,  # the number of random initialization points
                  noise=0.1**2,       # the noise level (optional)
                  random_state=1234)  # the random seed


# In[67]:


"x^*=%.4f, f(x^*)=%.4f" % (res.x[0], res.fun)


# In[53]:


# https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html#sphx-glr-auto-examples-bayesian-optimization-py


# In[68]:


from skopt.plots import plot_convergence
plot_convergence(res);


# In[70]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # to adapt

# ### Simple Models:

# In[ ]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

import seaborn as sns


# #### Naive Mean

# In[ ]:


NaiveMAE = np.mean(abs(YNorm[testIndex] - np.mean(YNorm[trainIndex])))

NaiveMAE


# In[ ]:





# #### Genotype Numeric

# In[ ]:


print('The names of the numeric genotype are not safe for formula use (contain "+")')
print('We\'ll substitue a number for these.')
GNumNames
GNumNameStandIn = ['x'+str(i) for i in range(len(GNumNames))]


# In[ ]:


## ====
temp = pd.DataFrame(
    GNormNum[trainIndex], 
    columns = GNumNameStandIn)
temp['Y'] = YNorm[trainIndex]


## ====
test = pd.DataFrame(
    GNormNum[testIndex], 
    columns = GNumNameStandIn)
test['Y'] = YNorm[testIndex]

temp.head()


# In[ ]:





# In[ ]:


mod = smf.ols(formula= ''.join(['Y ~ ', ' + '.join(GNumNameStandIn)]), data=temp)
res = mod.fit()
print(res.summary())

test['Yhat'] = res.predict(test)


# In[ ]:


sns.set_theme(style="white", color_codes=True)

g = sns.jointplot(data=test, x="Yhat", y="Y",
                  kind="reg", truncate=False,
                  color="m", height=7)


# In[ ]:


# Calculate MAE
test['Error'] = test['Y'] - test['Yhat']

GNumMAE = np.mean(abs(test['Error']))


# #### Genotype PCA

# In[ ]:





# In[ ]:


## ====
temp = pd.DataFrame(
    GNormPCA[trainIndex], 
    columns = GPCANames)
temp['Y'] = YNorm[trainIndex]


## ====
test = pd.DataFrame(
    GNormPCA[testIndex], 
    columns = GPCANames)
test['Y'] = YNorm[testIndex]

temp.head()


# In[ ]:





# In[ ]:


mod = smf.ols(formula= ''.join(['Y ~ ', ' + '.join(GPCANames)]), data=temp)
res = mod.fit()
# print(res.summary())

test['Yhat'] = res.predict(test)


# In[ ]:


sns.set_theme(style="white", color_codes=True)

g = sns.jointplot(data=test, x="Yhat", y="Y",
                  kind="reg", truncate=False,
                  color="m", height=7)


# In[ ]:


# Calculate MAE
test['Error'] = test['Y'] - test['Yhat']

GPCAMAE = np.mean(abs(test['Error']))


# #### GAM ???

# In[ ]:


from statsmodels.gam.api import GLMGam, BSplines




# In[ ]:


# import data
from statsmodels.gam.tests.test_penalized import df_autos

# create spline basis for weight and hp
x_spline = df_autos[['weight', 'hp']]

bs = BSplines(x_spline, df=[12, 10], degree=[3, 3])

# penalization weight
alpha = np.array([21833888.8, 6460.38479])

gam_bs = GLMGam.from_formula('city_mpg ~ fuel + drive', data=df_autos,
                             smoother=bs, alpha=alpha)


res_bs = gam_bs.fit()

print(res_bs.summary())


# #### Soil

# In[ ]:


## ====
temp = pd.DataFrame(
    SNorm[trainIndex], 
    columns = soilNames)
temp['Y'] = YNorm[trainIndex]


## ====
test = pd.DataFrame(
    SNorm[testIndex], 
    columns = soilNames)
test['Y'] = YNorm[testIndex]

temp.head()


# In[ ]:





# In[ ]:


mod = smf.ols(formula= ''.join(['Y ~ ', ' + '.join(soilNames)]), data=temp)
res = mod.fit()
print(res.summary())

test['Yhat'] = res.predict(test)


# In[ ]:


sns.set_theme(style="white", color_codes=True)

g = sns.jointplot(data=test, x="Yhat", y="Y",
                  kind="reg", truncate=False,
                  color="m", height=7)


# In[ ]:


# Calculate MAE
test['Error'] = test['Y'] - test['Yhat']

SoilMAE = np.mean(abs(test['Error']))


# #### Weather Group

# In[ ]:


## ====
temp = pd.DataFrame(zip(
    YNorm[trainIndex], 
    WCluster[trainIndex]), columns = ['Y', 'WCluster'])

## ====
test = pd.DataFrame(zip(
    YNorm[testIndex], 
    WCluster[testIndex]), columns = ['Y', 'WCluster'])



temp.head()


# In[ ]:


mod = smf.ols(formula='Y ~ WCluster', data=temp)
res = mod.fit()
print(res.summary())

test['Yhat'] = res.predict(test)


# In[ ]:


sns.set_theme(style="white", color_codes=True)

g = sns.jointplot(data=test, x="Yhat", y="Y",
                  kind="reg", truncate=False,
                  color="m", height=7)


# In[ ]:


# Calculate MAE
test['Error'] = test['Y'] - test['Yhat']

WClusterMAE = np.mean(abs(test['Error']))


# In[ ]:





# #### Kitchen Sink

# In[ ]:


temp = pd.DataFrame(
    GNormPCA[trainIndex], 
    columns = GPCANames).reset_index()

temp = temp.merge(
    pd.DataFrame(
    SNorm[trainIndex], 
    columns = soilNames).reset_index())

temp = temp.merge(
    pd.DataFrame(
    WCluster[trainIndex], 
    columns = ['WCluster']).reset_index())

temp['Y'] = YNorm[trainIndex]

temp = temp.drop(columns = 'index')

temp.shape


# In[ ]:





# In[ ]:





# In[ ]:


test = pd.DataFrame(
    GNormPCA[testIndex], 
    columns = GPCANames).reset_index()

test = test.merge(
    pd.DataFrame(
    SNorm[testIndex], 
    columns = soilNames).reset_index())

test = test.merge(
    pd.DataFrame(
    WCluster[testIndex], 
    columns = ['WCluster']).reset_index())

test['Y'] = YNorm[testIndex]

test = test.drop(columns = 'index')

test.shape


# In[ ]:





# In[ ]:


xs = [x for x in list(temp) if x not in ['Y']]

mod = smf.ols(formula= ''.join(['Y ~ ', ' + '.join(xs)]), data=temp)
res = mod.fit()
print(res.summary())

test['Yhat'] = res.predict(test)


# In[ ]:


sns.set_theme(style="white", color_codes=True)

g = sns.jointplot(data=test, x="Yhat", y="Y",
                  kind="reg", truncate=False,
                  color="m", height=7)


# In[ ]:


# Calculate MAE
test['Error'] = test['Y'] - test['Yhat']

KitchenSinkMAE = np.mean(abs(test['Error']))

test = test.drop(columns = ['Yhat', 'Error'])


# ### ML 

# In[ ]:


import sklearn


# #### Nearest Neighbor

# In[ ]:


# K Nearest Neighbor


# In[ ]:





# In[ ]:


knbrs= sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, 
                                      weights='uniform', 
                                      algorithm='auto', 
                                      leaf_size=30,
                                      p=2, 
                                      metric='minkowski', 
                                      metric_params=None, 
                                      n_jobs=None)

knbrs = knbrs.fit(temp.drop(columns = 'Y'), temp['Y'])


# In[ ]:


yHat = knbrs.predict(test.drop(columns = 'Y'))


# In[ ]:


knnbrMAE = np.mean(abs(yHat - test['Y']))


# In[ ]:





# In[ ]:


# Radius Neighbor


# In[ ]:


# RadiusNeighborsRegressor


# In[ ]:


r_nbrs = sklearn.neighbors.RadiusNeighborsRegressor(radius=18.359375, # 18.75-  -15.625
                                                    weights='uniform', 
                                                    algorithm='auto', 
                                                    leaf_size=30, 
                                                    p=2, 
                                                    metric='minkowski', 
                                                    metric_params=None, 
                                                    n_jobs=None)


r_nbrs = r_nbrs.fit(temp.drop(columns = 'Y'), temp['Y'])


# In[ ]:


yHat = r_nbrs.predict(test.drop(columns = 'Y'))


# In[ ]:


rnnbrMAE = np.mean(abs(yHat - test['Y']))


# #### RF

# In[ ]:


from sklearn import ensemble


# In[ ]:



regr = ensemble.RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(temp.drop(columns = 'Y'), temp['Y'])


# In[ ]:


yHat = regr.predict(test.drop(columns = 'Y'))


# In[ ]:


rfMAE = np.mean(abs(yHat - test['Y']))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:






# saving out models

# import pickle
# pkl_filename = 'pickle_model.pkl'
# # save
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model, file)
    
# #load
# with open(pkl_filename, 'rb') as file:
#     pickle_model = pickle.load(file)


# from sklearn.externals import joblib

# joblib_file = 'joblib_model.pkl'
# #save
# joblib.dump(model, joblib_file)
# #load
# joblib_model = joblib.load(joblib_file)




# In[ ]:





# #### SVM

# In[ ]:


from sklearn import svm


# In[ ]:


# # regr = svm.SVR()
# # Kernel Options are
# #  * 'linear' ----- 
# #  * 'polynomial' - uses degree, coef0
# #  * 'rbf' -------- uses gamma (must >0)
# #  * 'sigmoid' ---- uses coef
# #  see:
# #  https://scikit-learn.org/stable/modules/svm.html#kernel-functions

# regr = svm.LinearSVR() # faster but only linear kernel

# regr.fit(temp.drop(columns = 'Y'), temp['Y'])


# In[ ]:





# In[ ]:





# In[ ]:


# linear models

# Mixed Linear Model

# glm

# y ~ g1 ... gj + s1 ... sk + w


# gam


# In[ ]:





# In[ ]:





# ### comparison

# In[ ]:


# let's evaluate models in the same way we'll evaluate the DNN models.


# In[ ]:


compareMAE = pd.DataFrame(zip(
    ["Naive",
    "GNum",
    "GPCA",
     "Soil",
     "WCluster",
    "KitchenSink",
    
    "knnbr",
    "rnnbr",
    "rf"
    ],
    [NaiveMAE,
     GNumMAE,
     GPCAMAE,
     SoilMAE,
     WClusterMAE,
    KitchenSinkMAE,
    
    knnbrMAE,
    rnnbrMAE,
    rfMAE
    ]
    
), columns = ['Model', 'TestMAE'])

plt.xticks(rotation=45)
plt.ylabel('MAE')



plt.plot('Model', 'TestMAE', data = compareMAE)
plt.scatter('Model', 'TestMAE', data = compareMAE)
for i in range(compareMAE.shape[0]):
    plt.vlines(x=compareMAE.loc[i, 'Model'], 
               ymin = 0.76, 
               ymax=compareMAE.loc[i, 'TestMAE'])


plt.hlines(y = 0.84, xmin = 'Naive', xmax = 'rf', color = 'red', alpha = 0.6)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Classic ML

# In[26]:


import sklearn
from sklearn import ensemble # for rf


# In[ ]:





# In[ ]:





# ### Random Forest

# In[27]:


i = 0

x_train = foldwise_data_list[i]['train']['x']
y_train = foldwise_data_list[i]['train']['y']


# In[28]:



#FIXME
# for speed we'll use a slice of the data
x_train[0] = x_train[0][0:1000]
y_train = y_train[0:1000]


# In[29]:


regr = ensemble.RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(x_train[0], y_train)


# In[30]:


# within sample
yHat_train = regr.predict(x_train[0])
np.mean(y_train - yHat_train)


# In[31]:


# out of sample
yHat_test = regr.predict(x_test[0])
np.mean(y_test - yHat_test)


# In[ ]:





# In[32]:


# TODO
"""
[] Collapse W into something more useful.

For each method
[] Implement toy version
[] CV search with
    [] HPS search
    [] HPS save
[] Get consensus HPS
[] Final model train
[] Final model save
[] Final model eval

[] Keep adding methods until we're happy with the number of them
"""

# HP tuning:
from sklearn.model_selection import RandomizedSearchCV
# from scipy.stats import uniform
distributions = dict(max_depth=[1,2,3])

clf = RandomizedSearchCV(ensemble.RandomForestRegressor(random_state=0), 
                         distributions, 
                         random_state=0)
search = clf.fit(x_train[0], y_train)
search.best_params_


# In[ ]:





# In[ ]:





# In[ ]:





# ### XGBoost

# In[ ]:





# In[ ]:





# ### Random Forest

# In[ ]:





# In[ ]:





# ### Random Forest

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Hypermodel Definition:

# In[33]:


# # instead of building the model first then adding kt, let's work out from kt
# def build_model(hp):
#     ## Global hyperparameters ==================================================
#     adam_learning_rate = hp.Choice('learning_rate', [0.1, 0.01, 0.001, 0.0001])
#     adam_learning_beta1 = hp.Float('beta1', 0.9, 0.9999)
#     adam_learning_beta2 = hp.Float('beta2', 0.9, 0.9999) 

#     ## Subnetwork hyperparameters ==============================================
#     g_unit_min = 4
#     g_unit_max = 256 #1725
#     g_dropout_max = 0.3

#     s_unit_min = 4
#     s_unit_max = 64
#     s_dropout_max = 0.3

#     w_filter_min = 4
#     w_filter_max = 512

#     cat_unit_min = 4
#     cat_unit_max = 256 #1024
#     cat_dropout_max = 0.3
    
#     ## Inputs ==================================================================
#     gIn = tf.keras.Input(shape=(1725),       dtype='float32', name = 'gIn')
#     sIn = tf.keras.Input(shape=(21),         dtype='float32', name = 'sIn')
#     # wIn = tf.keras.Input(shape=(288, 19, 1), dtype = 'float32', name = 'wIn')
#     wIn = tf.keras.Input(shape=(288, 19), dtype='float32', name = 'wIn')
#     # Subnetworks --------------------------------------------------------------
#     ## G PCA ===================================================================
#     g_num_layers = hp.Int("g_num_layers", 1, 7)    
#     #gIn = tf.keras.Input(shape=(1725), dtype='float32', name = 'gIn') 
#     gX = gIn
#     for i in range(g_num_layers):
#         gX = layers.Dense(
#             units= hp.Int("g_fc_units_"+str(i), g_unit_min, g_unit_max),               
#             activation = 'relu'
#         )(gX)
#         gX = layers.BatchNormalization()(gX)
#         gX = layers.Dropout(
#                 rate = hp.Float("g_fc_dropout_"+str(i), 0.0, g_dropout_max)   
#         )(gX)   
#     gOut = gX
        
#     ## S =======================================================================
#     s_num_layers = hp.Int("s_num_layers", 1, 7)
#     #sIn = tf.keras.Input(shape=(21), dtype='float32', name = 'sIn')
#     sX = sIn    
#     for i in range(s_num_layers):
#         sX = layers.Dense(
#             units= hp.Int("s_fc_units_"+str(i), s_unit_min, s_unit_max),       
#             activation = 'relu'
#         )(sX)
#         sX = layers.BatchNormalization()(sX)
#         sX = layers.Dropout(
#             rate = hp.Float("s_fc_dropout_"+str(i), 0.0, s_dropout_max) 
#         )(sX)
#     sOut = sX
        
#     ## W c1d ===================================================================
#     pool_type_1d = hp.Choice("pool_type_1d", ['max1d', 'ave1d']) 
#     #wIn = tf.keras.Input(shape=(288, 19), dtype='float32', name = 'wIn')
#     wX = wIn
#     for i in range(hp.Int("conv1d_num_layer_blocks", 1, 7)): 
#         for j in range(hp.Int("conv1d_num_layers",   1, 4)):   
#             wX = layers.Conv1D(
#                 filters = hp.Int("c1_filters_"+str(i), w_filter_min, w_filter_max),  
#                 kernel_size = 3,
#                 strides  = 1,
#                 padding='valid',
#                 activation='relu',
#                 use_bias = True,
#                 kernel_initializer='random_uniform',
#                 kernel_regularizer=tf.keras.regularizers.l2(),
#                 bias_initializer='zeros')(wX)
#         wX = layers.BatchNormalization()(wX)
#         if pool_type_1d == 'max1d':
#             wX = layers.MaxPooling1D(
#                 pool_size = 2,
#                 strides =   1,
#                 padding ='same')(wX)
#         if pool_type_1d == 'ave1d':
#             wX = layers.AveragePooling1D(
#                 pool_size = 2,
#                 strides =   1,
#                 padding ='same')(wX)
#     wX = layers.Flatten()(wX)
#     wOut = wX
    
#     ## Interaction layers ======================================================
#     x_num_layers = hp.Int("x_num_layers", 1, 7)
#     cX = keras.layers.concatenate([gOut, sOut, wOut ])   
    
#     for i in range(x_num_layers):
#         cX = layers.Dense(
#             units= hp.Int("x_fc_units_"+str(i), cat_unit_min, cat_unit_max),
#             activation = 'relu'
#         )(cX)
#         cX = layers.BatchNormalization()(cX)
#         cX = layers.Dropout(
#                 rate = hp.Float("x_fc_dropout_"+str(i), 0.0, cat_dropout_max)   
#             )(cX)        
#     x = cX        
#     yHat = layers.Dense(1, activation='linear', name='yHat')(x) 
    
#     ## Setup ====
#     model = keras.Model(inputs=[gIn, sIn, wIn], outputs=[yHat])

#     opt = tf.keras.optimizers.Adam(
#         learning_rate = adam_learning_rate,
#         beta_1 = adam_learning_beta1, 
#         beta_2 = adam_learning_beta2  
#     )

#     model.compile(
#         optimizer= opt,
#         loss='mse',
#         metrics = ['mae']
#     )
#     return model  

# model = build_model(hp)


# In[34]:


# class CVTuner(kt.engine.tuner.Tuner):
#     def run_trial(self, trial, 
#                   foldwise_data_list,
#                   epochs,
#                   batch_size,
#                   *fit_args, **fit_kwargs
#                  ):
        
#         model = self.hypermodel.build(trial.hyperparameters)
#         fit_kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=16)  
        
#         val_losses = []
#         add_metrics_dict = {}
#         # for i in range(len(foldwise_data_list)): # previously we moved over all the folds. This 10x the training time. 
#         i = k_cv_rng.choice([i for i in range(len(foldwise_data_list))], 1)[0] # randomly select a fold to train on    
#         print(i, 'fold')
#         x_train = foldwise_data_list[i]['train']['x']
#         y_train = foldwise_data_list[i]['train']['y']
#         x_test  = foldwise_data_list[i]['test' ]['x']
#         y_test  = foldwise_data_list[i]['test' ]['y']
#         model = self.hypermodel.build(trial.hyperparameters)
#         model.fit(x_train, y_train, 
#                   batch_size=batch_size, 
#                   epochs=epochs
#                  )
#         model_fold_loss = model.evaluate(x_test, y_test)
#         print("mfl", model_fold_loss)
#         val_losses.append(model_fold_loss[0]) # loss is the first metic, 
#                                               # mae is the second. See the model definition.
#         add_metrics_dict.update({"k"+str(i)+"_loss": model_fold_loss[0],
#                                  "k"+str(i)+"_mae" : model_fold_loss[1]})
        
#         add_metrics_dict.update({'val_loss': np.nanmean(val_losses)}) #using nanmean here just in case something goes off the rails 
#         self.oracle.update_trial(trial.trial_id, add_metrics_dict)
#         self.save_model(trial.trial_id, model)
# # ## Train


# In[35]:


# tuner = CVTuner(
#     hypermodel=build_model,
#     oracle=kt.oracles.BayesianOptimization(
#         objective='val_loss',
#         max_trials=maxTrials
#     ),
#     # directory="C:/Users/drk8b9/rm_me", #for windows
#     directory=".", 
#     project_name = projectName,
#     overwrite=True
# )

# tuner.search(
#     foldwise_data_list,
#     batch_size = 20, # <-  this should be an int, but it'll get overwritten during the training.
#     epochs = numEpochs,
#     callbacks = [
#         keras.callbacks.EarlyStopping(monitor="val_loss", patience= 10),
#         keras.callbacks.CSVLogger("/training_log.csv", append=True)
#     ]
# )


# In[ ]:




