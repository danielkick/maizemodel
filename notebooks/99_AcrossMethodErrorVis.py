# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
import os, re, json, shutil 

#os.system('pip install tqdm') # TODO add to dockerfile

# jupyter labextension install jupyterlab-plotly # run in shell
# ^ this doesn't look to be working. Here's the work around: save each plot to one file name
# fig.write_html('00_plotly_current.html')
# then in a split window one need only refresh to get the current plot. 
# jk changing the renderer fixes this -v  # https://stackoverflow.com/questions/54064245/plot-ly-offline-mode-in-jupyter-lab-not-displaying-plots
import plotly.io as pio
pio.renderers.default = 'iframe' # or 'notebook' or 'colab' or 'jupyterlab'

import numpy as np
import pandas as pd
import math

import scipy.stats as stats        
from scipy import stats # for qq plot
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf_keras_vis
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

import datetime # For timing where %time doesn't work
import tqdm
import pickle as pkl
# useful pattern for speeding up runs ----v
# path = "./data_intermediates/cvFoldCenterScaleDict.p"
# if os.path.exists(path):
#     cvFoldCenterScaleDict = pkl.load(open(path, 'rb'))
# else:
#     pkl.dump(cvFoldCenterScaleDict, open(path, 'wb'))

# %%
pio.renderers.default = [
    'iframe',
#     'notebook',
#     'colab',
#     'jupyterlab'
][0]



# %%
"../data/atlas/models/3_finalize_model_syr__rep_G/"
"../data/atlas/models/3_finalize_model_syr__rep_S/"
"../data/atlas/models/3_finalize_model_syr__rep_W/"
"../data/atlas/models/3_finalize_model_syr__rep_full/"

# %% [markdown]
# ## Set up data -- used in salinence, pulling observations

# %%
phenotype = pd.read_csv('../data/processed/tensor_ref_phenotype.csv')
soil = pd.read_csv('../data/processed/tensor_ref_soil.csv')
weather = pd.read_csv('../data/processed/tensor_ref_weather.csv')

# %%
# %%
# Specified Options -----------------------------------------------------------
## General ====
projectName = 'kt' # this is the name of the folder which will be made to hold the tuner's progress.
# hp = kt.HyperParameters()
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


# %%
# Index prep -------------------------------------------------------------------
path     = '../' 
pathData = path+'data/processed/'

indexDictList = json.load(open(pathData+'indexDictList_syr.txt')) 

trainIndex    = indexDictList[splitIndex]['Train']
trainGroups   = indexDictList[splitIndex]['TrainGroups']
testGroups    = indexDictList[splitIndex]['TestGroups'] 
testIndex     = indexDictList[splitIndex]['Test']
# %%

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

# %%
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

# %%
x_all = x

# %%
# Set up data: ----------------------------------------------------------------
i = list(cvFoldCenterScaleDict.keys())[0]

# Center and scale testing and training data: ---------------------------------
## Training data: =============================================================
x_all[0] = ((x_all[0] - cvFoldCenterScaleDict[i]['GMean']) / cvFoldCenterScaleDict[i]['GStd'])
x_all[0] =   x_all[0].astype('float32') 

x_all[1] = ((x_all[1] - cvFoldCenterScaleDict[i]['SMean']) / cvFoldCenterScaleDict[i]['SStd'])
x_all[1] =   x_all[1].astype('float32')

x_all[2] = ((x_all[2] - cvFoldCenterScaleDict[i]['WMean']) / cvFoldCenterScaleDict[i]['WStd'])
x_all[2] =   x_all[2].astype('float32')

# %%
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
# %%
x_train = foldwise_data_list[0]['train']['x']
y_train = foldwise_data_list[0]['train']['y']
x_test  = foldwise_data_list[0]['test' ]['x']
y_test  = foldwise_data_list[0]['test' ]['y']

# ### Load Models, Rebuild if needed.

# %% [markdown]
# # What are some statistics about the data and train/test sets

# %% [markdown]
#
# - For All, test/train
# - Number of 
#     - Locations
#     - site-groups
#     - Genotypes
#     - Hybrids
#     - Inbreds
#     - Balance or lack there of re location x envs   

# %% [markdown]
# ## Full Dataset

# %%
df = phenotype

df['Split'] = 'None'
df.loc[trainIndex, 'Split'] = "Train"
df.loc[testIndex, 'Split'] = "Test"

print("Obs in Pheno:", df.shape[0])
print("Obs     used:", sum(df.Split != 'None'))
print("Obs    train:", len(trainIndex))
print("Obs     test:", len(testIndex))



# %%
# df.groupby(['Year', 'M']).size()

df['Hybrid'] = False
mask = (df.M != df.F)
df.loc[mask, 'Hybrid'] = True

ivs = ['ExperimentCode', 'Year', 'Hybrid', 'F', 'M']

# independent levels in df
pd.DataFrame(zip(
    ivs,
[len(list(set(df[iv]))) for iv in ivs]
    ), columns = ['Variable', 'Levels'])

# %%



print("Site x Year Combinations:", 
      df.groupby(['ExperimentCode', 'Year']).size().reset_index().shape[0] )

print("Genotypes", 
      df.groupby(['F', 'M']).size().reset_index().shape[0] )

print("Hybrids:  ", sum(df['Hybrid']))
print("Inbreds:  ", sum(~df['Hybrid']))
print("Hybrid %: ", np.mean(df['Hybrid']))




# %%
# replicates across experiments
temp = df.groupby(['F', 'M'#, 'ExperimentCode'#, 'Year'
           ]).size().reset_index().rename(columns = {0:'count'})

# how many replicates of each hybrid are there?
temp.agg(
    min = ('count', np.min),
    med = ('count', np.median),
    mean = ('count', np.mean),
    max = ('count', np.max),
        ).rename(columns = {'count':'Genotype Replicates'})

# %%
# top values
temp = temp.sort_values('count', ascending= False)
temp.head(10)




# %%
# replicates across site year
temp = df.groupby(['F', 'M', 'ExperimentCode', 'Year'
           ]).size().reset_index().rename(columns = {0:'count'})

# how many replicates of each hybrid are there?
temp.agg(
    min = ('count', np.min),
    med = ('count', np.median),
    mean = ('count', np.mean),
    max = ('count', np.max),
        ).rename(columns = {'count':'Genotype Replicates per Loc'})

# %%
# top values
temp = temp.sort_values('count', ascending= False)
temp.head(10)

# %%

# %%
# site x year obs
df_siteyear = df.groupby(['ExperimentCode', 'Year']).size().reset_index()
# df_siteyear.melt(id_vars = 'ExperimentCode', value_vars = 'Year')

mask = (df_siteyear.loc[:, 0] == df_siteyear.loc[:, 0].min())
print("Least Obs:", df_siteyear.loc[mask, ])

mask = (df_siteyear.loc[:, 0] == df_siteyear.loc[:, 0].max())
print("Most Obs:", df_siteyear.loc[mask, ])

print("Mean Obs:", np.mean(df_siteyear[0]) )

print("Med. Obs:", np.median(df_siteyear[0]) )

# df_siteyear = pd.pivot(df_siteyear, index = 'ExperimentCode', columns = 'Year', values = 0)#.reset_index()
# df_siteyear.to_csv("../output/z_site_x_year_count.csv")
# df_siteyear

# df.groupby(ivs).size()

# %%
# Summary counts w.r.t. train/test
df_siteyearsplit = df.loc[(df.Split != "None"), ].groupby(['ExperimentCode', 'Year', 'Split']).size().reset_index()

pd.pivot(df_siteyearsplit, 
         index = ['Split', 'ExperimentCode'], 
         columns = ['Year'], values = 0
        ).to_csv("../output/z_site_x_year_COND_split_count.csv")

# %% [markdown]
# ## Train/Test

# %%
df = phenotype

df['Split'] = 'None'
df.loc[trainIndex, 'Split'] = "Train"
df.loc[testIndex, 'Split'] = "Test"

df = df.loc[df.Split != "None"]

print("Obs in Pheno:", df.shape[0])
print("Obs     used:", sum(df.Split != 'None'))
print("Obs    train:", len(trainIndex))
print("Obs     test:", len(testIndex))

# %%
# df.groupby(['Year', 'M']).size()

df['Hybrid'] = False
mask = (df.M != df.F)
df.loc[mask, 'Hybrid'] = True

ivs = ['ExperimentCode', 'Year', 'Hybrid', 'F', 'M']

# independent levels in df
pd.DataFrame(zip(
    ivs,
[len(list(set(df[iv]))) for iv in ivs]
    ), columns = ['Variable', 'Levels'])

# %%

# %%
print("Site x Year Combinations:", 
      df.groupby(['ExperimentCode', 'Year']).size().reset_index().shape[0] )

print("Genotypes", 
      df.groupby(['F', 'M']).size().reset_index().shape[0] )

print("Hybrids:  ", sum(df['Hybrid']))
print("Inbreds:  ", sum(~df['Hybrid']))
print("Hybrid %: ", np.mean(df['Hybrid']))

# %%
# which inbred is represented?
df.loc[~df.Hybrid, ]

# %%
# replicates across experiments
temp = df.groupby(['F', 'M'#, 'ExperimentCode'#, 'Year'
           ]).size().reset_index().rename(columns = {0:'count'})

# how many replicates of each hybrid are there?
temp.agg(
    min = ('count', np.min),
    med = ('count', np.median),
    mean = ('count', np.mean),
    max = ('count', np.max),
        ).rename(columns = {'count':'Genotype Replicates'})

# %%
# top values
temp = temp.sort_values('count', ascending= False)
temp.head(10)

# %%
# replicates across site year
temp = df.groupby(['F', 'M', 'ExperimentCode', 'Year'
           ]).size().reset_index().rename(columns = {0:'count'})

# how many replicates of each hybrid are there?
temp.agg(
    min = ('count', np.min),
    med = ('count', np.median),
    mean = ('count', np.mean),
    max = ('count', np.max),
        ).rename(columns = {'count':'Genotype Replicates per Loc'})

# %%
# top values
temp = temp.sort_values('count', ascending= False)
temp.head(10)

# %%

# %%
# site x year obs
df_siteyear = df.groupby(['ExperimentCode', 'Year']).size().reset_index()
# df_siteyear.melt(id_vars = 'ExperimentCode', value_vars = 'Year')

mask = (df_siteyear.loc[:, 0] == df_siteyear.loc[:, 0].min())
print("Least Obs:", df_siteyear.loc[mask, ])

mask = (df_siteyear.loc[:, 0] == df_siteyear.loc[:, 0].max())
print("Most Obs:", df_siteyear.loc[mask, ])

print("Mean Obs:", np.mean(df_siteyear[0]) )

print("Med. Obs:", np.median(df_siteyear[0]) )

# df_siteyear = pd.pivot(df_siteyear, index = 'ExperimentCode', columns = 'Year', values = 0)#.reset_index()
# df_siteyear.to_csv("./z_site_x_year_count.csv")
# df_siteyear

# df.groupby(ivs).size()

# %%
# Summary counts w.r.t. train/test
df_siteyearsplit = df.loc[(df.Split != "None"), ].groupby(['ExperimentCode', 'Year', 
                                                           'Split']).size().reset_index()
pd.pivot(df_siteyearsplit, 
         index = ['ExperimentCode'], 
         columns = ['Split', 'Year'], values = 0
        ).to_csv("../output/z_site_x_year_COND_split_COND_used_count.csv")

# %%
# siteyears used w.r.t. train/test
df_siteyearsplit = df.loc[(df.Split != "None"), ].groupby(['ExperimentCode', 'Year', 
                                                           'Split']).size().reset_index()
df_siteyearsplit = pd.pivot(df_siteyearsplit, 
         index = ['ExperimentCode', 'Year'], 
         columns = ['Split'], values = 0
        ).reset_index()


# %%
# only in test set
mask = (df_siteyearsplit.Train.isna() & ~df_siteyearsplit.Test.isna())

print("obs in exclusive siteyear", df_siteyearsplit.loc[mask, 'Test'].sum())

df_siteyearsplit.loc[mask, :]

# %%
# only in test set
mask = (~df_siteyearsplit.Train.isna() & df_siteyearsplit.Test.isna())

print("obs in exclusive siteyear", df_siteyearsplit.loc[mask, 'Train'].sum())

df_siteyearsplit.loc[mask, :]

# %%

# %%
# _sites_ used w.r.t. train/test
df_siteyearsplit = df.loc[(df.Split != "None"), ].groupby(['ExperimentCode', #'Year', 
                                                           'Split']).size().reset_index()
df_siteyearsplit = pd.pivot(df_siteyearsplit, 
         index = ['ExperimentCode'], 
         columns = ['Split'], values = 0
        ).reset_index()


# %%
# only in test set
mask = (df_siteyearsplit.Train.isna() & ~df_siteyearsplit.Test.isna())

print("obs in exclusive site", df_siteyearsplit.loc[mask, 'Test'].sum())

df_siteyearsplit.loc[mask, :]

# %%
# only in test set
mask = (~df_siteyearsplit.Train.isna() & df_siteyearsplit.Test.isna())

print("obs in exclusive site", df_siteyearsplit.loc[mask, 'Train'].sum())

df_siteyearsplit.loc[mask, :].reset_index()

# %%
# in both
mask = (~df_siteyearsplit.Train.isna() & ~df_siteyearsplit.Test.isna())

print("obs in inclusive geno", df_siteyearsplit.loc[mask, 'Train'].sum(), df_siteyearsplit.loc[mask, 'Test'].sum())

df_siteyearsplit.loc[mask, :].reset_index()




# %%

# %%
# what does the overlap in genetics look like?
df

# %%
df_genobalance = df.loc[:, ['Split', 
                            'F', 
                            'M', 
           # 'ExperimentCode',
           'GrainYield']].groupby(["Split", 'F', 'M']).count().reset_index()


df_genobalance = pd.pivot(df_genobalance, 
         index = ['F', 'M' #, 'ExperimentCode'
                 ], 
         columns = ['Split'], 
         values = 'GrainYield'
        )

df_genobalance

# %%
# only in test set
mask = (df_genobalance.Train.isna() & ~df_genobalance.Test.isna())

print("obs in exclusive geno", df_genobalance.loc[mask, 'Test'].sum())

df_genobalance.loc[mask, :]

# %%
# only in training set
mask = (~df_genobalance.Train.isna() & df_genobalance.Test.isna())

print("obs in exclusive geno", df_genobalance.loc[mask, 'Train'].sum())

df_genobalance.loc[mask, :]

# %%
# found in both
mask = (~df_genobalance.Train.isna() & ~df_genobalance.Test.isna())

print("obs in inclusive geno", df_genobalance.loc[mask, 'Train'].sum(), df_genobalance.loc[mask, 'Test'].sum())

df_genobalance.loc[mask, :]


# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# # Compare Goodness of Fit Across and Within Models

# %% [markdown]
# ## Preparation

# %% code_folding=[2]
# classes to hold model results

class ml_results:
    """
    # .
    # |- All
    # |  |- knn
    # |  |- rnr
    # |  |- svrl
    # |  |- rf
    # |
    # |- G
    # |- S
    # |- WOneHot
    """
    def __init__(self, project_folder):
        self.project_folder = project_folder
        self.results = {}
        
    def _re_filter_list(self, in_list, regex ):
        return([e for e in in_list if re.findall(regex, e)])        
        
    def retrieve_results(self):
        dirs_in_proj = os.listdir(self.project_folder)
        res_dirs =  self._re_filter_list(in_list = dirs_in_proj, regex = 'hps_res') 
        
        for res_dir in res_dirs:
            files_in_res_dir = os.listdir(self.project_folder+res_dir)
            res_files = self._re_filter_list(
                            in_list = files_in_res_dir, 
                            regex = '^Best_hps\w*\.json$')            
            for res_file in res_files:
                with open(self.project_folder+'/'+res_dir+'/'+res_file, 'r') as f:
                    dat = json.load(f)        
        
                # Make keys to use
                results_key_lvl_0 = res_dir.replace('hps_res_', '')
                results_key_lvl_1 = res_file.replace('Best_hps_', '').split('_')[0]
                
                results_key_lvl_2 = res_file.replace('Best_hps_', '').replace('.json', '').split('_')[-1]                    #        
                # Check to see if lvl 0 key exists. Update that key if it does, otherwise add it with lvl 1 key within it.
                if results_key_lvl_0 in self.results.keys():
                    if results_key_lvl_1 in self.results[results_key_lvl_0].keys():
                        self.results[results_key_lvl_0][results_key_lvl_1].update({results_key_lvl_2:dat})                        
                    else:
                        self.results[results_key_lvl_0].update({results_key_lvl_1:{results_key_lvl_2:dat}})
#                     self.results[results_key_lvl_0].update({results_key_lvl_1:dat})

                else:
                    self.results.update({results_key_lvl_0:{results_key_lvl_1:{results_key_lvl_2:dat}}})
#                     self.results.update({results_key_lvl_0:{results_key_lvl_1:dat}})
                    

    
class tf_results(ml_results):
    """
    # .
    # |- full
    # |  |- '0'
    # |  |   |- 'use_tf_model_rep'  \\
    # |  |   |- 'y_train'           |
    # |  |   |- 'yHat_train'        > output from evalModelLean.py
    # |  |   |- 'y_test'            |
    # |  |   |- 'yHat_test'         /
    # |  |   |- 'loss'          \\
    # |  |   |- 'mae'           |
    # |  |   |- 'val_loss'      > From history
    # |  |   |- 'val_mae'       /
    # |  |
    # |  |- '1'
    # |  |  ...
    # |  |- '9'
    # |
    # |- G
    # |- S
    # |- W
    """
    # Replaced to fit with tf model format
    def retrieve_results(self):
        dirs_in_proj = os.listdir(self.project_folder+'/eval')            
        # We need to process these separately or merge one into the other.
        res_predictions = [e for e in dirs_in_proj if re.findall('predictions' , e)]
        res_histories = [e for e in dirs_in_proj if re.findall('history' , e)]

        results_key_lvl_0 = self.project_folder.split('/')[-2].split('_')[-1]

        ## Processing predictions
        for res_file in res_predictions:                
            results_key_lvl_1 = re.findall('rep_\d+', res_file)[0].replace('rep_', '')

            with open(self.project_folder+'/eval'+'/'+res_file, 'r') as f:
                dat = json.load(f)

            #Check to see if lvl 0 key exists. Update that key if it does, otherwise add it with lvl 1 key within it.
            # This is overkill for tf results since we only have one level of `results_key_lvl_0`
            if results_key_lvl_0 in self.results.keys():
                self.results[results_key_lvl_0].update({results_key_lvl_1:dat})
            else:
                self.results.update({results_key_lvl_0:{results_key_lvl_1:dat}})

        ## Processing histories
        for res_file in res_histories:                
            results_key_lvl_1 = re.findall('rep_\d+', res_file)[0].replace('rep_', '')

            with open(self.project_folder+'/eval'+'/'+res_file, 'r') as f:
                dat = json.load(f)

            add_dict = {
                        'loss':dat['loss'],
                        'mae':dat['mae'],
                        'val_loss':dat['val_loss'],
                        'val_mae':dat['val_mae']
                    }
            if results_key_lvl_0 in self.results.keys():
                if results_key_lvl_1 in self.results[results_key_lvl_0]:
                    self.results[results_key_lvl_0][results_key_lvl_1].update(add_dict)
            else:
                self.results.update({results_key_lvl_0:{results_key_lvl_1:add_dict}}) 
                
                
                         

# update lmlike to act more like bglr_results
class lmlike_results(ml_results):
    # Replaced to fit with the lm/lmer output from R
    def retrieve_results(self):
        dirs_in_proj = os.listdir(self.project_folder+'/results')  
        # find all the blups 
        dirs_in_proj = [e for e in dirs_in_proj if re.match("lm_.*", e)] 
        # limit to those with results
        res_names = [e for e in dirs_in_proj if 'fm_predictions.csv' in os.listdir(self.project_folder+'results/'+e)]

        for res_name in res_names:
            # These files associated with each result name.       
            # _formula.txt -----> Used for 'params'
            # _predictions.csv -> Used for 'y.*'
            # .rds -------------> Saved, but only used in R

            res_dict = {'use_lmlike_model': res_name}
            
            formula_path = self.project_folder+'/results/'+res_name+"/fm_formula.txt"
            predictions_path = self.project_folder+'/results/'+res_name+"/fm_predictions.csv"

            if os.path.exists(formula_path):
                with open(formula_path) as f:
                    dat = f.readlines()
                res_dict.update({'params': dat})
            if os.path.exists(predictions_path):
                dat = pd.read_csv(predictions_path)
                res_dict.update({
                       'y_train' : dat.loc[dat.type == "Train", 'y'   ].to_list(), 
                    'yHat_train' : dat.loc[dat.type == "Train", 'yHat'].to_list(), 
                        'y_test' : dat.loc[dat.type == "Test",  'y'   ].to_list(), 
                     'yHat_test' : dat.loc[dat.type == "Test",  'yHat'].to_list()
                })            
            
            # find out which group to use:
            has_G = has_S = has_W = has_multi = False
            if re.match(".*G.*", res_name):
                has_G = True
            if re.match(".*S.*", res_name):
                has_S = True
            if re.match(".*W.*", res_name):
                has_W = True
            if (has_G & has_S) | (has_G & has_W) | (has_S & has_W):
                has_multi = True
                
                
            # Because there are multiples, 
            if has_multi:
                results_key_lvl_0 = 'Multi'
            elif has_G:
                results_key_lvl_0 = 'G'
            elif has_S:
                results_key_lvl_0 = 'S'
            elif has_W:
                results_key_lvl_0 = 'W'
                
            results_key_lvl_1 = "lm"

            # Extract replicate number or impute (first may not have 0 in the name)
            if not re.findall("Rep|rep", res_name):
                results_key_lvl_2 = '0'
            else:
                results_key_lvl_2 = re.findall("\d+$", res_name)
            results_key_lvl_2 = 'rep'+results_key_lvl_2[0]
            
            if results_key_lvl_0 not in self.results.keys():
                self.results.update({results_key_lvl_0:{results_key_lvl_1:{results_key_lvl_2:res_dict}}})
            elif results_key_lvl_1 not in self.results[results_key_lvl_0].keys():
                self.results[results_key_lvl_0].update({results_key_lvl_1:{results_key_lvl_2:res_dict}})
            elif results_key_lvl_2 not in self.results[results_key_lvl_0][results_key_lvl_1].keys():
                self.results[results_key_lvl_0][results_key_lvl_1][results_key_lvl_2] = res_dict


# %%
# based on the lmlike_results() function we're going to package up the blups using bglr_results()
# `BGLR` is an r library from cimmyt that can run blups and bayesian alphabet models as a Bayesian Generalized Linear Regression.
class bglr_results(ml_results):
    # Replaced to fit with the lm/lmer output from R
    def retrieve_results(self):
        dirs_in_proj = os.listdir(self.project_folder)
        # find all the blups 
        dirs_in_proj = [e for e in dirs_in_proj if re.match("BLUP_.*", e)] 
        # limit to those with results
        res_names = [e for e in dirs_in_proj if 'BlupYHats.csv' in os.listdir(self.project_folder+'/'+e)]

        for res_name in res_names:
            # Associated files: 
            # 'BLUP.R' ----------- Defines the blup to run
            # 'runBLUP' ---------- sbatch file that runs the blup
            # 'slurm-{\d+}.out' -- log file
            #
            # 'BlupYHats.csv' ---- Contains columns `0` (row index), `Y`, `YTrain` (test data is NaN), `YHat`
            # 'BlupRunTime.csv' -- Contains `start`, `end`, `duration` of the BGLR() run (excluding prep). 
            #
            # 'fm.rds' ----------- Fitted model in case we need to use it.
            # '*.dat' ------------ BGLR generated files

            res_dict = {'use_blup_model': res_name}
            predictions_path = self.project_folder+'/'+res_name+"/BlupYHats.csv"

            # parse dir name into formula of form
            """G+S+W+GxS+GxW+SxW"""
            formula_chunks = res_name.replace('BLUP_', '').split('_')
            
            formula_chunks = [
                '+'.join([e1 for e1 in e]) if re.findall('x', e) != ['x'] # find additive effects and put a '+' between each char
                else e # all the chunks with an 'x' are interactions so leave them be
                for e in formula_chunks]
            formula = '+'.join(formula_chunks)
            res_dict.update({'params': formula})

            # don't have to check if the path exists because we limited the `res_names` to those with BlupYHats.csv above
            dat = pd.read_csv(predictions_path)
            dat['type'] = ''
            mask = dat.YTrain.isna()
            dat.loc[mask, 'type'] = 'Test'
            dat.loc[~mask, 'type'] = 'Train'
            
            # backup non center/scaled results
            dat['Y_no_cs'] = dat['Y']
            dat['YHat_no_cs'] = dat['YHat']
            
            # we used the training data to center/scale to avoid information leakage
            train_mean = np.mean(list(dat.loc[dat.type == "Train", 'Y']))
            train_sd = np.std(list(dat.loc[dat.type == "Train", 'Y']))            

            dat['Y'] = ((dat['Y'] - train_mean)/train_sd)
            dat['YHat'] = ((dat['YHat'] - train_mean)/train_sd)
            
            res_dict.update({
                   'y_train' : dat.loc[dat.type == "Train", 'Y'   ].to_list(), 
                'yHat_train' : dat.loc[dat.type == "Train", 'YHat'].to_list(), 
                    'y_test' : dat.loc[dat.type == "Test",  'Y'   ].to_list(), 
                 'yHat_test' : dat.loc[dat.type == "Test",  'YHat'].to_list(),
                # as long as we have it, let's save the un-center/scaled estimates too 
                   'y_train_no_cs' : dat.loc[dat.type == "Train", 'Y_no_cs'   ].to_list(), 
                'yHat_train_no_cs' : dat.loc[dat.type == "Train", 'YHat_no_cs'].to_list(), 
                    'y_test_no_cs' : dat.loc[dat.type == "Test",  'Y_no_cs'   ].to_list(), 
                 'yHat_test_no_cs' : dat.loc[dat.type == "Test",  'YHat_no_cs'].to_list()
            })

            # Note! We assume there is only one replicate of each model since it's a linear model.
            # We'll tweak our groupings too. ML has All and TF has full, 
            #  here we add multi to capture models with 2+ data sources.
            # match fo find out which group to use:
            has_G = has_S = has_W = has_multi = False
            if re.match(".*G.*", res_name):
                has_G = True
            if re.match(".*S.*", res_name):
                has_S = True
            if re.match(".*W.*", res_name):
                has_W = True
            if (has_G & has_S) | (has_G & has_W) | (has_S & has_W):
                has_multi = True
                
                
            # Because there are multiples, 
            if has_multi:
                results_key_lvl_0 = 'Multi'
            elif has_G:
                results_key_lvl_0 = 'G'
            elif has_S:
                results_key_lvl_0 = 'S'
            elif has_W:
                results_key_lvl_0 = 'W'
                
            results_key_lvl_1 = "BLUP"

            # Extract replicate number or impute (first may not have 0 in the name)
            if not re.findall("Rep|rep", res_name):
                results_key_lvl_2 = '0'
            else:
                results_key_lvl_2 = re.findall("\d+$", res_name)
            results_key_lvl_2 = 'rep'+results_key_lvl_2[0]
            
#             print(res_name, results_key_lvl_1) debugging


            
#             self.results.update({results_key_lvl_0:{results_key_lvl_1:{results_key_lvl_2:res_dict}}})
#             if has_G:

            if results_key_lvl_0 not in self.results.keys():
                self.results.update({results_key_lvl_0:{results_key_lvl_1:{results_key_lvl_2:res_dict}}})
            elif results_key_lvl_1 not in self.results[results_key_lvl_0].keys():
                self.results[results_key_lvl_0].update({results_key_lvl_1:{results_key_lvl_2:res_dict}})
            elif results_key_lvl_2 not in self.results[results_key_lvl_0][results_key_lvl_1].keys():
                self.results[results_key_lvl_0][results_key_lvl_1][results_key_lvl_2] = res_dict

#             print(results_key_lvl_0, results_key_lvl_1, results_key_lvl_2)
#             print(self.results["G"]["BLUP"].keys())

#             if results_key_lvl_0 in self.results.keys():
#                 if "BLUP" in self.results[results_key_lvl_0]:
#                     if results_key_lvl_2 in self.results[results_key_lvl_0]["BLUP"]:
#                         self.results[results_key_lvl_0]["BLUP"][results_key_lvl_2].update(res_dict)
#             else:
#                 self.results.update({results_key_lvl_0:{"BLUP":{results_key_lvl_2:res_dict}}})  




#             if has_multi:
#                 if "Multi" in self.results.keys():
#                     self.results["Multi"].update({res_name:{"rep0":res_dict}})
#                 else:
#                     self.results.update({"Multi":{res_name:{"rep0":res_dict}}})
#             elif has_G:
#                 if "G" in self.results.keys():
#                     self.results["G"].update({res_name:{"rep0":res_dict}})
#                 else:
#                     self.results.update({"G":{res_name:{"rep0":res_dict}}})
#             elif has_S:
#                 if "S" in self.results.keys():
#                     self.results["S"].update({res_name:{"rep0":res_dict}})
#                 else:
#                     self.results.update({"S":{res_name:{"rep0":res_dict}}})
#             elif has_W:
#                 if "W" in self.results.keys():
#                     self.results["W"].update({res_name:{"rep0":res_dict}})
#                 else:
#                     self.results.update({"W":{res_name:{"rep0":res_dict}}}) 


# %% [markdown]
# | . |rep |  train_rmse   | train_r2      | test_rmse    | test_r2       |
# |---|----|---------------|---------------|--------------|---------------|
# | G | 0  |  0            |  0            | 0            |  2.220446e-16 |
# | G | 1  |  0            |  1.110223e-16 | 0            |  0            |
# | G | 2  |  0            |  1.110223e-16 | 0            |  0            |
# | G | 4  |  0            |  1.110223e-16 | 0            |  0            |
# | G | 6  |  1.110223e-16 | -1.110223e-16 | 0            |  2.220446e-16 |
# | G | 9  | -1.110223e-16 |  1.110223e-16 | 0            |  0            |
# | S | 3  | -1.110223e-16 |  2.220446e-16 | 0            |  0            |
# | W | 0  |  1.110223e-16 | -1.110223e-16 | 0            |  0            |
# | W | 1  |  1.110223e-16 | -1.110223e-16 | 0            |  0            |
# | W | 2  |            0  |  1.110223e-16 | 0            |  2.220446e-16 |
# | W | 5  | -1.110223e-16 |  1.110223e-16 | 0            |  0            |
# | SO| 0  |  0            |  0            | 2.220446e-16 | -2.220446e-16 |
# | SO| 5  |  5.551115e-17 |  0            | 0            |  0            |
# | CO| 2  |  3.552714e-15 | -5.684342e-14 | 0            | -1.110223e-16 |
# | CO| 7  |  1.110223e-16 |  0            | 0            |  0            |
# | CO| 8  |  0            |  0            |-1.110223e-16 |  1.110223e-16 |
# | CO| 9  |  1.110223e-16 |  0            | 0            |  0            |

# %%

# %%
bglr_res = bglr_results(project_folder="../models/3_finalize_model_BGLR_BLUPS")
bglr_res.retrieve_results()
bglr_res = bglr_res.results


# %%
# bglr_res.keys()
# bglr_res['Multi']['BLUP'].keys()

# %%
# ml_res['All']['knn'].keys()

# %%
# def _temp(irep = 0):
#     with open( "../data/atlas/models/3_finalize_classic_ml_10x/hps_res_All/Best_hps_rf_4All_rep"+str(irep)+".json", 'r') as f:
#         dat = json.load(f)    

#     return(pd.DataFrame({str(irep): dat['y_train']}))

# rf_temp = pd.concat([_temp(irep = i) for i in range(10)], axis = 1)
# rf_temp

# %%

# %% code_folding=[]
# print(ml_results.__doc__)
# ml_res = ml_results(project_folder = "../data/atlas/models/3_finalize_classic_ml/")
ml_res = ml_results(project_folder = "../models/3_finalize_classic_ml_10x/")
ml_res.retrieve_results()
ml_res = ml_res.results
            
# ml_res.results['All']['rf'].keys()
# dict_keys(['rep6', 'rep2', 'rep8', 'rep7', 'rep1', 'rep3', 'rep0', '4All', 'rep9', 'rep5', 'rep4'])

# %%
# ml_res

# %% code_folding=[]
# print(tf_results.__doc__)
tf_res = {}

for proj in ["3_finalize_model_syr__rep_G", 
             "3_finalize_model_syr__rep_S",
             "3_finalize_model_syr__rep_W",
             "3_finalize_model_syr__rep_full",
             "3_finalize_model_syr__rep_cat"
            ]:
    temp = tf_results(project_folder = "../models/"+proj+"/")
    temp.retrieve_results()
    temp = temp.results

    if list(tf_res.keys()) == []:
        tf_res = temp
    else:
        tf_res.update(temp)


# %%
lmlike_res = lmlike_results(project_folder="../models/3_finalize_lm/")
lmlike_res.retrieve_results()
lmlike_res = lmlike_res.results

# %%

# %%
# tf_res

# %%
from sklearn.metrics import r2_score

# %%

# %% code_folding=[35, 45, 111]
# temp = ml_res['All']['knn']

## Create Empty Summary DataFrame ==============================================
# Make a df with all the keys so we can populate it with the statistics we want.
# make a df with all the keys for each experiment we ran. 
keys_to_get_ys = []
for key0 in ml_res.keys():
    for key1 in ml_res[key0].keys():
        for key2 in ml_res[key0][key1].keys():
            keys_to_get_ys.append(['ml_res', key0, key1, key2])
        
for key0 in tf_res.keys():
    for key1 in tf_res[key0].keys():
        keys_to_get_ys.append(['tf_res', key0, key1])

        
for key0 in lmlike_res.keys():
    for key1 in lmlike_res[key0].keys():
        for key2 in lmlike_res[key0][key1].keys():
            keys_to_get_ys.append(['lmlike_res', key0, key1, key2])

        
for key0 in bglr_res.keys():
    for key1 in bglr_res[key0].keys():
        for key2 in bglr_res[key0][key1].keys():
            keys_to_get_ys.append(['bglr_res', key0, key1, key2])
            
summary_df = pd.DataFrame(keys_to_get_ys, columns = ['source', 'key0', 'key1', 'key2'])


summary_df[['train_r','train_r2','train_mse','train_rmse', 
            'test_r', 'test_r2', 'test_mse', 'test_rmse']] = np.nan

## Populate Summary DataFrame with Error Statistics ============================

def replace_na_w_0(in_df = temp,
                   col_vars = ['y_train', 'yHat_train']):
    for col_var in col_vars:
        na_mask = in_df[col_var].isna()
        n_missings = sum(na_mask)
        if n_missings > 0:
            print('Replacing '+str(n_missings)+' with 0.')
            in_df.loc[na_mask, col_var] = 0
    return(in_df)

def calc_performances(observed, predicted, outtype = 'list'):
    r = stats.pearsonr(observed, predicted)[0] # index 1 is the p value
#     r2 = r**2
    r2 = r2_score(observed, predicted)
    MSE = mean_squared_error(observed, predicted)
    RMSE = math.sqrt(MSE)
    
    #additional metics 
    # relative RMSE
    relRMSE = RMSE / np.nanmean(observed)
    # normalized RMSE
    normRMSE = RMSE / (np.nanmax(observed) - np.nanmin(observed))
    
    
    if outtype == 'list':
        return([r, r2, MSE, RMSE,
               relRMSE, normRMSE
               ])
    else:
        return({'r':r,
               'r2':r2,
              'mse':MSE,
             'rmse':RMSE,
          'relRMSE':relRMSE,
         'normRMSE':normRMSE
               })

for i in summary_df.index:
    source = summary_df.loc[i, 'source']
    key0 = summary_df.loc[i, 'key0']
    key1 = summary_df.loc[i, 'key1']
    key2 = summary_df.loc[i, 'key2']
    
    
    if source == 'ml_res':
        temp = pd.DataFrame()
        temp['y_train'] = ml_res[key0][key1][key2]['y_train']
        temp['yHat_train'] = ml_res[key0][key1][key2]['yHat_train']
        temp = replace_na_w_0(in_df = temp, col_vars = ['y_train', 'yHat_train'])
        summary_df.loc[i, ['train_r','train_r2','train_mse','train_rmse',
                          'train_relRMSE', 'train_normRMSE']] = calc_performances(observed = temp['y_train'], predicted = temp['yHat_train'] )

        temp = pd.DataFrame()
        temp['y_test'] = ml_res[key0][key1][key2]['y_test']
        temp['yHat_test'] = ml_res[key0][key1][key2]['yHat_test']
        temp = replace_na_w_0(in_df = temp, col_vars = ['y_test', 'yHat_test'])
        summary_df.loc[i, ['test_r', 'test_r2', 'test_mse', 'test_rmse',
                          'test_relRMSE', 'test_normRMSE']] = calc_performances(observed = temp['y_test'], predicted = temp['yHat_test'])


    if source == 'tf_res':
        temp = pd.DataFrame()
        temp['y_train'] = tf_res[key0][key1]['y_train']
        temp['yHat_train'] = [e[0] for e in tf_res[key0][key1]['yHat_train']]
        temp = replace_na_w_0(in_df = temp, col_vars = ['y_train', 'yHat_train'])
        summary_df.loc[i, ['train_r','train_r2','train_mse','train_rmse',
                          'train_relRMSE', 'train_normRMSE']] = calc_performances(observed = temp['y_train'], predicted = temp['yHat_train'] )

        temp = pd.DataFrame()
        temp['y_test'] = tf_res[key0][key1]['y_test']
        temp['yHat_test'] = [e[0] for e in tf_res[key0][key1]['yHat_test']]
        temp = replace_na_w_0(in_df = temp, col_vars = ['y_test', 'yHat_test'])
        summary_df.loc[i, ['test_r', 'test_r2', 'test_mse', 'test_rmse',
                          'test_relRMSE', 'test_normRMSE']] = calc_performances(observed = temp['y_test'], predicted = temp['yHat_test'])
        
        
    if source == 'lmlike_res':
        temp = pd.DataFrame()
        temp['y_train'] = lmlike_res[key0][key1][key2]['y_train']
        temp['yHat_train'] = lmlike_res[key0][key1][key2]['yHat_train']
        temp = replace_na_w_0(in_df = temp, col_vars = ['y_train', 'yHat_train'])
        summary_df.loc[i, ['train_r','train_r2','train_mse','train_rmse',
                          'train_relRMSE', 'train_normRMSE']] = calc_performances(observed = temp['y_train'], predicted = temp['yHat_train'] )

        temp = pd.DataFrame()
        temp['y_test'] = lmlike_res[key0][key1][key2]['y_test']
        temp['yHat_test'] = lmlike_res[key0][key1][key2]['yHat_test']
        temp = replace_na_w_0(in_df = temp, col_vars = ['y_test', 'yHat_test'])
        summary_df.loc[i, ['test_r', 'test_r2', 'test_mse', 'test_rmse',
                          'test_relRMSE', 'test_normRMSE']] = calc_performances(observed = temp['y_test'], predicted = temp['yHat_test'])

        
    if source == 'bglr_res':
        #print(key0, key1, key2)
        
        
        temp = pd.DataFrame() 
        temp['y_train'] = bglr_res[key0][key1][key2]['y_train']
        temp['yHat_train'] = bglr_res[key0][key1][key2]['yHat_train']
        temp = replace_na_w_0(in_df = temp, col_vars = ['y_train', 'yHat_train'])
        summary_df.loc[i, ['train_r','train_r2','train_mse','train_rmse',
                          'train_relRMSE', 'train_normRMSE']] = calc_performances(observed = temp['y_train'], predicted = temp['yHat_train'] )

        temp = pd.DataFrame()
        temp['y_test'] = bglr_res[key0][key1][key2]['y_test']
        temp['yHat_test'] = bglr_res[key0][key1][key2]['yHat_test']
        temp = replace_na_w_0(in_df = temp, col_vars = ['y_test', 'yHat_test'])
        summary_df.loc[i, ['test_r', 'test_r2', 'test_mse', 'test_rmse',
                          'test_relRMSE', 'test_normRMSE']] = calc_performances(observed = temp['y_test'], predicted = temp['yHat_test'])


# %%
# bglr_res[key0][key1][key2]


# %%
summary_df


# %%
summary_df.loc[(summary_df.source == "lmlike_res"), ]

# %%
## Add in a Naieve estimate -- Guessing the Training Mean Every Time ===========
ys = tf_res['G']['0']['y_train']
yNaive = [0 for e in range(len(ys))]

train_MSE = mean_squared_error(ys, yNaive)
train_RMSE = math.sqrt(train_MSE)
train_r2 = r2_score(ys, yNaive)
train_relRMSE = train_RMSE / np.nanmean(ys)
train_normRMSE = train_RMSE / (np.nanmax(ys) - np.nanmin(ys))

ys = tf_res['G']['0']['y_test']
yNaive = [0 for e in range(len(ys))]

test_MSE = mean_squared_error(ys, yNaive)
test_RMSE = math.sqrt(test_MSE)
test_r2 = r2_score(ys, yNaive)
test_relRMSE = test_RMSE / np.nanmean(ys)
test_normRMSE = test_RMSE / (np.nanmax(ys) - np.nanmin(ys))


# pd.concat(summary_df , ['Mean', 'Mean', 0, np.nan, np.nan, train_MSE, train_RMSE, np.nan, np.nan, test_MSE, test_RMSE])
# source	key0	key1	train_r	train_r2	train_mse	train_rmse	test_r	test_r2	test_mse	test_rmse


temp = pd.DataFrame(
    zip(
    ['Mean', 'G', 'Mean',   np.nan, train_r2, train_MSE, train_RMSE, train_relRMSE, train_normRMSE,
                            np.nan,  test_r2,  test_MSE,  test_RMSE,  test_relRMSE,  test_normRMSE],
    ['Mean', 'S', 'Mean',   np.nan, train_r2, train_MSE, train_RMSE, train_relRMSE, train_normRMSE,
                            np.nan,  test_r2,  test_MSE,  test_RMSE,  test_relRMSE,  test_normRMSE],
    ['Mean', 'W', 'Mean',   np.nan, train_r2, train_MSE, train_RMSE, train_relRMSE, train_normRMSE,
                            np.nan,  test_r2,  test_MSE,  test_RMSE,  test_relRMSE,  test_normRMSE],
    ['Mean', 'All', 'Mean', np.nan, train_r2, train_MSE, train_RMSE, train_relRMSE, train_normRMSE,
                            np.nan,  test_r2,  test_MSE,  test_RMSE,  test_relRMSE,  test_normRMSE],
    ['Mean', 'cat', 'Mean', np.nan, train_r2, train_MSE, train_RMSE, train_relRMSE, train_normRMSE,
                            np.nan,  test_r2,  test_MSE,  test_RMSE,  test_relRMSE,  test_normRMSE])

).T



temp = temp.rename(columns ={
    0:'source',
    1:'key0',
    2:'key1',
    3:'train_r',
    4:'train_r2',
    5:'train_mse',
    6:'train_rmse',
    7:'train_relRMSE',
    8:'train_normRMSE',
    9: 'test_r',
    10:'test_r2',
    11:'test_mse',
    12:'test_rmse',
    13:'test_relRMSE',
    14:'test_normRMSE'
})

summary_df = pd.concat([summary_df, temp])    

# %%
summary_df.loc[:, ['key0', 'key1'
                                              ]].drop_duplicates()

# set(summary_df.key0)

# %%

# %%
summary_df['model_class'] = ''
summary_df['data_source'] = ''
summary_df['model'] = ''
summary_df['replicate'] = ''

## Model Groupings =============================================================
summary_df.loc[summary_df.source == 'ml_res', 'model_class'] = 'ML' 
summary_df.loc[summary_df.source == 'tf_res', 'model_class'] = 'DNN' 
summary_df.loc[summary_df.source == 'lmlike_res', 'model_class'] = 'LM' 
summary_df.loc[summary_df.source == 'bglr_res', 'model_class'] = 'BLUP' 
summary_df.loc[summary_df.source == 'Mean', 'model_class'] = 'LM' 

## Model Names =================================================================
summary_df['model'] = summary_df['key1']

mask = ((summary_df.model_class == 'DNN') & (summary_df.key0 == 'G'))
summary_df.loc[mask, 'model'] = 'DNN-Con.' 
mask = ((summary_df.model_class == 'DNN') & (summary_df.key0 == 'W'))
summary_df.loc[mask, 'model'] = 'DNN-Con.' 
mask = ((summary_df.model_class == 'DNN') & (summary_df.key0 == 'S'))
summary_df.loc[mask, 'model'] = 'DNN-Con.' 
mask = ((summary_df.model_class == 'DNN') & (summary_df.key0 == 'cat'))
summary_df.loc[mask, 'model'] = 'DNN-Con.' 
mask = ((summary_df.model_class == 'DNN') & (summary_df.key0 == 'full'))
summary_df.loc[mask, 'model'] = 'DNN-Sim.' 

## Data Groupings ==============================================================
# # copy and then overwrite the values we want to change
kv_pairs = [
    ['WOneHot', 'W'],
    ['All',     'Multi'],
    ['full',    'Multi'],
    ['cat',     'Multi']
]
summary_df['data_source'] = summary_df['key0']
for i in range(len(kv_pairs)):
    ith_key = kv_pairs[i][0]
    ith_value = kv_pairs[i][1]   
    mask = (summary_df.key0 == ith_key)
    summary_df.loc[mask, 'data_source'] = ith_value 


## Replicate Numbers ===========================================================
mask = summary_df.model_class == 'ML'
summary_df.loc[mask, 'replicate']  = summary_df.loc[mask, 'key2'].str.replace('rep', '')

mask = summary_df.model_class == 'DNN'
summary_df.loc[mask, 'replicate']  = summary_df.loc[mask, 'key1']#.str.replace('rep', '')

mask = summary_df.model_class == 'LM'
summary_df.loc[mask, 'replicate']  = summary_df.loc[mask, 'key2'].str.replace('rep', '')

# empty values (from naive lm/mean) set to replicate 0
summary_df.loc[summary_df.replicate.isna(), 'replicate'] = 0 

# %%

# %%

# %%
# parse lmlikes
summary_df['annotation'] = ''
# mask = summary_df.model_class == 'LM'


# Note -- ubuntu seesm to have no issue with "*" but windows is representing it as "\uf02a"
# to get around this I have duplicated some of the lines below. 
lmlike_additional_info = pd.DataFrame([
# Name     blue/p  annotation                   # Single Data Sources
['Mean',       'Fixed', 'Intercept Only'],      # |--Simple Fixed Models
# ['g8f',        'Fixed', '31% Varience'],        # |  .
# ['g50f',       'Fixed', '50% Varience'],        # |  .
# ['s*f',        'Fixed', 'All Factors'],         # |  .
# ['s\uf02af',        'Fixed', 'All Factors'],         # |  .     --------------------- dupe for encoding
# ['w5f',        'Fixed', 'Top 5 Factors'],       # |  . 
# ['w*f',        'Fixed', 'All Factors'],         # |  .
# ['w\uf02af',        'Fixed', 'All Factors'],         # |  .     --------------------- dupe for encoding
#                                                 # |--Simple Rand. Models
# ['g8r',        'Rand.', '31% Varience'],        #    .
# ['s*r',        'Rand.', 'All Factors'],         #    .
# ['s\uf02ar',        'Rand.', 'All Factors'],         #    .     --------------------- dupe for encoding
# ['w5r',        'Rand.', 'Top 5 Factors'],       #    .

#                                                 # Multiple Sources of Data
# ['g8fw5fs*f',  'Fixed', 'G+S+W'],               # |--Fixed
# ['g8fw5fs\uf02af',  'Fixed', 'G+S+W'],               # |--Fixed --------------------- dupe for encoding
# ['g8fw5f',     'Fixed', 'G+W'],                 # |  .
# ['g8fw5f_gxw', 'Fixed', 'G+W+G:W'],             # |  . - Interaction
#                                                 # |--Rand.
# ['g8rw5r',     'Rand.', '(1|G)+(1|W)'],         #    .
# ['g8rw5r_gxw', 'Rand.', '(1|G)+(1|W)+(1|G:W)'], #    . - Interaction
], columns = ['key1', 'effect_type', 'annotation'])


for i in range(lmlike_additional_info.shape[0]):
    mask = (summary_df.key1 == lmlike_additional_info.loc[i, 'key1'])
    summary_df.loc[mask, 'model'] = lmlike_additional_info.loc[i, 'effect_type']
    summary_df.loc[mask, 'annotation'] = lmlike_additional_info.loc[i, 'annotation']


# %%
# # Replace those labels with  * or \uf02a with something easier to work with:
# troublesome_labels = {
#     'g8fw5fs*f' : ['g8fw5fs\uf02af', 'g8fw5fs*f'], 
#           's*f' : ['s\uf02af', 's*f'], 
#           's*r' : ['s\uf02ar', 's*r'], 
#           'w*f' : ['w\uf02af', 'w*f']
# }
# for ii in troublesome_labels.items():
#     mask = (summary_df.key1.isin(ii[1]))
#     summary_df.loc[mask, 'key1'] = ii[0]


# %%

# %%
summary_df

# %%

# %%

# %%
# for e in list(LLLLLL):
# #     print("'"+e+"': {'model':'Fixed', 'Annotation':''}")
#     print("['"+e+"', 'Fixed', ''],")    

# %%
tmp = summary_df#.loc[:, [
#     'train_rmse',
#     'test_rmse',
#     'model_class',
#     'data_source',
#     'model',
#     'replicate',
#     'annotation'
# ]]
tmp.head()

# %%
tmp2 = tmp

tmp2.loc[tmp2.model == 'rnr', 'model'] =  "Radius NR"
tmp2.loc[tmp2.model == 'knn', 'model'] =  "KNN"
tmp2.loc[tmp2.model == 'rf', 'model'] =  "Rand.Forest"
tmp2.loc[tmp2.model == 'svrl', 'model'] =  "SVR (linear)"

tmp2.loc[tmp2.model == 'lm', 'model'] =  "LM"
# tmp2.loc[tmp2.model == 'Rand.', 'model'] =  "LM (Rand.)"
# tmp2.loc[tmp2.model == 'Fixed', 'model'] =  "LM (fixed)"


tmp2.loc[tmp2.annotation == 'Intercept Only', 'model'] =  "Training Mean"

# %%
# tmp2["data_source_order"] = 0

# tmp2.loc[tmp2.data_source == "G", "data_source_order"] = 0
# tmp2.loc[tmp2.data_source == "S", "data_source_order"] = 1
# tmp2.loc[tmp2.data_source == "W", "data_source_order"] = 2
# tmp2.loc[tmp2.data_source == "Mulit", "data_source_order"] = 3

# tmp2 = tmp2.sort_values("data_source_order")

tmp2.loc[tmp2.model == 'LM', ]


# %%
# tmp2.to_clipboard()

# %% [markdown]
# ### For final figs

# %%
tmp2.to_csv("../output/r_performance_across_models.csv")

# %%

# %% code_folding=[0]
# fig = px.box(
#     tmp2, 
#     x = 'model', 
#     y = 'test_rmse', 
#     facet_col='data_source', 
#     color = 'model', 
#     points="all",
#     hover_data = ['annotation'],
#     title = 'Root Mean Squared Error on Testing Data',
#     labels={
#        "test_rmse": "RMSE",
#        "model": ""
#          },
#     category_orders={"data_source": ["G", "S", "W", "Multi"],
#                      "model":[
#                          'Training Mean',
#                          'LM (fixed)',
#                          'LM (Rand.)',

#                          'Radius NR',
#                          'KNN',
#                          'Rand.Forest',
#                          'SVR (linear)',

#                          'DNN-Sim.',
#                          'DNN-Con.'
#                          ]
#                     },
#     template = 'plotly_white',
#     width = 800, 
#     height= 600
#     )

# fig.update_xaxes(tickangle=-65)
# fig.update_layout(showlegend=False)
# fig.update_layout(yaxis_range=[0.9, 1.61])

# fig.write_image("../output/Performance_RMSE_Box.svg")
# fig

# %% code_folding=[0]
# # -------------------------------------------------------------
# fig = px.scatter(
#     tmp2, 
#     x = 'model', 
#     y = 'test_rmse', 
#     facet_col='data_source', 
#     color = 'model', 

#     hover_data = ['annotation'],
#     title = 'Root Mean Squared Error on Testing Data',
#     labels={
#        "test_rmse": "RMSE",
#        "model": ""
#          },
#     category_orders={"data_source": ["G", "S", "W", "Multi"],
#                      "model":[
#                          'Training Mean',
#                          'LM (fixed)',
#                          'LM (Rand.)',

#                          'Radius NR',
#                          'KNN',
#                          'Rand.Forest',
#                          'SVR (linear)',

#                          'DNN-Sim.',
#                          'DNN-Con.'
#                          ]
#                     },
#     template = 'plotly_white',
#     width = 800, 
#     height= 600
#     )

# fig.update_xaxes(tickangle=-65)
# fig.update_layout(showlegend=False)
# fig.update_layout(yaxis_range=[0.9, 1.61])

# fig.write_image("../output/Performance_RMSE_Scatter.svg")
# fig

# %% code_folding=[0]
# fig = px.box(
#     tmp2, 
#     x = 'model', 
#     y = 'test_r2', 
#     facet_col='data_source', 
#     color = 'model', 
#     points="all",
#     hover_data = ['annotation'],
#     title = 'R2 on Testing Data',
#     labels={
#        "test_r2": "R2",
#        "model": ""
#          },
#     category_orders={"data_source": ["G", "S", "W", "Multi"],
#                      "model":[
#                          'Training Mean',
#                          'LM (fixed)',
#                          'LM (Rand.)',

#                          'Radius NR',
#                          'KNN',
#                          'Rand.Forest',
#                          'SVR (linear)',

#                          'DNN-Sim.',
#                          'DNN-Con.'
#                          ]
#                     },
#     template = 'plotly_white',
#     width = 800, 
#     height= 600
#     )
# fig.update_xaxes(tickangle=-65)
# fig.update_layout(showlegend=False)
# fig.update_layout(yaxis_range=[-1.4, 0.25])

# fig.write_image("../output/Performance_R2_Box.svg")
# fig


# %% code_folding=[0]
# # -------------------------------------------------------------
# fig = px.scatter(
#     tmp2, 
#     x = 'model', 
#     y = 'test_r2', 
#     facet_col='data_source', 
#     color = 'model', 

#     hover_data = ['annotation'],
#     title = 'R2 on Testing Data',
#     labels={
#        "test_r2": "R2",
#        "model": ""
#          },
#     category_orders={"data_source": ["G", "S", "W", "Multi"],
#                      "model":[
#                          'Training Mean',
#                          'LM (fixed)',
#                          'LM (Rand.)',

#                          'Radius NR',
#                          'KNN',
#                          'Rand.Forest',
#                          'SVR (linear)',

#                          'DNN-Sim.',
#                          'DNN-Con.'
#                          ]
#                     },
#     template = 'plotly_white',
#     width = 800, 
#     height= 600
#     )
# fig.update_xaxes(tickangle=-65)
# fig.update_layout(showlegend=False)
# fig.update_layout(yaxis_range=[-1.4, 0.25])

# fig.write_image("../output/Performance_R2_Scatter.svg")
# fig


# %%

# %%

# %% code_folding=[0]
# np.mean(tmp2.loc[((tmp2.data_source == "Multi") & (tmp2.model == "DNN-Con.")), 'test_rmse'])
# np.mean(tmp2.loc[((tmp2.data_source == "W") & (tmp2.model == "DNN-Con.")), 'test_rmse'])

# dnn

# 1.0180510685301187 - 0.9481427239637169

# lm
# 0.9589-0.9814
# -0.022500000000000075

# %% code_folding=[0]
# fig = px.scatter(tmp2, 
#            x = 'model', 
#            y = 'test_r2', 
           
#            facet_col='data_source', 
# #            facet_row='data_source', 
#            color = 'model', 
#            hover_data = ['annotation'],
#            title = 'Correlation Between Prediction, Observation',
#            labels={
#                "test_r2": "R^2",
#                "model": ""
#                  },
#             category_orders={"data_source": ["G", "S", "W", "Multi"],
#                              "model":[
#                                  'Training Mean',
#                                  'DNN-Con.',
#                                  'DNN-Sim.',
#                                  'LM (fixed)',
#                                  'LM (Rand.)',
#                                  'Radius NR',
#                                  'KNN',
#                                  'Rand.Forest',
#                                  'SVR (linear)'
#                                  ]
#                             },
#            template = 'plotly_white')
# fig.update_xaxes(tickangle=45)
# # fig.update_layout(legend=dict(
# #     orientation="h",
# # #     yanchor="top",
# #     y=1,
# #     xanchor="right",
# #     x=1
# # ))

    # %%
    # px.scatter(tmp, 
    #            x = 'model', y = 'test_r2', 
    #            facet_col='data_source', 
    #            color = 'model', 
    #            hover_data = ['annotation'],
    #            template = 'plotly_white')

    # %%
    # px.scatter(tmp,
    #            x = 'train_rmse', 
    #            y = 'test_rmse', 
    # #            color = 'model_class'
    #            color = 'model'
    #           )

# %%

# %%

# %%
## Edit the keys so there are consistent groupings =============================

summary_df.loc[summary_df['key0'] == 'full', 'key0'] = 'All'

# get replicate for tf
summary_df['rep'] = [int(e) if re.findall('\d$', e) else 0 for e in summary_df['key1'] ]
summary_df['key1'] = [e if not re.findall('\d', e) else 'dnn' for e in summary_df['key1'] ]
# then get replicate for ml
summary_df['extracted_rep_num'] = [int(e.replace('rep', '')) if re.findall('rep\d+', str(e)) else -1 for e in summary_df['key2'] ]
mask = (summary_df['source'] == 'ml_res')
summary_df.loc[mask, 'rep'] = summary_df.loc[mask, 'extracted_rep_num']
summary_df = summary_df.drop(columns = ['key2', 'extracted_rep_num'])

summary_df.loc[summary_df['key0'] == 'WOneHot', 'key0'] = 'W'

# %%
# summary_df

# %%

# %%
# [-1 for e in summary_df['key2'] if not re.findall('rep\d+', str(e))]



# %% code_folding=[0]
# # temp = ml_res['All']['knn']

# ## Create Empty Summary DataFrame ==============================================
# # Make a df with all the keys so we can populate it with the statistics we want.
# # make a df with all the keys for each experiment we ran. 
# keys_to_get_ys = []
# for key0 in ml_res.keys():
#     for key1 in ml_res[key0].keys():
#         keys_to_get_ys.append(['ml_res', key0, key1])
        
# for key0 in tf_res.keys():
#     for key1 in tf_res[key0].keys():
#         keys_to_get_ys.append(['tf_res', key0, key1])

# summary_df = pd.DataFrame(keys_to_get_ys, columns = ['source', 'key0', 'key1'])


# summary_df[['train_r','train_r2','train_mse','train_rmse', 
#             'test_r', 'test_r2', 'test_mse', 'test_rmse']] = np.nan
# summary_df.head()


# ## Populate Summary DataFrame with Error Statistics ============================

# def replace_na_w_0(in_df = temp,
#                    col_vars = ['y_train', 'yHat_train']):
#     for col_var in col_vars:
#         na_mask = in_df[col_var].isna()
#         n_missings = sum(na_mask)
#         if n_missings > 0:
#             print('Replacing '+str(n_missings)+' with 0.')
#             in_df.loc[na_mask, col_var] = 0
#     return(in_df)

# def calc_performances(observed, predicted, outtype = 'list'):
#     r = stats.pearsonr(observed, predicted)[0] # index 1 is the p value
#     r2 = r**2
#     MSE = mean_squared_error(observed, predicted)
#     RMSE = math.sqrt(MSE)
    
#     if outtype == 'list':
#         return([r, r2, MSE, RMSE])
#     else:
#         return({'r':r,
#                'r2':r2,
#               'mse':MSE,
#              'rmse':RMSE})

# for i in summary_df.index:
#     source = summary_df.loc[i, 'source']
#     key0 = summary_df.loc[i, 'key0']
#     key1 = summary_df.loc[i, 'key1']
    
#     if source == 'ml_res':
#         temp = pd.DataFrame()
#         temp['y_train'] = ml_res[key0][key1]['y_train']
#         temp['yHat_train'] = ml_res[key0][key1]['yHat_train']
#         temp = replace_na_w_0(in_df = temp, col_vars = ['y_train', 'yHat_train'])
#         summary_df.loc[i, ['train_r','train_r2','train_mse','train_rmse']] = calc_performances(observed = temp['y_train'], predicted = temp['yHat_train'] )

#         temp = pd.DataFrame()
#         temp['y_test'] = ml_res[key0][key1]['y_test']
#         temp['yHat_test'] = ml_res[key0][key1]['yHat_test']
#         temp = replace_na_w_0(in_df = temp, col_vars = ['y_test', 'yHat_test'])
#         summary_df.loc[i, ['test_r', 'test_r2', 'test_mse', 'test_rmse']] = calc_performances(observed = temp['y_test'], predicted = temp['yHat_test'])


#     if source == 'tf_res':
#         temp = pd.DataFrame()
#         temp['y_train'] = tf_res[key0][key1]['y_train']
#         temp['yHat_train'] = [e[0] for e in tf_res[key0][key1]['yHat_train']]
#         temp = replace_na_w_0(in_df = temp, col_vars = ['y_train', 'yHat_train'])
#         summary_df.loc[i, ['train_r','train_r2','train_mse','train_rmse']] = calc_performances(observed = temp['y_train'], predicted = temp['yHat_train'] )

#         temp = pd.DataFrame()
#         temp['y_test'] = tf_res[key0][key1]['y_test']
#         temp['yHat_test'] = [e[0] for e in tf_res[key0][key1]['yHat_test']]
#         temp = replace_na_w_0(in_df = temp, col_vars = ['y_test', 'yHat_test'])
#         summary_df.loc[i, ['test_r', 'test_r2', 'test_mse', 'test_rmse']] = calc_performances(observed = temp['y_test'], predicted = temp['yHat_test'])

# summary_df


# ## Add in a Naieve estimate -- Guessing the Training Mean Every Time ===========
# ys = tf_res['G']['0']['y_train']
# yNaive = [0 for e in range(len(ys))]

# train_MSE = mean_squared_error(ys, yNaive)
# train_RMSE = math.sqrt(train_MSE)



# ys = tf_res['G']['0']['y_test']
# yNaive = [0 for e in range(len(ys))]

# test_MSE = mean_squared_error(ys, yNaive)
# test_RMSE = math.sqrt(test_MSE)

# # pd.concat(summary_df , ['Mean', 'Mean', 0, np.nan, np.nan, train_MSE, train_RMSE, np.nan, np.nan, test_MSE, test_RMSE])
# # source	key0	key1	train_r	train_r2	train_mse	train_rmse	test_r	test_r2	test_mse	test_rmse


# temp = pd.DataFrame(
#     zip(['Mean', 'G', 'Mean', np.nan, np.nan, train_MSE, train_RMSE, np.nan, np.nan, test_MSE, test_RMSE],
#     ['Mean', 'S', 'Mean', np.nan, np.nan, train_MSE, train_RMSE, np.nan, np.nan, test_MSE, test_RMSE],
#     ['Mean', 'W', 'Mean', np.nan, np.nan, train_MSE, train_RMSE, np.nan, np.nan, test_MSE, test_RMSE],
#     ['Mean', 'All', 'Mean', np.nan, np.nan, train_MSE, train_RMSE, np.nan, np.nan, test_MSE, test_RMSE],
#     ['Mean', 'cat', 'Mean', np.nan, np.nan, train_MSE, train_RMSE, np.nan, np.nan, test_MSE, test_RMSE])

# ).T

# temp = temp.rename(columns ={
#     0:'source',
#     1:'key0',
#     2:'key1',
#     3:'train_r',
#     4:'train_r2',
#     5:'train_mse',
#     6:'train_rmse',
#     7:'test_r',
#     8:'test_r2',
#     9:'test_mse',
#     10:'test_rmse'    
# })

# summary_df = pd.concat([summary_df, temp])    


# ## Edit the keys so there are consistent groupings =============================

# summary_df.loc[summary_df['key0'] == 'full', 'key0'] = 'All'

# summary_df['rep'] = [int(e) if re.findall('\d', e) else 0 for e in summary_df['key1'] ]
# summary_df['key1'] = [e if not re.findall('\d', e) else 'dnn' for e in summary_df['key1'] ]


# summary_df.loc[summary_df['key0'] == 'WOneHot', 'key0'] = 'W'

# %%
summary_df.head()

# %% [markdown]
# ## Visualization

# %%
# mask = ((summary_df['key1'] == 'dnn') & (summary_df['key0'] == 'cat'))
# summary_df.loc[mask, ['key0', 'key1'] ] = ['All', 'dnn\'']

# summary_df = summary_df.loc[summary_df['key0'] != 'cat']

# %%
# px.scatter(summary_df, 
#            x = 'key1', y = 'test_rmse', facet_col='key0', 
#            color = 'key1', 
#            template = 'plotly_white')


# %%
# px.scatter(summary_df, 
#            x = 'key1', y = 'test_r2', facet_col='key0', 
#            color = 'key1', 
#            template = 'plotly_white')

# %%
# fig = px.box(summary_df, 
#            x = 'key1', y = 'test_rmse', facet_col='key0', 
#            color = 'key1', 
#            template = 'plotly_white')

# # fig.write_html('RMSE_compare.html')

# fig


# %%
# fig = px.box(summary_df, 
#            x = 'key1', y = 'test_r2', facet_col='key0', 
#            color = 'key1', 
#            template = 'plotly_white')

# # fig.write_html('RMSE_compare.html')

# fig

# %% [markdown]
# ## Look for Evidence of Over Training

# %%
# for i in range(4):
#     for j in range(10):
#         group0 = ['G', 'S', 'W', 'full'][i]
#         group1 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'][j]

df_list = []


for group0 in ['G', 'S', 'W', 'full', 'cat']:
    for group1 in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
#         print(group0)
        temp = tf_res[group0][group1]
        

        epochs = [e for e in range(len(temp['loss']))]
        temp_df = pd.DataFrame()
        
        temp_df['epoch'] = epochs
        temp_df['loss'] = temp['loss']
        temp_df['val_loss'] = temp['val_loss']
        temp_df['Network'] = group0
        temp_df['Replicate'] = group1
        
        df_list.append(temp_df)

df = pd.concat(df_list)
df = df.melt(id_vars=['epoch', 'Network', 'Replicate'], value_vars=['loss', 'val_loss'] , var_name= 'Set', value_name='Loss')



# %%
df.loc[df['Set'] == 'loss', 'Set'] = 'Training'
df.loc[(df.Set == 'val_loss'), 'Set'] = 'Test'

# %%
df.to_csv("../output/r_overtraining_across_models.csv")

# %%
df

# %% [markdown]
# ### Genome 

# %%
px.line(df.loc[df.Network == 'G', :],
        x = 'epoch', y = 'Loss', 
        color = 'Set',
        facet_col = 'Replicate', facet_row = 'Network', 
       template = 'plotly_white')

# %%
px.line(df.loc[((df.Set == 'Test') & (df.Network == 'G')), :],
        x = 'epoch', y = 'Loss', 
        # color = 'Replicate',
        facet_col = 'Replicate',
        template = 'plotly_white')

# %%
df_cross_rep_mean = df.groupby(['Set', 'epoch', 'Network']).agg(Loss = ('Loss', np.mean)).reset_index()


mask = ((df_cross_rep_mean.Set == 'Test') & (df_cross_rep_mean.Network == 'G'))


fig = px.line(df_cross_rep_mean.loc[mask, :],
        x = 'epoch', y = 'Loss', 
        # color = 'Replicate',
#         facet_col = 'Replicate',
        template = 'plotly_white')

df_temp_min = df_cross_rep_mean.loc[mask, :].loc[(df_cross_rep_mean.loc[mask, 'Loss'] == np.min(df_cross_rep_mean.loc[mask, 'Loss'])), ]
fig.add_trace(go.Scatter(x = df_temp_min.epoch,
                         y = df_temp_min.Loss ))

fig

# %% [markdown]
# ### Soil 

# %%
px.line(df.loc[df.Network == 'S', :],
        x = 'epoch', y = 'Loss', 
        color = 'Set',
        facet_col = 'Replicate', facet_row = 'Network', 
       template = 'plotly_white')

# %%
px.line(df.loc[((df.Set == 'Test') & (df.Network == 'S')), :],
        x = 'epoch', y = 'Loss', 
        # color = 'Replicate',
        facet_col = 'Replicate',
        template = 'plotly_white')

# %%
df_cross_rep_mean = df.groupby(['Set', 'epoch', 'Network']).agg(Loss = ('Loss', np.mean)).reset_index()


mask = ((df_cross_rep_mean.Set == 'Test') & (df_cross_rep_mean.Network == 'S'))


fig = px.line(df_cross_rep_mean.loc[mask, :],
        x = 'epoch', y = 'Loss', 
        # color = 'Replicate',
#         facet_col = 'Replicate',
        template = 'plotly_white')

df_temp_min = df_cross_rep_mean.loc[mask, :].loc[(df_cross_rep_mean.loc[mask, 'Loss'] == np.min(df_cross_rep_mean.loc[mask, 'Loss'])), ]
fig.add_trace(go.Scatter(x = df_temp_min.epoch,
                         y = df_temp_min.Loss ))

fig

# %% [markdown]
# ### Weather 

# %%
px.line(df.loc[df.Network == 'W', :],
        x = 'epoch', y = 'Loss', 
        color = 'Set',
        facet_col = 'Replicate', facet_row = 'Network', 
       template = 'plotly_white')

# %%
px.line(df.loc[((df.Set == 'Test') & (df.Network == 'W')), :],
        x = 'epoch', y = 'Loss', 
        # color = 'Replicate',
        facet_col = 'Replicate',
        template = 'plotly_white')

# %%
df_cross_rep_mean = df.groupby(['Set', 'epoch', 'Network']).agg(Loss = ('Loss', np.mean)).reset_index()


mask = ((df_cross_rep_mean.Set == 'Test') & (df_cross_rep_mean.Network == 'W'))


fig = px.line(df_cross_rep_mean.loc[mask, :],
        x = 'epoch', y = 'Loss', 
        # color = 'Replicate',
#         facet_col = 'Replicate',
        template = 'plotly_white')

df_temp_min = df_cross_rep_mean.loc[mask, :].loc[(df_cross_rep_mean.loc[mask, 'Loss'] == np.min(df_cross_rep_mean.loc[mask, 'Loss'])), ]
fig.add_trace(go.Scatter(x = df_temp_min.epoch,
                         y = df_temp_min.Loss ))

fig

# %% [markdown]
# ### Full 

# %%
px.line(df.loc[df.Network == 'full', :],
        x = 'epoch', y = 'Loss', 
        color = 'Set',
        facet_col = 'Replicate', facet_row = 'Network', 
#        facet_row = 'Replicate', facet_col = 'Network', 
       template = 'plotly_white')

# %%
px.line(df.loc[((df.Set == 'Test') & (df.Network == 'full')), :],
        x = 'epoch', y = 'Loss', 
        # color = 'Replicate',
        facet_col = 'Replicate',
        template = 'plotly_white')

# %%
df_cross_rep_mean = df.groupby(['Set', 'epoch', 'Network']).agg(Loss = ('Loss', np.mean)).reset_index()


mask = ((df_cross_rep_mean.Set == 'Test') & (df_cross_rep_mean.Network == 'full'))


fig = px.line(df_cross_rep_mean.loc[mask, :],
        x = 'epoch', y = 'Loss', 
        # color = 'Replicate',
#         facet_col = 'Replicate',
        template = 'plotly_white')

df_temp_min = df_cross_rep_mean.loc[mask, :].loc[(df_cross_rep_mean.loc[mask, 'Loss'] == np.min(df_cross_rep_mean.loc[mask, 'Loss'])), ]
fig.add_trace(go.Scatter(x = df_temp_min.epoch,
                         y = df_temp_min.Loss ))

fig

# %% [markdown]
# ### Cat 

# %%
px.line(df.loc[df.Network == 'cat', :],
        x = 'epoch', y = 'Loss', 
        color = 'Set',
        facet_col = 'Replicate', facet_row = 'Network', 
#        facet_row = 'Replicate', facet_col = 'Network', 
       template = 'plotly_white')

# %%
px.line(df.loc[((df.Set == 'Test') & (df.Network == 'cat')), :],
        x = 'epoch', y = 'Loss', 
        # color = 'Replicate',
        facet_col = 'Replicate',
        template = 'plotly_white')

# %%
df_cross_rep_mean = df.groupby(['Set', 'epoch', 'Network']).agg(Loss = ('Loss', np.mean)).reset_index()


mask = ((df_cross_rep_mean.Set == 'Test') & (df_cross_rep_mean.Network == 'cat'))


fig = px.line(df_cross_rep_mean.loc[mask, :],
        x = 'epoch', y = 'Loss', 
        # color = 'Replicate',
#         facet_col = 'Replicate',
        template = 'plotly_white')

df_temp_min = df_cross_rep_mean.loc[mask, :].loc[(df_cross_rep_mean.loc[mask, 'Loss'] == np.min(df_cross_rep_mean.loc[mask, 'Loss'])), ]
fig.add_trace(go.Scatter(x = df_temp_min.epoch,
                         y = df_temp_min.Loss ))

fig

# %% [markdown]
# ### All Models

# %%
px.line(df.loc[df.Set == 'Test', :],
        x = 'epoch', y = 'Loss', 
        color = 'Replicate',
        facet_row = 'Network', 
       template = 'plotly_white')

# %% [markdown]
# #  Predictions by condition

# %%
# go across all tf models and replicates. 
# Produce a csv with the predictions for each so that we can get RMSE conditional on location.

# tf_key = "cat"
# tf_rep = '1'

for tf_key in ['G', 'S', 'W', 'full', 'cat']:
    for i in range(10):
        tf_rep = str(i)

        # setup data placeholders
        p_copy = phenotype.copy()
        p_copy = p_copy.loc[:, ['Unnamed: 0', 'Pedigree', 'F', 'M', 'ExperimentCode', 
                                'Year', 'DatePlanted', 'DateHarvested', 'Split', 'Hybrid'
                      ]].rename(columns = {'Unnamed: 0':"TensorIndex"})


        # p_copy['y'] = np.nan
        # p_copy['yHat'] = np.nan

        p_train = p_copy.loc[p_copy.index.isin(trainIndex) ]
        p_train["Split"] = "Train"
        p_test = p_copy.loc[p_copy.index.isin(testIndex) ]
        p_test["Split"]  = "Test"

        p_summary = pd.concat([p_train, p_test])
        p_summary["Model"] = tf_key
        p_summary["Rep"] = tf_rep


        # pull data
        if tf_rep == '0':
            temp_test = pd.DataFrame({
                'TensorIndex' : testIndex,
                'y': tf_res[tf_key][tf_rep]['y_test'],
                'yHat_Rep'+tf_rep: [e[0] for e in tf_res[tf_key][tf_rep]['yHat_test']]
            })
            p_test = p_test.merge(temp_test, on = "TensorIndex")

            temp_train = pd.DataFrame({
                'TensorIndex' : trainIndex,
                'y': tf_res[tf_key][tf_rep]['y_train'],
                'yHat_Rep'+tf_rep: [e[0] for e in tf_res[tf_key][tf_rep]['yHat_train']]

            })
            p_train = p_train.merge(temp_train, on = "TensorIndex")
        else:
            temp_test = pd.DataFrame({
                'TensorIndex' : testIndex,
                'yHat_Rep'+tf_rep: [e[0] for e in tf_res[tf_key][tf_rep]['yHat_test']]
            })
            p_test = p_test.merge(temp_test, on = "TensorIndex")

            temp_train = pd.DataFrame({
                'TensorIndex' : trainIndex,
                'yHat_Rep'+tf_rep: [e[0] for e in tf_res[tf_key][tf_rep]['yHat_train']]
            })
            p_train = p_train.merge(temp_train, on = "TensorIndex")


        if tf_rep == '0':
            output = pd.concat([p_test, p_train])
        else:
            output = output.merge(pd.concat([p_test, p_train]))

    output['Model'] = tf_key
    output.to_csv("../output/errorConditionalDNN_"+tf_key+".csv")

# %%

# %%
for bglr_key in ['G', 'S', 'W', 'Multi']:
    for i in range(10):
        bglr_rep = str(i)
    
        # setup data placeholders
        p_copy = phenotype.copy()
        p_copy = p_copy.loc[:, ['Unnamed: 0', 'Pedigree', 'F', 'M', 'ExperimentCode', 
                                'Year', 'DatePlanted', 'DateHarvested', 'Split', 'Hybrid'
                      ]].rename(columns = {'Unnamed: 0':"TensorIndex"})


        # p_copy['y'] = np.nan
        # p_copy['yHat'] = np.nan

        p_train = p_copy.loc[p_copy.index.isin(trainIndex) ]
        p_train["Split"] = "Train"
        p_test = p_copy.loc[p_copy.index.isin(testIndex) ]
        p_test["Split"]  = "Test"

        p_summary = pd.concat([p_train, p_test])
        p_summary["Model"] = bglr_key
        p_summary["Rep"] = bglr_rep  

        # pull data
        if bglr_rep == "0":
            temp_test = pd.DataFrame({
                'TensorIndex' : testIndex,
                'y': bglr_res[bglr_key]["BLUP"]["rep"+bglr_rep]['y_test'],
                'yHat_Rep'+bglr_rep: bglr_res[bglr_key]["BLUP"]["rep"+bglr_rep]['yHat_test']
            })
            p_test = p_test.merge(temp_test, on = "TensorIndex")

            temp_train = pd.DataFrame({
                'TensorIndex' : trainIndex,
                'y': bglr_res[bglr_key]["BLUP"]["rep"+bglr_rep]['y_train'],
                'yHat_Rep'+bglr_rep: bglr_res[bglr_key]["BLUP"]["rep"+bglr_rep]['yHat_train']

            })
            p_train = p_train.merge(temp_train, on = "TensorIndex")
        else:
            temp_test = pd.DataFrame({
                'TensorIndex' : testIndex,
                'yHat_Rep'+bglr_rep: bglr_res[bglr_key]["BLUP"]["rep"+bglr_rep]['yHat_test']
            })
            p_test = p_test.merge(temp_test, on = "TensorIndex")

            temp_train = pd.DataFrame({
                'TensorIndex' : trainIndex,
                'yHat_Rep'+bglr_rep: bglr_res[bglr_key]["BLUP"]["rep"+bglr_rep]['yHat_train']
            })
            p_train = p_train.merge(temp_train, on = "TensorIndex")            
            
        if bglr_rep == '0':
            output = pd.concat([p_test, p_train])
        else:
            output = output.merge(pd.concat([p_test, p_train]))

    output['Model'] = bglr_key
    output.to_csv("../output/errorConditionalRKHS_"+bglr_key+".csv")

# %%
bglr_res.keys()

# %%
bglr_res["G"]["BLUP"]["rep0"].keys()


# %% [markdown]
# # How are model errors related?
# * Do models perform poorly in their own ways?
# * Are there features of the data that are correlated with better or worse performance? 
#

# %%
# Aggregate all the errors for comparison

def mk_error_df(tf_res = tf_res,
                dnn_class = 'G', 
                dnn_rep = '0'):
    temp = pd.DataFrame()
    temp['yHat'] = [e[0] for e in tf_res[dnn_class][dnn_rep]['yHat_test']]
    temp['y']    =  tf_res[dnn_class][dnn_rep]['y_test']
    temp['error'] = temp['yHat'] - temp['y']
    temp['ModelGroup'] = dnn_class
    temp['ModelRep'] = dnn_rep
    temp = temp.reset_index().rename(columns = {'index':'observation'})
    return(temp)



df_accumulator = []
for dnn_class in ['G', 'S', 'W', 'full']:
    df_accumulator.extend( [mk_error_df(tf_res = tf_res, dnn_class = dnn_class, dnn_rep = str(dnn_rep)) for dnn_rep in range(10)] )
df_accumulator = pd.concat(df_accumulator)
df_accumulator

# %%
temp = df_accumulator.loc[:, ['observation', 'error', 'ModelGroup', 'ModelRep']
                  ].pivot_table(
    index = ['ModelGroup', 'ModelRep'], 
    columns = ['observation'], 
    values = ['error']
    )#.reset_index()

pca = PCA(n_components = 3)

pca.fit(temp)

print(pca.explained_variance_ratio_)



# temp
# temp


# temp_pca = temp.loc[:, ]


# temp_pca.reset_index()

# temp_pca.reset_index().loc[:, ['ModelGroup', 'ModelRep']]

temp_pca = pd.DataFrame(pca.transform(temp))
temp_pca['ModelRep'] = temp.reset_index().loc[:, 'ModelRep']
temp_pca['ModelGroup'] = temp.reset_index().loc[:, 'ModelGroup']

temp_pca


px.scatter_3d(temp_pca, 
              x = 0, 
              y = 1, 
              z = 2,
             color = 'ModelGroup')

# %%

# %%
df_corrs = df_accumulator.pivot_table(
    index = 'observation', 
    columns = ['ModelGroup', 'ModelRep'], 
    values = 'error').corr().round(3)

df_corrs.head()

# %%

# %%
df_accumulator['Model'] = df_accumulator['ModelGroup']+'-'+df_accumulator['ModelRep']


df_corrs = df_accumulator.pivot_table(
    index = 'observation', 
    columns = ['Model'], 
    values = 'error').corr().round(3)#.reset_index()

df_corrs.head()

# %%
# Visualizing a Pandas Correlation Matrix Using Seaborn


sns.heatmap(df_corrs#, annot=True
           )
plt.show()

# %%

# %%


fig = px.imshow(df_corrs, aspect="auto", color_continuous_scale='RdBu_r')
fig.show()

# %%
# maybe the way to think about this with the within group sd...


# I want to know 1. How consistent are a group of models?
# How much information can we get by combining models? Are there complements to be had?

# %%
# temp = pd.DataFrame(np.triu(df_corrs.to_numpy()), columns = list(df_corrs))

# temp['variable1'] = list(df_corrs)
# temp = temp.melt(id_vars = 'variable1')
# temp

# %%


# %%
temp = df_accumulator.loc[df_accumulator.ModelGroup == 'G', ].pivot_table(
        index = 'observation', 
        columns = 'ModelRep', 
        values = 'error'
        ).corr(
        ).round(3
        ).reset_index(
        ).drop(columns = 'ModelRep'
        ).melt()

temp = temp.loc[temp.value != 1, ]

px.box(temp, x = "ModelRep", y = 'value', points = 'all')


# %%

# %%

# %%

# %%
tf_res['G']['0'].keys()

# 'y_test', 'yHat_test'

# %%
tf_res.keys()

# %%

# %% [markdown]
# ## What are the high error samples?

# %%
# go across all tf models and replicates. 
# Produce a csv with the predictions for each so that we can get RMSE conditional on location.

# tf_key = "cat"
# tf_rep = '1'

for tf_key in ['G', 'S', 'W', 'full', 'cat']:
    for i in range(10):
        tf_rep = str(i)

        # setup data placeholders
        p_copy = phenotype.copy()
        p_copy = p_copy.loc[:, ['Unnamed: 0', 'Pedigree', 'F', 'M', 'ExperimentCode', 
                                'Year', 'DatePlanted', 'DateHarvested', 'Split', 'Hybrid'
                      ]].rename(columns = {'Unnamed: 0':"TensorIndex"})


        # p_copy['y'] = np.nan
        # p_copy['yHat'] = np.nan

        p_train = p_copy.loc[p_copy.index.isin(trainIndex) ]
        p_train["Split"] = "Train"
        p_test = p_copy.loc[p_copy.index.isin(testIndex) ]
        p_test["Split"]  = "Test"

        p_summary = pd.concat([p_train, p_test])
        p_summary["Model"] = tf_key
        p_summary["Rep"] = tf_rep


        # pull data
        if tf_rep == '0':
            temp_test = pd.DataFrame({
                'TensorIndex' : testIndex,
                'y': tf_res[tf_key][tf_rep]['y_test'],
                'yHat_Rep'+tf_rep: [e[0] for e in tf_res[tf_key][tf_rep]['yHat_test']]
            })
            p_test = p_test.merge(temp_test, on = "TensorIndex")

            temp_train = pd.DataFrame({
                'TensorIndex' : trainIndex,
                'y': tf_res[tf_key][tf_rep]['y_train'],
                'yHat_Rep'+tf_rep: [e[0] for e in tf_res[tf_key][tf_rep]['yHat_train']]

            })
            p_train = p_train.merge(temp_train, on = "TensorIndex")
        else:
            temp_test = pd.DataFrame({
                'TensorIndex' : testIndex,
                'yHat_Rep'+tf_rep: [e[0] for e in tf_res[tf_key][tf_rep]['yHat_test']]
            })
            p_test = p_test.merge(temp_test, on = "TensorIndex")

            temp_train = pd.DataFrame({
                'TensorIndex' : trainIndex,
                'yHat_Rep'+tf_rep: [e[0] for e in tf_res[tf_key][tf_rep]['yHat_train']]
            })
            p_train = p_train.merge(temp_train, on = "TensorIndex")


        if tf_rep == '0':
            output = pd.concat([p_test, p_train])
        else:
            output = output.merge(pd.concat([p_test, p_train]))

    output.to_csv("../output/errorConditionalDNN_"+tf_key+".csv")

# %%

# %%
p_test.info()

# %%

p_test.merge(temp_test, on = "TensorIndex")

# %%

# p_summary

# %%
temp = pd.DataFrame({
    'tensor_index' : testIndex,
    'yHat_test': [e[0] for e in tf_res['cat']['0']['yHat_test']],
    'y_test': tf_res['cat']['0']['y_test']
})

temp['errors'] = temp['yHat_test'] - temp['y_test']

group_labels = ['DNN']

p_copy = phenotype.copy()
p_copy['error_group'] = ''

# filter down to test set
mask = [True if e in testIndex else False for e in p_copy.index]
p_copy = p_copy.loc[mask, ]


p_copy['y_test'] = np.nan
p_copy['yHat_test'] = np.nan
p_copy['errors'] = np.nan
p_copy['quantile_group'] = ''

for i in p_copy.index:
    p_copy.loc[i, 'y_test'] =  float(temp.loc[temp.tensor_index == i, 'y_test'])
    p_copy.loc[i, 'yHat_test'] =  float(temp.loc[temp.tensor_index == i, 'yHat_test'])
    p_copy.loc[i, 'errors'] =  float(temp.loc[temp.tensor_index == i, 'errors'])
#     p_copy.loc[i, 'quantile_group'] =  list(temp.loc[temp.tensor_index == i, 'quantile_group'])[0]

p_copy['GrainYield_scaled'] = (p_copy['GrainYield'] - YMean)/YStd

p_copy["Year"] = p_copy.Year.astype(str)
p_copy["Exp_Year"] = p_copy["ExperimentCode"]+"_"+p_copy["Year"]
p_copy = p_copy.sort_values(by = "Exp_Year")
p_copy['abs_errors'] = np.abs(p_copy['errors'])


p_copy

# %%
figg=px.box(p_copy, x = "Exp_Year", #x = 'Exp_Year', 
       y = 'abs_errors'#, color = 'ExperimentCode'#, points = "all"
#           width=800, height=600
      )
figg

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

# %%

# %%

# %%

# %%
# testIndex

# %%
temp = pd.DataFrame({
    'tensor_index' : testIndex,
    'yHat_test': [e[0] for e in tf_res['cat']['0']['yHat_test']],
    'y_test': tf_res['cat']['0']['y_test']
})


temp['errors'] = temp['yHat_test'] - temp['y_test']


q10 = np.quantile(temp['errors'], .1)
q90 = np.quantile(temp['errors'], .9)
print('Most extreme 20%:'+' <= '+str(q10)+' & >= '+ str(q90))

temp['quantile_group'] = 'q10toq90'
temp.loc[temp.errors <= q10, 'quantile_group'] = 'q10'
temp.loc[temp.errors >= q90, 'quantile_group'] = 'q90'



# x = np.random.randn(1000)
# hist_data = [x]
# group_labels = ['distplot'] # name of the dataset

hist_data = [temp['errors']]
group_labels = ['DNN']

fig = ff.create_distplot(hist_data, group_labels)
fig.show()


# %%

# %%
p_copy = phenotype.copy()
p_copy['error_group'] = ''

# what's going on with the predictions where it undershot?
indices = temp.loc[temp.quantile_group == 'q10', 'tensor_index']
mask = [True if e in indices else False for e in p_copy.index]
p_copy.loc[mask, 'error_group'] = 'under'

# what's going on with the predictions where it overshot?
indices = temp.loc[temp.quantile_group == 'q90', 'tensor_index']
mask = [True if e in indices else False for e in p_copy.index]

p_copy.loc[mask, 'error_group'] = 'over'



# filter down to test set

mask = [True if e in testIndex else False for e in p_copy.index]
p_copy = p_copy.loc[mask, ]

# %%
# temp['errors']

# %%

# %%
p_copy['y_test'] = np.nan
p_copy['yHat_test'] = np.nan
p_copy['errors'] = np.nan
p_copy['quantile_group'] = ''

for i in p_copy.index:
    p_copy.loc[i, 'y_test'] =  float(temp.loc[temp.tensor_index == i, 'y_test'])
    p_copy.loc[i, 'yHat_test'] =  float(temp.loc[temp.tensor_index == i, 'yHat_test'])
    p_copy.loc[i, 'errors'] =  float(temp.loc[temp.tensor_index == i, 'errors'])
    p_copy.loc[i, 'quantile_group'] =  list(temp.loc[temp.tensor_index == i, 'quantile_group'])[0]

    
p_copy['GrainYield_scaled'] = (p_copy['GrainYield'] -  YMean)/YStd

# %%
# sanity check. 
px.scatter(p_copy, x = 'GrainYield_scaled', y = 'y_test')

# px.scatter(p_copy, x = 'GrainYield', y = 'y_test')

# %%
np.mean(p_copy['errors'])


sns.kdeplot(p_copy['errors'], shade = True)

# %%
px.scatter(p_copy, x = 'GrainYield_scaled', y = 'errors', #color = 'quantile_group',
           marginal_y="histogram",
          width=800, height=600)



# %%
# X_lognorm = np.random.lognormal(mean=0.0, sigma=1.7, size=500)


qq = stats.probplot(p_copy['errors'], dist='norm', sparams=(1))
x = np.array([qq[0][0][0], qq[0][0][-1]])

fig = go.Figure()
fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
fig.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines')
fig.layout.update(showlegend=False)
fig.show()



# %%

# %%
# temp = p_copy.loc[:, ['F', 'M', 'quantile_group', 'errors']].groupby(['F', 'M', 'quantile_group']).count().reset_index()

# px.scatter_3d(p_copy, x = 'F', y = 'M', z = 'errors', color = 'quantile_group')


# %%

# %%
# temp = p_copy.loc[:, ['ExperimentCode', 'Year', 'quantile_group', 'errors']].groupby(['ExperimentCode', 'Year', 'quantile_group']).count().reset_index()

# px.scatter_3d(p_copy, x = 'ExperimentCode', y = 'Year', z = 'errors', color = 'quantile_group')


# %%
# px.scatter(p_copy, x = 'GrainYield_scaled', y = 'errors', color = 'quantile_group')

# %%
# What do we want to compare? 
# Pedigree (F/M)
# ExperimentCode
# Year
# Yield



# %%

p_copy

# %%
p_copy["Year"] = p_copy.Year.astype(str)



# %%
p_copy["Exp_Year"] = p_copy["ExperimentCode"]+"_"+p_copy["Year"]

p_copy = p_copy.sort_values(by = "Exp_Year")

# %%
p_copy['abs_errors'] = np.abs(p_copy['errors'])

figg=px.box(p_copy, x = "Exp_Year", #x = 'Exp_Year', 
       y = 'abs_errors'#, color = 'ExperimentCode'#, points = "all"
#           width=800, height=600
      )
figg

# %%
figg.write_image("../../../../Desktop/quickplt.svg")

# %%
px.scatter(p_copy, x = "y_test", y = "errors")


# %%

# %%
px.scatter(p_copy, x = "Pedigree", y = "abs_errors")

# %%
px.box(p_copy, x = "F", y = "abs_errors")

# %%
px.box(p_copy, x = "M", y = "abs_errors")


# %%

# %% [markdown]
# # What are the salient features of the datasets?

# %% [markdown]
# ## Functions

# %% code_folding=[0, 182]
class dnn_model:
    def __init__(self, model_path, model_xs):
        self.model_path = model_path
        self.model = None
        self.model_xs = model_xs # data that the model will operate on. 3 tensors for full model, 1 tensor for rebuild models
        self.saliency_map = None
        
        self.results = {}
    
    # Methods for setup ============================================================================
    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
    
    
    # This function fixes a trouble with single tensor models -- they use one tensor but take several. 
    # That's be design to promote code reuse for multi tensor models but it prevents us from making salience maps.
    # The solution is to use the weights from each trained model to make a new model that has an identical (sequential)
    # structure but lacks the unused input tensors. 
    # 
    # Originally I used model.summary() to do this with code for each input type (G, S, W) _BUT_ this fails with replicates 
    # because the level indexer is not conserved (e.g. s replicate 0  has "dropout" up to "dropout_6" but replicate 1 doesn't
    # reset these -- it starts with "dropout_7")

    def remove_unused_inputs_for_salience(self,
        model_to_inspect, # = model_s,
        start_layer,      # = 'sIn',
        stop_layer        # = 'yHat'
    ):

        # look at all layers in the model and create key value pairs with each key being a layer and each value beiing the layer that follows it.
        def get_connection_dict_from_linear_dnn(model_to_inspect): # e.g. model_s
            model_dict = {}

            for nn_layer in model_to_inspect.layers:
                nn_layer_name = nn_layer.name
                nn_layer_input = model_to_inspect.get_layer(nn_layer_name).input.name.split('/')[0]

                # inputs have 'name:0' structure -- remove the ':0' so we have the right layer name
                if re.match("\w+:0", nn_layer_input):
                    nn_layer_input = nn_layer_input.replace(':0', '')
                # non-used inputs will have themselves as an output. (e.g. 'gIn:0' : 'gIn' ) since we removed the ':0' we can 
                # just check if the layer and output are the same. 
                if nn_layer_input == nn_layer_name:
                    pass
                else:
                    model_dict.update({nn_layer_input:nn_layer_name})

            return(model_dict)


        # wrap making the connection dictionary and then step throught the key/values of the dict. 
        # return a list with the sequence of layers. 
        def connection_dict_to_layer_list(model_dict,
                                         start_layer = 'sIn',
                                          stop_layer = 'yHat' # there will be an error for the last layer (not in dict) so we use it as a stopping condition.
                                         ):
            # search for the start layer. Even it's not immute to getting an indexer.
            start_layer = [e for e in list(model_dict.keys()) if re.match(start_layer+'[_\d+]*', e)][0]

            layer_list = [start_layer.split('_')[0]]
            # Transform the dictionary into a list

            next_layer = model_dict[start_layer]
            layer_list.extend([next_layer])

            for i in range(len(model_dict.keys()) - 1): # 
                if next_layer != stop_layer:
                    next_layer = model_dict[next_layer]
                    layer_list.extend([next_layer])
                else:
                    break

            return(layer_list)


        def layer_list_to_model(model_to_inspect,
                              layer_list):
            remade_model =  keras.Sequential()
            for nn_layer in layer_list:
                remade_model.add(model_to_inspect.get_layer(nn_layer))

            return(remade_model)    

        # work from model -> connections -> sequential list -> new model
        model_dict = get_connection_dict_from_linear_dnn(
            model_to_inspect = model_to_inspect)

        layer_list = connection_dict_to_layer_list(
            model_dict = model_dict,
            start_layer = start_layer,
            stop_layer = stop_layer)

        remade_model = layer_list_to_model(
            model_to_inspect = model_to_inspect,
            layer_list = layer_list)

        return(remade_model)
    
    
    def remake_sequential_model(self, start_layer, stop_layer):
        if self.model == None:
            self.load_model()
        self.model = self.remove_unused_inputs_for_salience(                                                            
            model_to_inspect = self.model,
            start_layer = start_layer,
            stop_layer = stop_layer)
        
        
        
    # Methods for Sanity checks ====================================================================
    def run_regression_test(self, original_xs):
        print("Regression Test only valid for remade sequential models!")
        # Regression Test
        model_yHat = self.model.predict(self.model_xs)

        original_model = tf.keras.models.load_model(self.model_path)
        original_yHat  = original_model.predict(original_xs)

        assert not False in model_yHat == original_yHat, 'Regression test failed!\nLoaded and recreated model are not equivalent!'
        print('Regression test passed.\nLoaded and recreated model are equivalent.')    
        
        
    # Methods for salience =========================================================================   
#     def mk_saliency_map(self, clone = False):
#         def score_function(output):
#             return output[:, 0]
        
#         saliency = Saliency(self.model, clone=False)
#         self.saliency_map = saliency(score_function, (self.model_xs), keepdims=True)
    
#     def get_salience_df(self):
#         temp = self.saliency_map.numpy()
#         temp = pd.DataFrame(temp)
#         return(temp)
        
    def get_salience_all_obs(self, clone = False):
        def score_function(output):
            return output[:, 0]
        
        saliency = Saliency(self.model, clone=clone)
        saliency_map = saliency(score_function, (self.model_xs), keepdims=True)
        temp = saliency_map.numpy()
        self.saliency_map = pd.DataFrame(temp)
        return(temp)        

    def get_salience_select_obs(self, obs_for_saliency, clone = False, use_tqdm = True):
        def score_function(output):
                return output[:, 0]

        sal_map_accumulator = [None for i in range(len(self.model_xs))] # set up a list to accumulate the numpy arrays

        # make this one at a time
        saliency = Saliency(self.model, clone=clone)
        if use_tqdm == True:
             for ob_for_saliency in tqdm.tqdm(obs_for_saliency):
                # Make a reduced dataset
                temp_model_xs = [ith_model_xs[ob_for_saliency] for ith_model_xs in self.model_xs]
                saliency_map = saliency(score_function, (temp_model_xs), keepdims=True) 

                for i in range(len(sal_map_accumulator)):
                    if type(sal_map_accumulator[i]) == type(None):
                        sal_map_accumulator[i] = saliency_map[i].numpy()
                    else:
                        sal_map_accumulator[i] = np.concatenate([sal_map_accumulator[i], saliency_map[i].numpy()]) 
                    
                    
        else:
            for ob_for_saliency in obs_for_saliency:
                # Make a reduced dataset
                temp_model_xs = [ith_model_xs[ob_for_saliency] for ith_model_xs in self.model_xs]
                saliency_map = saliency(score_function, (temp_model_xs), keepdims=True) 

                for i in range(len(sal_map_accumulator)):
                    if type(sal_map_accumulator[i]) == type(None):
                        sal_map_accumulator[i] = saliency_map[i].numpy()
                    else:
                        sal_map_accumulator[i] = np.concatenate([sal_map_accumulator[i], saliency_map[i].numpy()]) 
            
            
        self.saliency_map = sal_map_accumulator
        
        
    def get_salience_percent_obs_above_90(self):
        temp = self.get_salience_df()
        nObs = temp.shape[0]
        # I think a reasonable way of processing this data is to bin the data and look at the number of observations in each 0.1 bin
        decile_counts = pd.DataFrame(np.zeros(shape = temp.shape))

        for i in range(temp.shape[1]):
            decile_counts[i] = pd.cut(temp[i], 
                                      bins = [0.1+(i/10) for i in range(10)] # manually specify bins so we end up with the same cuts
                                     )

        decile_counts = decile_counts.melt()

        decile_counts = decile_counts.reset_index().groupby(['variable', 'value']).count()
        

        decile_counts= decile_counts.reset_index()#.pivot_table(index = 'value', columns = 'variable', values = 'index')
        


        decile_counts['TopDecile'] = [True if 0.95 in e else False for e in list(decile_counts['value'])]
        
        # if there are features with no observations that have a salience at 0.9 then they won't be in the final df. 
        # we can prevent this omission by adding back 0%s for each not in the df. 
        uniq_vars = list(decile_counts['variable'].drop_duplicates())
        decile_counts = decile_counts.loc[decile_counts['TopDecile']]
        


        decile_counts['PercentAbove0.9'] = decile_counts['index']/nObs
        decile_counts = decile_counts.loc[:, ['variable', 'PercentAbove0.9']].rename(columns = {'variable':'Variable'})
        
        # add back in any features that had no observations at the salience cutoff
        add_back_in = [uniq_var for uniq_var in uniq_vars if uniq_var not in list(decile_counts['Variable'].drop_duplicates())]        
        add_back_in_df = pd.DataFrame({'Variable':add_back_in, 
                                       'PercentAbove0.9': [0 for e in add_back_in]})
        decile_counts = pd.concat([decile_counts,add_back_in_df])
        decile_counts = decile_counts.sort_values('Variable')
        
        return(decile_counts)


# %% code_folding=[4, 10]
# ith_tensor = 1

# This just wraps the setup and calculation of the percent of observations with salience > 0.9 for each column/variable
# This allows for a list comprehension to get what information we need.  
def wrapper__percent_salience_above_09(
    model_path,  # = '../data/atlas/models/3_finalize_model_syr__rep_S/eval/hps_rep_2.h5',
    model_xs,    # = x_train[ith_tensor],
    start_layer, # = 'sIn', 
    stop_layer,   # = 'yHat'
    run_regression_test = False
):
    
    demo = dnn_model(model_path = model_path,
                     model_xs = model_xs)
    demo.load_model()
    demo.remake_sequential_model(start_layer = start_layer, stop_layer = stop_layer)
    if run_regression_test:
        demo.run_regression_test(x_train)
    demo.mk_saliency_map()
    percent_salience_above_09 = demo.get_salience_percent_obs_above_90()
    return(percent_salience_above_09)


# Example use
# percent_salience_above_09 = [
#     wrapper__percent_salience_above_09(
#     model_path = '../data/atlas/models/3_finalize_model_syr__rep_S/eval/hps_rep_'+str(i)+'.h5',
#     model_xs = x_train[ith_tensor],
#     start_layer = 'sIn', 
#     stop_layer = 'yHat') for i in range(10)
# ]


# %% code_folding=[]
print("""
# test case: multiple tensors in, multiple out

demo = dnn_model(model_path = '../data/atlas/models/3_finalize_model_syr__rep_full/eval/hps_rep_0.h5',
                     model_xs = [x_all[0], 
                                 x_all[1], 
                                 x_all[2]])

demo.load_model()
demo.get_salience_select_obs(obs_for_saliency = [0,1,100, 101])
demo.saliency_map[0]

# test case: one tensor in, one out
i=0
model_path = '../data/atlas/models/3_finalize_model_syr__rep_S/eval/hps_rep_'+str(i)+'.h5'
model_xs = [x_all[1]] # if we always pass
start_layer = 'sIn'
stop_layer = 'yHat'

demo = dnn_model(model_path = model_path,
                 model_xs = model_xs)
demo.load_model()
demo.remake_sequential_model(start_layer = start_layer, stop_layer = stop_layer)

# demo.run_regression_test(x_train)
demo.get_salience_select_obs(obs_for_saliency = [0,1,100, 101])
demo.saliency_map
""")

# %% [markdown]
# ## Setup minimum indices to look at.
#
#

# %%
# We want to use the fewest observations possible for calculating the salience. (initially to not max out memory, but also for speed.)
# To do this, we'll find the unique indices and select based on those.

# BEFORE excluding no genome entries:
# G --- 3847  - 'Pedigree'
# S --- 41*   -             'ExperimentCode' (should also include year. May be off because of imputation)
# W --- 163   -             'ExperimentCode', 'Year', 'DatePlanted'
# GSW - 59880 - 'Pedigree', 'ExperimentCode', 'Year', 'DatePlanted'


# %% code_folding=[]
## Genome ======================================================================
dedupe_idxs = list(
    phenotype.loc[:, ['Pedigree']
                 ].drop_duplicates().reset_index()['index'])

# do any on this list lack genotypes? -- yes
np.isnan(x_all[0][[dedupe_idxs], :]).sum()
# drop the indices with nans.
okay_idxs = [dedupe_idx for dedupe_idx in dedupe_idxs if np.isnan(x_all[0][[dedupe_idx], :]).sum() == 0]
dedupe_idxs = okay_idxs
dedupe_idxs_g = dedupe_idxs 

## Soil ========================================================================
dedupe_idxs = list(
    phenotype.loc[:, ['ExperimentCode', 'Year']
                 ].drop_duplicates().reset_index()['index'])

dedupe_idxs_s = dedupe_idxs 

## Weather =====================================================================
dedupe_idxs = list(
    phenotype.loc[:, ['ExperimentCode', 'Year', 'DatePlanted']
                 ].drop_duplicates().reset_index()['index'])

dedupe_idxs_w = dedupe_idxs 

## Full/Cat ====================================================================
dedupe_idxs = list(
    phenotype.loc[:, ['Pedigree', 'ExperimentCode', 'Year', 'DatePlanted']
                 ].drop_duplicates().reset_index()['index'])



# do any on this list lack genotypes? -- yes
np.isnan(x_all[0][[dedupe_idxs], :]).sum()
# drop the indices with nans.
okay_idxs = [dedupe_idx for dedupe_idx in dedupe_idxs if np.isnan(x_all[0][[dedupe_idx], :]).sum() == 0]
dedupe_idxs = okay_idxs
dedupe_idxs_full = dedupe_idxs 

# are there any values in one of the sub-lists that are not in the full list?
# g is a subset of full (flag not ins)
assert not True in [True if e not in dedupe_idxs_full else False for e in dedupe_idxs_g]
# s is a subset of full
assert not True in [True if e not in dedupe_idxs_full else False for e in dedupe_idxs_s]
# w is a subset of full
assert not True in [True if e not in dedupe_idxs_full else False for e in dedupe_idxs_w]

# %% [markdown]
# ## Confirm result folder setup, wrap above methods

# %%
# Because they're so resource intensive to generate we'll store the saliency maps.
sal_map_storage_path = "../data/result_intermediates/"
if not os.path.exists(sal_map_storage_path):
    os.mkdir(sal_map_storage_path)


# %% code_folding=[]
# wrapper for full, cat

def make_saliency_map_wrapper_3_tensors(
    current_model_path = '../data/atlas/models/3_finalize_model_syr__rep_full/eval/hps_rep_0.h5',
    current_model_xs = [x_all[0], x_all[1], x_all[2]],
    subset_xs_idxs = dedupe_idxs_full,
    sal_map_storage_path = sal_map_storage_path,
    force_remake_saliency_maps = False # Because these can take a ton of time to produce, 
    # the prefered way to regenerate them is to delete the files. Despite this, an override is supplied.
):
    # get from provided name. 
    # e.g. '../data/atlas/models/3_finalize_model_syr__rep_full/eval/hps_rep_0.h5'
    #                           |_________________________     |    |_______     |
    #                                                     |full|            |  __|
    #                                                                       |0|
    #                                                model_type       model_rep

    model_type = current_model_path.split('/')[-3].split('_')[-1]
    model_rep = current_model_path.split('/')[-1].split('_')[-1].split('.')[0]

    # expected sal. map paths:
    #               e.g. '../data/result_intermediates/full_rep_0_xs_0.pkl',
    expected_res_paths = [sal_map_storage_path+model_type+'_rep_'+str(model_rep)+'_xs_'+str(ith_xs)+'.pkl' 
                          for ith_xs in range(len(current_model_xs))]

    # do the saliency maps need to be remade?
    if ((False in [os.path.exists(e) for e in expected_res_paths]) | force_remake_saliency_maps):

        demo = dnn_model(model_path = current_model_path,
                         model_xs = current_model_xs)

        demo.load_model()
        demo.get_salience_select_obs(obs_for_saliency = subset_xs_idxs, use_tqdm = True)

        # save out all the generated maps
        for ii in range(len(demo.saliency_map)):
            pkl.dump(demo.saliency_map[ii], open(expected_res_paths[ii], 'wb'))


# %% code_folding=[]
# wrapper for G, S, W. Requires rebuilding model to accept only one tensor
def make_saliency_map_wrapper_1_tensor(
    current_model_path = '../data/atlas/models/3_finalize_model_syr__rep_G/eval/hps_rep_0.h5',
    current_model_xs = [x_all[0]],
    start_layer = 'gIn',
    stop_layer = 'yHat',
    subset_xs_idxs = dedupe_idxs_full,
    sal_map_storage_path = sal_map_storage_path,
    force_remake_saliency_maps = False
):

    # get from provided name. 
    model_type = current_model_path.split('/')[-3].split('_')[-1]
    model_rep = current_model_path.split('/')[-1].split('_')[-1].split('.')[0]

    # expected sal. map paths:
    expected_res_paths = [sal_map_storage_path+model_type+'_rep_'+str(model_rep)+'_xs_'+str(ith_xs)+'.pkl' 
                          for ith_xs in range(len(current_model_xs))]

    # do the saliency maps need to be remade?
    if ((False in [os.path.exists(e) for e in expected_res_paths]) | force_remake_saliency_maps):

        demo = dnn_model(model_path = current_model_path,
                         model_xs = current_model_xs 
                        )

        demo.load_model()
        demo.remake_sequential_model(start_layer = start_layer, stop_layer = stop_layer)
        demo.get_salience_select_obs(obs_for_saliency = subset_xs_idxs, use_tqdm = True)


        # save out all the generated maps
        for ii in range(len(demo.saliency_map)):
            pkl.dump(demo.saliency_map[ii], open(expected_res_paths[ii], 'wb'))

# %% [markdown]
# ## Create Saliency maps

# %% [markdown]
# ### `full` model

# %%
for rep_num in range(10):    
    make_saliency_map_wrapper_3_tensors(
        current_model_path = '../data/atlas/models/3_finalize_model_syr__rep_full/eval/hps_rep_'+str(rep_num)+'.h5',
        current_model_xs = [x_all[0], x_all[1], x_all[2]],
        subset_xs_idxs = dedupe_idxs_full,
        sal_map_storage_path = sal_map_storage_path,
        force_remake_saliency_maps = False
    )

# %% [markdown]
# ### `cat` model

# %%
for rep_num in range(10):    
    make_saliency_map_wrapper_3_tensors(
        current_model_path = '../data/atlas/models/3_finalize_model_syr__rep_cat/eval/hps_rep_'+str(rep_num)+'.h5',
        current_model_xs = [x_all[0], x_all[1], x_all[2]],
        subset_xs_idxs = dedupe_idxs_full,
        sal_map_storage_path = sal_map_storage_path,
        force_remake_saliency_maps = False
    )

# %% [markdown]
# ### `G` model

# %%
for rep_num in range(10):
    make_saliency_map_wrapper_1_tensor(
        current_model_path = '../data/atlas/models/3_finalize_model_syr__rep_G/eval/hps_rep_'+str(rep_num)+'.h5',
        current_model_xs = [x_all[0]],
        start_layer = 'gIn',
        stop_layer = 'yHat',
        subset_xs_idxs = dedupe_idxs_full,
        sal_map_storage_path = sal_map_storage_path,
        force_remake_saliency_maps = False
    )

# %% [markdown]
# ### `S` model

# %%
for rep_num in range(10):
    make_saliency_map_wrapper_1_tensor(
        current_model_path = '../data/atlas/models/3_finalize_model_syr__rep_S/eval/hps_rep_'+str(rep_num)+'.h5',
        current_model_xs = [x_all[1]],
        start_layer = 'sIn',
        stop_layer = 'yHat',
        subset_xs_idxs = dedupe_idxs_full,
        sal_map_storage_path = sal_map_storage_path,
        force_remake_saliency_maps = False
    )

# %% [markdown]
# ### `W` model

# %%
for rep_num in range(10):
    make_saliency_map_wrapper_1_tensor(
        current_model_path = '../data/atlas/models/3_finalize_model_syr__rep_W/eval/hps_rep_'+str(rep_num)+'.h5',
        current_model_xs = [x_all[2]],
        start_layer = 'wIn',
        stop_layer = 'yHat',
        subset_xs_idxs = dedupe_idxs_full,
        sal_map_storage_path = sal_map_storage_path,
        force_remake_saliency_maps = False
    )


# %% [markdown]
# ## Contrast Saliency Maps -- Between Networks

# %% code_folding=[30]
def mk_genome_mean_sal_map(in_map,# = alt_map,
                            in_title = 'Salience of Genomic Data',
                            diffs = False):
    in_map = pd.DataFrame(in_map.mean(axis = 0).reshape((1, 1725))
                         ).melt().rename(columns = {'variable':'Factor', 'value':'Salience'})

    if diffs:
        fig = go.Figure(data=go.Heatmap(
            z=in_map.Salience,
            x=['' for e in range(len(in_map.Salience))],
            y=in_map.Factor,
            colorscale='RdBu',
            zmid=0
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=in_map.Salience,
            x=['' for e in range(len(in_map.Salience))],
            y=in_map.Factor,
            colorscale='Purples'
        ))

    fig.update_layout(
        title=in_title
    )
    fig.show()
    
    
    
    
def mk_genome_mean_sal_lines(in_map,# = alt_map,
                            in_title = 'Salience of Genomic Data',
                            diffs = False):
    in_map = pd.DataFrame(in_map.mean(axis = 0).reshape((1, 1725))
                         ).melt().rename(columns = {'variable':'Factor', 'value':'Salience'})

    if diffs:
        fig = go.Figure(data=go.Scatter(
            x=in_map.Factor,
            y=in_map.Salience,
#             colorscale='RdBu',
#             zmid=0
        ))
    else:
        fig = go.Figure(data=go.Scatter(
            x=in_map.Factor,
            y=in_map.Salience,
#             colorscale='Purples'
        ))

    fig.update_layout(
        title=in_title
    )
    fig.show()
    
    
    
## model saliencences map Save for R ========================================
def mk_and_write_genome_mean_sal_map(in_map,# = alt_map,
                                      write_path = '../notebooks/to_r.csv'):
    in_map = pd.DataFrame(in_map.mean(axis = 0).reshape((1, 1725))
                         ).melt().rename(columns = {'variable':'Factor', 'value':'Salience'})

    in_map.to_csv(write_path)


# %%
def mk_soil_mean_sal_map(in_map,# = alt_map,
                            in_title = 'Salience of Soil Data',
                            diffs = False):
    in_map = pd.DataFrame(in_map.mean(axis = 0).reshape((1, 21))
                         ).melt().rename(columns = {'variable':'Factor', 'value':'Salience'})

    s_name_lookup = [e for e in list(soil) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]

    for i in range(len(s_name_lookup)):
        in_map.loc[in_map['Factor'] == i, 'Factor'] = s_name_lookup[i]



    if diffs:
        fig = go.Figure(data=go.Heatmap(
            z=in_map.Salience,
            x=['' for e in range(len(in_map.Salience))],
            y=in_map.Factor,
            colorscale='RdBu',
            zmid=0
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=in_map.Salience,
            x=['' for e in range(len(in_map.Salience))],
            y=in_map.Factor,
            colorscale='Purples'
        ))

    fig.update_layout(
        title=in_title
    )
    fig.show()
    
    
    
## model saliencences map Save for R ========================================
def mk_and_write_soil_mean_sal_map(in_map,# = alt_map,
                                      write_path = '../notebooks/to_r.csv'):
    in_map = pd.DataFrame(in_map.mean(axis = 0).reshape((1, 21))
                         ).melt().rename(columns = {'variable':'Factor', 'value':'Salience'})

    s_name_lookup = [e for e in list(soil) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]

    for i in range(len(s_name_lookup)):
        in_map.loc[in_map['Factor'] == i, 'Factor'] = s_name_lookup[i]

    in_map.to_csv(write_path)


# %%
## model saliencences map ===================================================
def mk_weather_mean_sal_map(in_map,# = alt_map,
                            in_title = 'Salience of Weather and Management Data',
                            diffs = False):


    daily_means = [pd.DataFrame(in_map[:, ith_day, :].mean(axis = 0).reshape((1,19))).melt().rename(columns = {'value':ith_day}) for ith_day in range(288)]
    # daily_means = [daily_means[ith_day].assign(Day=ith_day) for ith_day in range(288)]

    daily_means = [daily_means[ith_day] if ith_day == 0 else daily_means[ith_day].drop(columns = 'variable') for ith_day in range(288)]

    daily_means = pd.concat(
        daily_means, axis = 1
             ).rename(columns = {'variable':'Factor'}
             ).melt(id_vars = 'Factor'
             ).rename(columns = {'variable':'Day',
                                'value':'Salience'})

    daily_means['Day'] = daily_means['Day']-76 # set 0 as the planting date

    w_name_lookup = [e for e in list(weather) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]
    for i in range(len(w_name_lookup)):
        daily_means.loc[daily_means['Factor'] == i, 'Factor'] = w_name_lookup[i]
        
        
    if diffs:
        fig = go.Figure(data=go.Heatmap(
            z=daily_means.Salience,
            x=daily_means.Day,
            y=daily_means.Factor,
            colorscale='RdBu',
            zmid=0,
            zmax = 0.2,
            zmin = -0.2
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=daily_means.Salience,
            x=daily_means.Day,
            y=daily_means.Factor,
            colorscale='Purples'
        ))

    fig.update_layout(
        title=in_title
    )
    fig.show()
    
    
    
## model saliencences map Save for R ========================================
def mk_and_write_weather_mean_sal_map(in_map,# = alt_map,
                                      write_path = '../notebooks/to_r.csv'):
    daily_means = [pd.DataFrame(in_map[:, ith_day, :].mean(axis = 0).reshape((1,19))).melt().rename(columns = {'value':ith_day}) for ith_day in range(288)]
    daily_means = [daily_means[ith_day] if ith_day == 0 else daily_means[ith_day].drop(columns = 'variable') for ith_day in range(288)]

    daily_means = pd.concat(
        daily_means, axis = 1
             ).rename(columns = {'variable':'Factor'}
             ).melt(id_vars = 'Factor'
             ).rename(columns = {'variable':'Day',
                                'value':'Salience'})

    daily_means['Day'] = daily_means['Day']-76 # set 0 as the planting date

    w_name_lookup = [e for e in list(weather) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]
    for i in range(len(w_name_lookup)):
        daily_means.loc[daily_means['Factor'] == i, 'Factor'] = w_name_lookup[i]

    daily_means.to_csv(write_path)


# %%

# %% [markdown]
# #### Genome

# %%
cat_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_0.pkl', 'rb'))
alt_map = pkl.load(open(sal_map_storage_path+'G_rep_0_xs_0.pkl', 'rb'))
sub_map = cat_map - alt_map

# %%
# genome soil weather
mk_and_write_genome_mean_sal_map(in_map = cat_map, write_path = '../notebooks/r_salience_genome_catmodel.csv')
mk_and_write_genome_mean_sal_map(in_map = alt_map, write_path = '../notebooks/r_salience_genome_submodel.csv')

# %%

# %%

 # %%
 mk_genome_mean_sal_map(in_map = sub_map,
                      in_title = 'Difference between Saliences of Genome Data',
                      diffs = True)

# %%
mk_genome_mean_sal_lines(in_map = sub_map,
                      in_title = 'Salience of Genome Data',
                      diffs = True)

# %%
temp = pd.DataFrame({
    'xbar': np.mean(cat_map, axis = 0)
#     'name':[e for e in list(soil) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]
             }).sort_values('xbar', ascending = False).reset_index()

temp
# px.scatter(temp, x = 'index', y = 'xbar')

 # %%
 mk_genome_mean_sal_map(in_map = cat_map,
                      in_title = 'Consecutively Selected Model',
                      diffs = False)

 # %%
 mk_genome_mean_sal_lines(in_map = cat_map,
                      in_title = 'Consecutively Selected Model',
                      diffs = False)

 # %%
 mk_genome_mean_sal_map(in_map = alt_map,
                      in_title = 'Sub-Model',
                      diffs = False)

# %%
mk_genome_mean_sal_lines(in_map = alt_map,
                      in_title = 'Salience of Genome Data',
                      diffs = False)

# %% [markdown]
# #### Soil

# %%
cat_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_1.pkl', 'rb'))
alt_map = pkl.load(open(sal_map_storage_path+'S_rep_0_xs_0.pkl', 'rb'))
sub_map = cat_map - alt_map

# %%
# genome soil weather
mk_and_write_soil_mean_sal_map(in_map = cat_map, write_path = '../notebooks/r_salience_soil_catmodel.csv')
mk_and_write_soil_mean_sal_map(in_map = alt_map, write_path = '../notebooks/r_salience_soil_submodel.csv')

# %%

 # %%
 mk_soil_mean_sal_map(in_map = sub_map,
                      in_title = 'Difference between Saliences of Soil Data',
                      diffs = True)

# %%
pd.DataFrame({
    'xbar': np.mean(cat_map, axis = 0),
    'name':[e for e in list(soil) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]
             }).sort_values('xbar', ascending = False)

 # %%
 mk_soil_mean_sal_map(in_map = cat_map,
                      in_title = 'Consecutively Selected Model',
                      diffs = False)

 # %%
 mk_soil_mean_sal_map(in_map = alt_map,
                      in_title = 'Sub-Model',
                      diffs = False)

# %%

# %%

# %%

# %%

# %% [markdown]
# #### Weather

# %%
cat_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_2.pkl', 'rb'))
alt_map = pkl.load(open(sal_map_storage_path+'W_rep_0_xs_0.pkl', 'rb'))
sub_map = cat_map - alt_map


# %%
# genome soil weather
mk_and_write_weather_mean_sal_map(in_map = cat_map, write_path = '../notebooks/r_salience_weather_catmodel.csv')
mk_and_write_weather_mean_sal_map(in_map = alt_map, write_path = '../notebooks/r_salience_weather_submodel.csv')

# %%

# %%
(np.min(sub_map), np.max(sub_map))



# %%
mk_weather_mean_sal_map(in_map = sub_map,
                            in_title = 'Difference between Saliences of Weather and Management Data',
                            diffs = True)

# %%
mk_weather_mean_sal_map(in_map = cat_map,
                       in_title = 'Consecutively Selected Model')

# %%

# %%



# %%
pd.DataFrame({"xbar" : np.max(np.mean(cat_map, axis = 0), axis = 0),
             'name':[e for e in list(weather) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]}
            ).sort_values("xbar", ascending = False)

# %%
mst_important_factor = pd.DataFrame({"xbar" : np.max(np.mean(cat_map, axis = 0), axis = 0),
             'name':[e for e in list(weather) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]}
            )


px.line(mst_important_factor, x = 'xbar', y = 'name')

# %%
mst_important_day = pd.DataFrame({"xbar" : np.max(np.mean(cat_map, axis = 0), axis = 1),
              "day" : [i-76 for i in range(288)]
#              'name':[e for e in list(weather) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]
             }
            )

px.line(mst_important_day, x = 'day', y = 'xbar')

mst_important_day.sort_values("xbar", ascending = False).reset_index()[0:15                                                                      ]

# %%
mk_weather_mean_sal_map(in_map = alt_map,
                       in_title = 'Sub-Model')

# %%

# %%
stop here

# %%

# %% [markdown]
# ### Genome

# %%
ref_map = pkl.load(open(sal_map_storage_path+'G_rep_0_xs_0.pkl', 'rb'))

cat_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_0.pkl', 'rb'))
sub_map = ref_map - cat_map
# px.imshow(sub_map)

# %%

# %%
px.imshow(sub_map[0:1000, :])

# %%
px.histogram(
    pd.DataFrame(sub_map[:, 862].reshape((1,52252))).melt(),
    x = 'value',
    template = 'plotly_white' 
)

# %%
px.scatter(
    pd.DataFrame(sub_map.mean(axis = 0).reshape((1,1725))).melt(),
    x = 'variable',
    y = 'value',
    template = 'plotly_white',
    marginal_y="histogram"    
)

# %%

# %%

# %%

# %%
temp = pd.DataFrame(sub_map.mean(axis = 0).reshape((1,1725))).melt()

qq = stats.probplot(temp['value'], dist='norm', sparams=(1))
x = np.array([qq[0][0][0], qq[0][0][-1]])

fig = go.Figure()
fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
fig.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines')
fig.layout.update(showlegend=False)
fig.show()


# %%

# %%

# %%
# TODO how variable is this?

# %%

# %%

# %% [markdown]
# ### Soil

# %%
ref_map = pkl.load(open(sal_map_storage_path+'S_rep_0_xs_0.pkl', 'rb'))

cat_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_1.pkl', 'rb'))
sub_map = ref_map - cat_map
px.imshow(sub_map)

# %%

# %%
temp = pd.DataFrame(sub_map.mean(axis = 0).reshape((1,21))).melt()

qq = stats.probplot(temp['value'], dist='norm', sparams=(1))
x = np.array([qq[0][0][0], qq[0][0][-1]])

fig = go.Figure()
fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
fig.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines')
fig.layout.update(showlegend=False)
fig.show()

temp.loc[(temp['value'] < -0.015), ]

# %%
# What is variable 15 (16th)?




# %% [markdown]
# ### Weather

# %%
ref_map = pkl.load(open(sal_map_storage_path+'W_rep_0_xs_0.pkl', 'rb'))

cat_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_2.pkl', 'rb'))
sub_map = ref_map - cat_map
# px.imshow(sub_map)

# %%
px.imshow(sub_map[:, 76, :]) # one time point

# %%
daily_means = [pd.DataFrame(sub_map[:, ith_day, :].mean(axis = 0).reshape((1,19))).melt().rename(columns = {'value':ith_day}) for ith_day in range(288)]
# daily_means = [daily_means[ith_day].assign(Day=ith_day) for ith_day in range(288)]

daily_means = [daily_means[ith_day] if ith_day == 0 else daily_means[ith_day].drop(columns = 'variable') for ith_day in range(288)]

daily_means = pd.concat(
    daily_means, axis = 1
         ).rename(columns = {'variable':'Factor'}
         ).melt(id_vars = 'Factor'
         ).rename(columns = {'variable':'Day',
                            'value':'Salience'})

daily_means['Day'] = daily_means['Day']-76 # set 0 as the planting date

w_name_lookup = [e for e in list(weather) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]
for i in range(len(w_name_lookup)):
    daily_means.loc[daily_means['Factor'] == i, 'Factor'] = w_name_lookup[i]

# %%
fig = go.Figure(data=go.Heatmap(
        z=daily_means.Salience,
        x=daily_means.Day,
        y=daily_means.Factor,
#         colorscale='Viridis'
        colorscale='RdBu',
        zmid=0
))

fig.update_layout(
#     title='GitHub commits per day',
#     xaxis_nticks=36
    
)

fig.show()

# %%

# %%

# %%

# %%

# %%

# %%
# This is useful, but it would be good to see the best model's saliences too. 

# %%

# %%
daily_means = [pd.DataFrame(cat_map[:, ith_day, :].mean(axis = 0).reshape((1,19))).melt().rename(columns = {'value':ith_day}) for ith_day in range(288)]
# daily_means = [daily_means[ith_day].assign(Day=ith_day) for ith_day in range(288)]

daily_means = [daily_means[ith_day] if ith_day == 0 else daily_means[ith_day].drop(columns = 'variable') for ith_day in range(288)]

daily_means = pd.concat(
    daily_means, axis = 1
         ).rename(columns = {'variable':'Factor'}
         ).melt(id_vars = 'Factor'
         ).rename(columns = {'variable':'Day',
                            'value':'Salience'})

daily_means['Day'] = daily_means['Day']-76 # set 0 as the planting date

w_name_lookup = [e for e in list(weather) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]
for i in range(len(w_name_lookup)):
    daily_means.loc[daily_means['Factor'] == i, 'Factor'] = w_name_lookup[i]

# %%
fig = go.Figure(data=go.Heatmap(
        z=daily_means.Salience,
        x=daily_means.Day,
        y=daily_means.Factor,
#         colorscale='Cividis'
        colorscale='Purples'
))

fig.update_layout(
    title='Salience of Weather and Management Data'
)

fig.show()

# %% [markdown]
# ### Cat vs Full

# %%
ith_tensor = 1

cat_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_'+str(ith_tensor)+'.pkl', 'rb'))
full_map = pkl.load(open(sal_map_storage_path+'full_rep_0_xs_'+str(ith_tensor)+'.pkl', 'rb'))
sub_map = cat_map - full_map

if ith_tensor != 2:
    fig = px.imshow(sub_map)
else:
    fig = px.imshow(sub_map[:, 76, :]) # one time point
fig

# %% [markdown]
# #### Genome

# %%
cat_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_0.pkl', 'rb'))
alt_map = pkl.load(open(sal_map_storage_path+'full_rep_0_xs_0.pkl', 'rb'))
sub_map = cat_map - alt_map


# %%

# %%
def mk_genome_mean_sal_map(in_map = alt_map,
                            in_title = 'Salience of Genomic Data',
                            diffs = False):
    in_map = pd.DataFrame(in_map.mean(axis = 0).reshape((1, 1725))
                         ).melt().rename(columns = {'variable':'Factor', 'value':'Salience'})

    if diffs:
        fig = go.Figure(data=go.Heatmap(
            z=in_map.Salience,
            x=['' for e in range(len(in_map.Salience))],
            y=in_map.Factor,
            colorscale='RdBu',
            zmid=0
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=in_map.Salience,
            x=['' for e in range(len(in_map.Salience))],
            y=in_map.Factor,
            colorscale='Purples'
        ))

    fig.update_layout(
        title=in_title
    )
    fig.show()


# %%
def mk_genome_mean_sal_lines(in_map = alt_map,
                            in_title = 'Salience of Genomic Data',
                            diffs = False):
    in_map = pd.DataFrame(in_map.mean(axis = 0).reshape((1, 1725))
                         ).melt().rename(columns = {'variable':'Factor', 'value':'Salience'})

    if diffs:
        fig = go.Figure(data=go.Scatter(
            x=in_map.Factor,
            y=in_map.Salience,
#             colorscale='RdBu',
#             zmid=0
        ))
    else:
        fig = go.Figure(data=go.Scatter(
            x=in_map.Factor,
            y=in_map.Salience,
#             colorscale='Purples'
        ))

    fig.update_layout(
        title=in_title
    )
    fig.show()


 # %%
 mk_genome_mean_sal_map(in_map = sub_map,
                      in_title = 'Salience of Genome Data',
                      diffs = True)

# %%
mk_genome_mean_sal_lines(in_map = sub_map,
                      in_title = 'Salience of Genome Data',
                      diffs = True)

 # %%
 mk_genome_mean_sal_map(in_map = cat_map,
                      in_title = 'Salience of Genome Data',
                      diffs = False)

 # %%
 mk_genome_mean_sal_lines(in_map = cat_map,
                      in_title = 'Salience of Genome Data',
                      diffs = False)

 # %%
 mk_genome_mean_sal_map(in_map = alt_map,
                      in_title = 'Salience of Genome Data',
                      diffs = False)

# %%
mk_genome_mean_sal_lines(in_map = alt_map,
                      in_title = 'Salience of Genome Data',
                      diffs = False)

# %% [markdown]
# #### Soil

# %%
cat_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_1.pkl', 'rb'))
alt_map = pkl.load(open(sal_map_storage_path+'full_rep_0_xs_1.pkl', 'rb'))
sub_map = cat_map - alt_map


# %%
def mk_soil_mean_sal_map(in_map = alt_map,
                            in_title = 'Salience of Soil Data',
                            diffs = False):
    in_map = pd.DataFrame(in_map.mean(axis = 0).reshape((1, 21))
                         ).melt().rename(columns = {'variable':'Factor', 'value':'Salience'})

    s_name_lookup = [e for e in list(soil) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]

    for i in range(len(s_name_lookup)):
        in_map.loc[in_map['Factor'] == i, 'Factor'] = s_name_lookup[i]



    if diffs:
        fig = go.Figure(data=go.Heatmap(
            z=in_map.Salience,
            x=['' for e in range(len(in_map.Salience))],
            y=in_map.Factor,
            colorscale='RdBu',
            zmid=0
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=in_map.Salience,
            x=['' for e in range(len(in_map.Salience))],
            y=in_map.Factor,
            colorscale='Purples'
        ))

    fig.update_layout(
        title=in_title
    )
    fig.show()

 # %%
 mk_soil_mean_sal_map(in_map = sub_map,
                      in_title = 'Salience of Soil Data',
                      diffs = True)

 # %%
 mk_soil_mean_sal_map(in_map = cat_map,
                      in_title = 'Salience of Soil Data',
                      diffs = False)

 # %%
 mk_soil_mean_sal_map(in_map = alt_map,
                      in_title = 'Salience of Soil Data',
                      diffs = False)

# %%

# %%

# %%

# %%

# %% [markdown]
# #### Weather

# %%
cat_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_2.pkl', 'rb'))
alt_map = pkl.load(open(sal_map_storage_path+'full_rep_0_xs_2.pkl', 'rb'))
sub_map = cat_map - alt_map


# %%
# daily_means = [pd.DataFrame(sub_map[:, ith_day, :].mean(axis = 0).reshape((1,19))).melt().rename(columns = {'value':ith_day}) for ith_day in range(288)]

# daily_means = [daily_means[ith_day] if ith_day == 0 else daily_means[ith_day].drop(columns = 'variable') for ith_day in range(288)]

# daily_means = pd.concat(
#     daily_means, axis = 1
#          ).rename(columns = {'variable':'Factor'}
#          ).melt(id_vars = 'Factor'
#          ).rename(columns = {'variable':'Day',
#                             'value':'Salience'})

# daily_means['Day'] = daily_means['Day']-76 # set 0 as the planting date

# w_name_lookup = [e for e in list(weather) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]
# for i in range(len(w_name_lookup)):
#     daily_means.loc[daily_means['Factor'] == i, 'Factor'] = w_name_lookup[i]

# %%
# fig = go.Figure(data=go.Heatmap(
#         z=daily_means.Salience,
#         x=daily_means.Day,
#         y=daily_means.Factor,
#         colorscale='RdBu',
#         zmid=0
# ))

# fig.update_layout()

# fig.show()

# %%

# %%
## model saliencences map ===================================================
def mk_weather_mean_sal_map(in_map = alt_map,
                            in_title = 'Salience of Weather and Management Data',
                            diffs = False):


    daily_means = [pd.DataFrame(in_map[:, ith_day, :].mean(axis = 0).reshape((1,19))).melt().rename(columns = {'value':ith_day}) for ith_day in range(288)]
    # daily_means = [daily_means[ith_day].assign(Day=ith_day) for ith_day in range(288)]

    daily_means = [daily_means[ith_day] if ith_day == 0 else daily_means[ith_day].drop(columns = 'variable') for ith_day in range(288)]

    daily_means = pd.concat(
        daily_means, axis = 1
             ).rename(columns = {'variable':'Factor'}
             ).melt(id_vars = 'Factor'
             ).rename(columns = {'variable':'Day',
                                'value':'Salience'})

    daily_means['Day'] = daily_means['Day']-76 # set 0 as the planting date

    w_name_lookup = [e for e in list(weather) if e not in ['Unnamed: 0', 'ExperimentCode', 'Year', 'Date']]
    for i in range(len(w_name_lookup)):
        daily_means.loc[daily_means['Factor'] == i, 'Factor'] = w_name_lookup[i]
        
        
    if diffs:
        fig = go.Figure(data=go.Heatmap(
            z=daily_means.Salience,
            x=daily_means.Day,
            y=daily_means.Factor,
            colorscale='RdBu',
            zmid=0
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
            z=daily_means.Salience,
            x=daily_means.Day,
            y=daily_means.Factor,
            colorscale='Purples'
        ))

    fig.update_layout(
        title=in_title
    )
    fig.show()

# %%
mk_weather_mean_sal_map(in_map = sub_map,
                            in_title = 'Salience of Weather and Management Data',
                            diffs = True)

# %%
mk_weather_mean_sal_map(in_map = cat_map)

# %%
mk_weather_mean_sal_map(in_map = alt_map)

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

# %%

# %%

# %%

# %%
# todo -- get constrasts of salience and salience for all obs. 
# os.listdir(sal_map_storage_path)

# %%

# %%

# %% [markdown]
# ## Is one modality responsible for most of the salience?

# %%
cat_g_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_0.pkl', 'rb'))
cat_s_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_1.pkl', 'rb'))
cat_w_map = pkl.load(open(sal_map_storage_path+'cat_rep_0_xs_2.pkl', 'rb'))


# %%
cg = pd.DataFrame(cat_g_map.reshape(90134700, 1), columns = ['x'])
cg['type'] = 'g'

# %%
cs = pd.DataFrame(cat_s_map.reshape(1097292, 1), columns = ['x'])
cs['type'] = 's'

# %%
cw = pd.DataFrame(cat_w_map.reshape(285922944, 1), columns = ['x'])
cw['type'] = 'w'


# %%
def four_num_summary(vx):
    df = pd.DataFrame({
        'stat': ['q1', 'mean', 'q2', 'q3'],
        'val':  [np.quantile(vx, .25),
        np.mean(vx), 
        np.quantile(vx, .50), 
        np.quantile(vx, .75)
                
                ]
    })
    return(df)


# %%
# group all the 


# Mg = four_num_summary(vx = cg['x'])
# Mg['Type'] = 'G'

# Ms = four_num_summary(vx = cs['x'])
# Ms['Type'] = 'S'

# Mw = four_num_summary(vx = cw['x'])
# Mw['Type'] = 'W'


# M = pd.concat([
# Mg, Ms, Mw
#     ])
# M


# px.scatter(M, x = 'Type', y = 'val', color = 'stat')

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Contrast Saliency Maps -- Within Networks

# %%
def sal_map_2d_tidy(innp):
    df = pd.DataFrame(innp)
    df['UID'] = df.index
    df = df.melt(id_vars = ['UID'])
    df = df.rename(columns = {'variable':'Factor', 'value':'Salience'})
    return(df)

# %% [markdown]
# ### Genome

# %%

# %%


def disposable_fcn(i):
    res = pkl.load(open(sal_map_storage_path+'G_rep_'+str(i)+'_xs_0.pkl', 'rb'))
    df = sal_map_2d_tidy(res)
    
#     df['Rep'] = str(i)

    df = df.rename(columns = {'Salience':str(i)})
    return(df)

res = [disposable_fcn(i) for i in range(10)]

# %%
# Drop the non-saliance columns for all except the first dataframe. This allows for concatenation without any issues.
res = [res[i] if i == 0 else res[i].drop(columns = ['UID', 'Factor']) for i in range(len(res))]

# %%

# %%
# res = pd.concat(res, axis = 1)

# %%
res[0]

# %%
# ith = 0

# pc_num = [e for e in list(res[ith]) if e not in ['UID', 'Factor']][0]

# pc_num
test = pd.concat(res, axis = 1)
test

# %%
temp

# %%
mask_df = pd.DataFrame(test['UID'].drop_duplicates())

# %%
mask_df['mask'] = 'f'

for i in range(len(dedupe_idxs_g)):
    mask_df.loc[mask_df['UID'] == dedupe_idxs_g[i], 'mask'] = 't'

# %%

# %%
test = mask_df.merge(test, how = 'outer')
test

# %%
test = test.loc[test['mask'] == 't', :]

# %%
test['Factor'].drop_duplicates()

# %%
res = test.drop(columns = ['mask'])

# %%
# def _temp_fcn(i):
#     mask = test['UID'] == dedupe_idxs_g[0]
#     return(test.loc[mask])



# %%
# deduped_list_of_saliences = [_temp_fcn(i) for i in dedupe_idxs_g]

# %%
# res[1]

# %%

# %%

# %%
# <--v--v--v--v--v--v--v--v--v--v-- do for each! --v--v--v--v--v--v--v--v--v--v--> #
# mask = [True if uid in dedupe_idxs_g else False for uid in res['UID']] # <----  Note! have to change for each network type
# much faster to join than to use a list compreshension
    # dedupe_uid = list(res['UID'].drop_duplicates())
    # dedupe_mask = [True if uid in dedupe_idxs_g else False for uid in dedupe_uid]
    # dedupe_mask_df = pd.DataFrame(zip(dedupe_uid,dedupe_mask)).rename(columns = {0:'UID', 1:'mask'})
    # dedupe_mask_df = dedupe_mask_df.merge(res['UID'])
    # mask = dedupe_mask_df['mask']

    # res = res.loc[mask]
res = res.reset_index().drop(columns = 'index')

res['Factor'] = res['Factor'].astype(str)
# <--^--^--^--^--^--^--^--^--^--^-- do for each! --^--^--^--^--^--^--^--^--^--^--> #


# %%
res['Factor'].drop_duplicates()

# %%


# %% [markdown]
# #### Summary Corr( salience )
#

# %%
ith_factor = 0

def correlate_salience_one_factor(res = res,
                                 ith_factor = 0):

    temp_corr = res.loc[res.Factor == ith_factor
                       ].drop(columns = ['UID', 'Factor']
                       ).corr()

    temp_corr = temp_corr.reset_index(
                        ).melt(id_vars = 'index'
                        ).rename(columns = {
           'index':'rep_i', 
        'variable':'rep_j', 
           'value':'corr'})

    # shave to half of corr matrix; drop unity
    temp_corr['rep_i'] = temp_corr['rep_i'].astype(int)
    temp_corr['rep_j'] = temp_corr['rep_j'].astype(int)
    mask = (temp_corr['rep_j'] > temp_corr['rep_i']) 
    temp_corr = temp_corr.loc[mask, ]
    temp_corr['Factor'] = ith_factor

    return(temp_corr)


# %%

# %%

# %%
# res_corr

# %%

# %%
file_path = '../data/result_intermediates/sal_df_g_corr_list.pkl'
if os.path.exists(file_path):
    res_corr = pkl.load(open(file_path, 'rb'))
else:    
    res_corr = []

    for ii in tqdm.tqdm( list(set(res.Factor)) ):
        res_corr.extend([correlate_salience_one_factor(res, ith_factor = ii)])

    res_corr = pd.concat(res_corr, axis = 0)
    res_corr['corr_ij'] = res_corr['rep_i'].astype(str)+'-'+res_corr['rep_j'].astype(str) 
        
    pkl.dump(res_corr, open(file_path, 'wb'))

# %%
# res_corr = [correlate_salience_one_factor(res, ith_factor = ii) for ii in set(res.Factor)]
# res_corr = pd.concat(res_corr, axis = 0)
# res_corr['corr_ij'] = res_corr['rep_i'].astype(str)+'-'+res_corr['rep_j'].astype(str)
# res_corr

# %%
res

# %% code_folding=[]
res_corr_summary = res_corr.copy()

res_corr_summary = res_corr_summary.groupby(['Factor']).agg(
    Mean_Salience =('corr', np.mean),
    Sd_Salience =  ('corr', np.std),
    Min_Salience = ('corr', np.min),
    Max_Salience = ('corr', np.max)
    ).reset_index()
res_corr_summary.head(2)

# %%
# sort for plotting -- plotly uses alphabetical instead of ggplots factor default.
res_corr = res_corr.sort_values('Factor')


# %%

# %%
res_corr_summary['Factor'] = res_corr_summary['Factor'].astype(int)
res_corr_summary = res_corr_summary.sort_values('Factor')



# %%

fig = px.line(res_corr_summary, x = 'Factor', y = 'Mean_Salience', title = 'Genome: Correlations of Salience', template = 'plotly_white')

# I don't like that this isn't vectorized but it works for adding ~1k lines.
# for idx in res_corr.index:
#     fig.add_trace(
#         go.Scatter(x = res_corr.loc[idx, 'Factor'],
#                    y = res_corr.loc[idx, 'corr'],
#                    mode = 'lines',
#                    line=dict(width=0.5),
#                    marker=dict(color='rgba(68, 68, 68, 0.03)'),
#                    showlegend=False
#                    ))

# add band
fig.add_trace(
    go.Scatter(x = res_corr_summary['Factor'], y = res_corr_summary['Mean_Salience']+res_corr_summary['Sd_Salience'], #y = res_corr_summary['Max_Salience'], 
               mode = 'lines',
               marker=dict(color="rgba(68, 68, 68, 1)"),# "#444"),
               line=dict(width=0.5), #0),
               showlegend=False)
)

fig.add_trace(
    go.Scatter(x = res_corr_summary['Factor'], y = res_corr_summary['Mean_Salience']-res_corr_summary['Sd_Salience'], #y = res_corr_summary['Min_Salience'], 
               mode = 'lines',
               marker=dict(color="rgba(68, 68, 68, 1)"),# "#444"),
               line=dict(width=0.5), #0),
               fillcolor='rgba(68, 68, 68, 0.3)',
#                fill='tonexty',
               showlegend=False)
)


# add points for mean

# fig.add_trace(
#     go.Scatter(x = res_corr_summary['Factor'], 
#                y = res_corr_summary['Mean_Salience'],
#                mode = 'markers',
#                marker=dict(color="rgba(256, 256, 256, 1)"),
#                showlegend=False
#     )
# )

fig.update_layout(yaxis_range=[-1,1])

fig

# %% [markdown]
# #### Summary salience
#

# %%
# mask = [True if uid in dedupe_idxs_s else False for uid in res['UID']] # <----  Note! have to change for each network type

res_sal = res.melt(id_vars = ['UID', 'Factor']).rename(columns = {'variable':'Rep', 'value':'Salience'})
res_sal = res_sal.reset_index().drop(columns = 'index')

res_sal_summary = res_sal.groupby(['Factor']).agg(
    Mean_Salience =('Salience', np.mean),
    Sd_Salience =  ('Salience', np.std),
    Min_Salience = ('Salience', np.min),
    Max_Salience = ('Salience', np.max)
    ).reset_index()
res_sal_summary

# %%
# Sort so it plots correctly
res_sal = res_sal.sort_values(['Factor'])
res_sal

# %%
res_sal_summary['Factor'] = res_sal_summary['Factor'].astype(int)

res_sal_summary = res_sal_summary.sort_values(['Factor'])
res_sal_summary

# %%

fig = px.line(res_sal_summary, x = 'Factor', y = 'Mean_Salience', title = 'Genome: Salience', template = 'plotly_white')

# for uid in list(res_sal.UID.drop_duplicates()):
#     temp = res_sal.loc[res_sal.UID == uid]
#     for rep in list(temp.Rep.drop_duplicates()):
#         fig.add_trace(
#             go.Scatter(x = temp.loc[temp.Rep == rep, 'Factor'],
#                        y = temp.loc[temp.Rep == rep, 'Salience'],
#                        mode = 'lines',
#                        line=dict(width=0.5),
#                        marker=dict(color='rgba(68, 68, 68, 0.02)'),
#                        showlegend=False
#                        ))


# add band
fig.add_trace(
    go.Scatter(x = res_sal_summary['Factor'], y = res_sal_summary['Mean_Salience']+res_sal_summary['Sd_Salience'], #y = res_sal_summary['Max_Salience'], 
               mode = 'lines',
               marker=dict(color="#444"),
               line=dict(width=0),
               showlegend=False)
)

fig.add_trace(
    go.Scatter(x = res_sal_summary['Factor'], y = res_sal_summary['Mean_Salience']-res_sal_summary['Sd_Salience'], #y = res_sal_summary['Min_Salience'], 
               mode = 'lines',
               marker=dict(color="#444"),
               line=dict(width=0),
               fillcolor='rgba(68, 68, 68, 0.3)',
               fill='tonexty',
               showlegend=False)
)


# add points for mean

# fig.add_trace(
#     go.Scatter(x = res_sal_summary['Factor'], 
#                y = res_sal_summary['Mean_Salience'],
#                mode = 'markers',
#                marker=dict(color="rgba(256, 256, 256, 1)"),
#                showlegend=False
#     )
# )




fig.update_layout(yaxis_range=[0,1])

fig


# %%

# %%

# %%

# %% [markdown]
# ### Soil

# %%
def sal_map_2d_tidy(innp):
    df = pd.DataFrame(innp)
    df['UID'] = df.index
    df = df.melt(id_vars = ['UID'])
    df = df.rename(columns = {'variable':'Factor', 'value':'Salience'})
    return(df)

def disposable_fcn(i):
    res = pkl.load(open(sal_map_storage_path+'S_rep_'+str(i)+'_xs_0.pkl', 'rb'))
    df = sal_map_2d_tidy(res)
    
#     df['Rep'] = str(i)

    df = df.rename(columns = {'Salience':str(i)})
    return(df)

res = [disposable_fcn(i) for i in range(10)]

# %%
# Drop the non-saliance columns for all except the first dataframe. This allows for concatenation without any issues.
res = [res[i] if i == 0 else res[i].drop(columns = ['UID', 'Factor']) for i in range(len(res))]

# %%
res = pd.concat(res, axis = 1)

# %%
# <--v--v--v--v--v--v--v--v--v--v-- do for each! --v--v--v--v--v--v--v--v--v--v--> #
mask = [True if uid in dedupe_idxs_s else False for uid in res['UID']] # <----  Note! have to change for each network type
res = res.loc[mask]
res = res.reset_index().drop(columns = 'index')



soil_cols = list(soil)
soil_cols = [soil_cols[e] for e in range(len(soil_cols)) if e>=(len(soil_cols)-21)]
res['Factor'] = res['Factor'].astype(str)

for i in range(len(soil_cols)):
#     res = res.rename(columns = {i:soil_cols[i]})
    res.loc[res['Factor'] == str(i), 'Factor'] = soil_cols[i]
# <--^--^--^--^--^--^--^--^--^--^-- do for each! --^--^--^--^--^--^--^--^--^--^--> #


# %% [markdown]
# #### Summary Corr( salience )
#

# %%
ith_factor = 0

def correlate_salience_one_factor(res = res,
                                 ith_factor = 0):

    temp_corr = res.loc[res.Factor == ith_factor
                       ].drop(columns = ['UID', 'Factor']
                       ).corr()

    temp_corr = temp_corr.reset_index(
                        ).melt(id_vars = 'index'
                        ).rename(columns = {
           'index':'rep_i', 
        'variable':'rep_j', 
           'value':'corr'})

    # shave to half of corr matrix; drop unity
    temp_corr['rep_i'] = temp_corr['rep_i'].astype(int)
    temp_corr['rep_j'] = temp_corr['rep_j'].astype(int)
    mask = (temp_corr['rep_j'] > temp_corr['rep_i']) 
    temp_corr = temp_corr.loc[mask, ]
    temp_corr['Factor'] = ith_factor

    return(temp_corr)


# %%
res_corr = [correlate_salience_one_factor(res, ith_factor = ii) for ii in set(res.Factor)]
res_corr = pd.concat(res_corr, axis = 0)
res_corr['corr_ij'] = res_corr['rep_i'].astype(str)+'-'+res_corr['rep_j'].astype(str)
res_corr

# %% code_folding=[]
res_corr_summary = res_corr.copy()

res_corr_summary = res_corr_summary.groupby(['Factor']).agg(
    Mean_Salience =('corr', np.mean),
    Sd_Salience =  ('corr', np.std),
    Min_Salience = ('corr', np.min),
    Max_Salience = ('corr', np.max)
    ).reset_index()
res_corr_summary.head(2)

# %%
# sort for plotting -- plotly uses alphabetical instead of ggplots factor default.
res_corr = res_corr.sort_values('Factor')


# %%

fig = px.line(res_corr_summary, x = 'Factor', y = 'Mean_Salience', title = 'Soil: Correlations of Salience', template = 'plotly_white')

# add band
fig.add_trace(
    go.Scatter(x = res_corr_summary['Factor'], y = res_corr_summary['Mean_Salience']+res_corr_summary['Sd_Salience'], #y = res_corr_summary['Max_Salience'], 
               mode = 'lines',
               marker=dict(color="rgba(68, 68, 68, 1)"),# "#444"),
               line=dict(width=0.5), #0),
               showlegend=False)
)

fig.add_trace(
    go.Scatter(x = res_corr_summary['Factor'], y = res_corr_summary['Mean_Salience']-res_corr_summary['Sd_Salience'], #y = res_corr_summary['Min_Salience'], 
               mode = 'lines',
               marker=dict(color="rgba(68, 68, 68, 1)"),# "#444"),
               line=dict(width=0.5), #0),
               fillcolor='rgba(68, 68, 68, 0.3)',
#                fill='tonexty',
               showlegend=False)
)

fig.update_layout(yaxis_range=[-1,1])

fig

# %%

# %% [markdown]
# #### Summary salience
#

# %%
# mask = [True if uid in dedupe_idxs_s else False for uid in res['UID']] # <----  Note! have to change for each network type

res_sal = res.melt(id_vars = ['UID', 'Factor']).rename(columns = {'variable':'Rep', 'value':'Salience'})
res_sal = res_sal.reset_index().drop(columns = 'index')

res_sal_summary = res_sal.groupby(['Factor']).agg(
    Mean_Salience =('Salience', np.mean),
    Sd_Salience =  ('Salience', np.std),
    Min_Salience = ('Salience', np.min),
    Max_Salience = ('Salience', np.max)
    ).reset_index()
res_sal_summary

# %%
# Sort so it plots correctly
res_sal = res_sal.sort_values(['Factor'])
res_sal

# %%

fig = px.line(res_sal_summary, x = 'Factor', y = 'Mean_Salience', title = 'Soil: Salience', template = 'plotly_white')

# for uid in list(res_sal.UID.drop_duplicates()):
#     temp = res_sal.loc[res_sal.UID == uid]
#     for rep in list(temp.Rep.drop_duplicates()):
#         fig.add_trace(
#             go.Scatter(x = temp.loc[temp.Rep == rep, 'Factor'],
#                        y = temp.loc[temp.Rep == rep, 'Salience'],
#                        mode = 'lines',
#                        line=dict(width=0.5),
#                        marker=dict(color='rgba(68, 68, 68, 0.02)'),
#                        showlegend=False
#                        ))


# add band
fig.add_trace(
    go.Scatter(x = res_sal_summary['Factor'], y = res_sal_summary['Mean_Salience']+res_sal_summary['Sd_Salience'], #y = res_sal_summary['Max_Salience'], 
               mode = 'lines',
               marker=dict(color="#444"),
               line=dict(width=0),
               showlegend=False)
)

fig.add_trace(
    go.Scatter(x = res_sal_summary['Factor'], y = res_sal_summary['Mean_Salience']-res_sal_summary['Sd_Salience'], #y = res_sal_summary['Min_Salience'], 
               mode = 'lines',
               marker=dict(color="#444"),
               line=dict(width=0),
               fillcolor='rgba(68, 68, 68, 0.3)',
               fill='tonexty',
               showlegend=False)
)


# add points for mean

# fig.add_trace(
#     go.Scatter(x = res_sal_summary['Factor'], 
#                y = res_sal_summary['Mean_Salience'],
#                mode = 'markers',
#                marker=dict(color="rgba(256, 256, 256, 1)"),
#                showlegend=False
#     )
# )




fig.update_layout(yaxis_range=[0,1])

fig


# %% [markdown]
# ### Weather

# %%
def disposable_fcn_W(i, dedupe_list = dedupe_idxs_w):
    # i, dedupe_list = dedupe_idxs_w

    # clean up each day's data and add it into the list. We ultimately want tidy data for the plot.
    def _anon_fcn(res, dedupe_list,  day):
        df = sal_map_2d_tidy(innp = res[:, day, :] )
        # mask = [True if uid in dedupe_list else False for uid in df['UID']] # <-  it seems like this should help but the list comprehension greatly slows down the process.
        # df = df.loc[mask]
        df = df.reset_index().drop(columns = 'index')
        df = df.rename(columns = {'Salience':str(day)})
#         df['Day'] = day # <- new! 
        return(df)

    res = pkl.load(open(sal_map_storage_path+'W_rep_'+str(i)+'_xs_0.pkl', 'rb'))
    res_list = [_anon_fcn(res = res, dedupe_list = dedupe_list , day = ith_day) for ith_day in range(res.shape[1])]
    # Drop the non-saliance columns for all except the first dataframe. This allows for concatenation without any issues.
    res_list = [res_list[i] if i == 0 else res_list[i].drop(columns = ['UID', 'Factor']) for i in range(len(res_list))]
    df = pd.concat(res_list, axis = 1)
    df['Rep'] = str(i)
    df = df.copy() # for defragmentation
    return(df)


# %%

# This isn't so long it's too but but will act as a template for later.

file_path = '../data/result_intermediates/sal_df_W_list.pkl'
if os.path.exists(file_path):
    res = pkl.load(open(file_path, 'rb'))
    
else:
    # clean up weather salience data
    # ~ 3 minutes
    res = [disposable_fcn_W(i, dedupe_list = dedupe_idxs_w) for i in range(10)]
    # replace factor ints with names
    weather_cols = list(weather)
    weather_cols = [weather_cols[e] for e in range(len(weather_cols)) if e>=(len(weather_cols)-19)]
    
    for i in range(len(res)):
        res[i]['Factor'] = res[i]['Factor'].astype(str)
        for j in range(len(weather_cols)):
            res[i].loc[res[i]['Factor'] == str(j), 'Factor'] = weather_cols[j]  
    
    pkl.dump(res, open(file_path, 'wb'))


# %%

# %%
# path = "./data_intermediates/cvFoldCenterScaleDict.p"
# if os.path.exists(path):
#     cvFoldCenterScaleDict = pkl.load(open(path, 'rb'))
# else:
#     pkl.dump(cvFoldCenterScaleDict, open(path, 'wb'))




# %%

# %%

# %%

# %% [markdown]
# #### Summary Corr( salience )
#

# %%
def mk_weather_daily_corr(results_list = res,
                    current_day = 0):
    # drop all but current day and grouping var* (* in the first one)
    temp_corr = [results_list[i].loc[:, ['Factor', str(current_day)]] if i == 0 else results_list[i].loc[:, [str(current_day)]] for i in range(len(results_list))]

    # rename day so that entries can be correlated easily
    temp_corr = [temp_corr[i].rename(columns = {str(current_day):str(i)}) for i in range(len(temp_corr))]
    temp_corr = pd.concat(temp_corr, axis = 1)
    temp_corr = temp_corr.groupby('Factor'
                                 ).corr(
                                 ).reset_index(
                                 ).melt(
        id_vars = ['Factor', 'level_1']
                                 ).rename(
        columns = {'level_1':'rep_i', 
                  'variable':'rep_j', 
                     'value':'corr'})

    # shave to half of corr matrix; drop unity
    temp_corr['rep_i'] = temp_corr['rep_i'].astype(int)
    temp_corr['rep_j'] = temp_corr['rep_j'].astype(int)
    mask = (temp_corr['rep_j'] > temp_corr['rep_i']) 
    temp_corr = temp_corr.loc[mask, ]
    temp_corr['Day'] = current_day

    return(temp_corr)


# %%
# make list of values to iterate over
all_days = [e for e in list(res[0]) if e not in ['UID', 'Factor', 'Rep']]

# %%
for ii in tqdm.tqdm(range(2)):
    
    file_path = '../data/result_intermediates/sal_df_W_corr_part_day'+['0-144', '145-287'][ii]+'.pkl'
    
    if os.path.exists(file_path):
        res_corr = pkl.load(open(file_path, 'rb'))

    else:
        # running this as a list comprehension on all ten causes the computer to crash. 
        # We can get around this issue thus:

        weather_daily_corr_list = [mk_weather_daily_corr(
            results_list = res,
            current_day = current_day) for current_day in [
            [int(e) for e in np.linspace(start = 0, stop = 144, num = 144)],
            [int(e) for e in np.linspace(start = 145, stop = 287, num = 144)]
        ][ii]
                                   
                                   #all_days 
                             ]

        weather_corr = pd.concat(weather_daily_corr_list)
        weather_corr['corr_ij'] = weather_corr['rep_i'].astype(str)+'-'+weather_corr['rep_j'].astype(str)
        res_corr = weather_corr   

        pkl.dump(res_corr, open(file_path, 'wb'))

# %%
# stitch together the split parts.
res_corr = [pkl.load(open('../data/result_intermediates/sal_df_W_corr_part_day'+ii+'.pkl', 'rb')) for ii in ['0-144', '145-287']]
res_corr = pd.concat(res_corr)

# %%
# if we can do all at once

file_path = '../data/result_intermediates/sal_df_W_corr_summary.pkl'
if os.path.exists(file_path):
    res_corr_summary = pkl.load(open(file_path, 'rb'))

else:
    res_corr_summary = res_corr
    res_corr_summary = res_corr_summary.groupby(['Factor', 'Day']).agg(
        Mean_Salience =('corr', np.mean),
        Sd_Salience =  ('corr', np.std),
        Min_Salience = ('corr', np.min),
        Max_Salience = ('corr', np.max)
        ).reset_index()
   
    pkl.dump(res_corr_summary, open(file_path, 'wb'))

    
res_corr_summary.head(2) 

# %% code_folding=[]

# %%

res_corr_summary['Day'] = res_corr_summary['Day'].astype(int)
res_corr_summary = res_corr_summary.sort_values(['Day'])



# %%
# res_corr.loc[idx, ]

# %%
current_res_corr_summary = res_corr_summary

# # current_res_corr_summary = current_res_corr_summary.loc[current_res_corr_summary['Factor'] == current_Factor, :]
# current_res_corr_summary['Day'] = current_res_corr_summary['Day'].astype(int)
# current_res_corr_summary = current_res_corr_summary.sort_values('Day')

# fig = px.line(current_res_corr_summary, x = 'Day', y = 'Mean_Salience', color = 'Factor',
#               title = 'Weather: Correlations of Salience', template = 'plotly_white')

# fig.update_layout(yaxis_range=[-1,1])

# fig.show()

# %%
# make up a color/factor dict to use
rng = np.random.default_rng(354384)
w_sal_fct_col = {}

factors = list(current_res_corr_summary.Factor.drop_duplicates())

for fct in factors:
    w_sal_fct_col.update({fct:list(rng.integers(low=0, high=255, size=3))})

# %%

factors = list(current_res_corr_summary.Factor.drop_duplicates())

fig = go.Figure()

for ith_factor in range(len(factors)): 
    def color_from_list(inlist, mode = 'Band'):
        if mode.lower() == 'band':
            return('rgba('+', '.join([str(e) for e in inlist] )+', 0.2)')
        else:
            return('rgba('+', '.join([str(e) for e in inlist] )+', 1)')

    current_factor = factors[ith_factor]
    current_color_list = w_sal_fct_col[current_factor]

    
    mask = current_res_corr_summary['Factor'] == current_factor
    

    

    # add band
    fig.add_trace(
        go.Scatter(x = current_res_corr_summary.loc[mask, 'Day'], 
                   y = current_res_corr_summary.loc[mask, 'Mean_Salience']+current_res_corr_summary.loc[mask, 'Sd_Salience'], 
                   mode = 'lines',
                   marker=dict(color="#444"),
                   line=dict(width=0),
                   showlegend=False,
                  )
    )
    
    if ith_factor == 0:
        fig.add_trace(
            go.Scatter(x = current_res_corr_summary.loc[mask, 'Day'], 
                       y = current_res_corr_summary.loc[mask, 'Mean_Salience']-current_res_corr_summary.loc[mask, 'Sd_Salience'],
                       mode = 'lines',
                       marker=dict(color="#444"),
                       line=dict(width=0),
                       fillcolor= color_from_list(
                           inlist = current_color_list, 
                           mode = 'Band'),
                       fill='tonexty',
                       legendgroup= 'Standard Deviations',
                       name= 'Standard Deviations')
        )
    else:
        fig.add_trace(
            go.Scatter(x = current_res_corr_summary.loc[mask, 'Day'], 
                       y = current_res_corr_summary.loc[mask, 'Mean_Salience']-current_res_corr_summary.loc[mask, 'Sd_Salience'],
                       mode = 'lines',
                       marker=dict(color="#444"),
                       line=dict(width=0),
                       fillcolor= color_from_list(
                           inlist = current_color_list, 
                           mode = 'Band'),
                       fill='tonexty',
                       showlegend=False,
                       legendgroup= 'Standard Deviations',
                       name= 'Standard Deviations')
        )


    fig.add_trace(
        go.Scatter(x = current_res_corr_summary.loc[mask, 'Day'], 
                   y = current_res_corr_summary.loc[mask, 'Mean_Salience'], 
                   mode = 'lines',
                   marker=dict(color= color_from_list(
                       inlist = current_color_list, 
                       mode = 'NotBand')),
                   name=str(current_factor)
        )
    )
        
        
fig.update_layout(yaxis_range=[-1,1],
                 title = 'Weather: Correlations of Salience', template = 'plotly_white')

fig.show()


# %%

# %% code_folding=[0]
def _quick_line_weather_salience_corr(current_res_corr_summary = res_corr_summary,
                                      current_Factor = 'WaterTotalInmm'):
    current_res_corr_summary = current_res_corr_summary.loc[current_res_corr_summary['Factor'] == current_Factor, :]
    current_res_corr_summary['Day'] = current_res_corr_summary['Day'].astype(int)
    current_res_corr_summary = current_res_corr_summary.sort_values('Day')

    fig = px.line(current_res_corr_summary, x = 'Day', y = 'Mean_Salience', 
                  title = 'Weather: Correlations of Salience'+': '+current_Factor, template = 'plotly_white')

    # add band
    fig.add_trace(
        go.Scatter(x = current_res_corr_summary['Day'], y = current_res_corr_summary['Mean_Salience']+current_res_corr_summary['Sd_Salience'], #y = res_corr_summary['Max_Salience'], 
                   mode = 'lines',
                   marker=dict(color="#444"),
                   line=dict(width=0),
                   showlegend=False)
    )

    fig.add_trace(
        go.Scatter(x = current_res_corr_summary['Day'], y = current_res_corr_summary['Mean_Salience']-current_res_corr_summary['Sd_Salience'], #y = res_corr_summary['Min_Salience'], 
                   mode = 'lines',
                   marker=dict(color="#444"),
                   line=dict(width=0),
                   fillcolor='rgba(68, 68, 68, 0.3)',
                   fill='tonexty',
                   showlegend=False)
    )

    fig.update_layout(yaxis_range=[-1,1])

#     fig.show()
    return(fig)


# %%
_quick_line_weather_salience_corr(
        current_res_corr_summary = res_corr_summary,
        current_Factor = 'VaporPresEst')

# %%
_quick_line_weather_salience_corr(
        current_res_corr_summary = res_corr_summary,
        current_Factor = 'TempMin').show()


# %%
# # all the plots!
# [ _quick_line_weather_salience_corr(
#     current_res_corr_summary = res_corr_summary,
#     current_Factor = cF).show() for cF in list(set(res_corr_summary['Factor']))
# ]

# %% [markdown]
# #### Summary salience
#

# %%
# get the weather data for each day in long format.
# NOTE! the position encodes the day so we'll need to add that back in before plotting. 

def _get_weather_long(
    results_list = res,
    current_day = 0):
    # drop all but current day and grouping var* (* in the first one)
    temp = [results_list[i].loc[:, ['Factor', str(current_day)]] if i == 0 else results_list[i].loc[:, [str(current_day)]] for i in range(len(results_list))]

    # rename day so that entries 
    temp = [temp[i].rename(columns = {str(current_day):str(i)}) for i in range(len(temp))]
    temp = pd.concat(temp, axis = 1)
    temp

    res_sal = temp.melt(id_vars = ['Factor']).rename(columns = {'variable':'Rep', 'value':'Salience'})
    res_sal['Day'] = current_day
    return(res_sal)


# %%
# sal_long_list = [_get_weather_long(results_list = res,
#                                    current_day = ith_day) for ith_day in all_days]

# %%
# TODO fix this dying.

# file_path = '../data/result_intermediates/sal_df_W_long.pkl'
# if os.path.exists(file_path):
#     sal_long_list = pkl.load(open(file_path, 'rb'))

# else:
#     sal_long_list = [_get_weather_long(results_list = res,
#                                    current_day = ith_day) for ith_day in all_days] 
    
#     pkl.dump(sal_long_list, open(file_path, 'wb'))

# %%

# %%
# Works, but slow process. 
# This may no longer work. If the above takes too much ram we might need to clip this into smaller parts.
# No, it's worse than that. There's too little ram to load the ist.

for ii in tqdm.tqdm(range(3)):
    # Check if the summary has been made. If not, then we'll need to selectivly run cells or clear memory for this to run
    if not os.path.exists(
        '../data/result_intermediates/sal_df_W_summary_part_day'+[
            '0-95', '96-191', '192-287'][ii]+'.pkl'): 

        
        file_path = '../data/result_intermediates/sal_df_W_long_part_day'+['0-95', 
                                                                           '96-191', 
                                                                           '192-287'][ii]+'.pkl'
        if os.path.exists(file_path):
            sal_long_list = pkl.load(open(file_path, 'rb'))

        else:
            # running this as a list comprehension on all ten causes the computer to crash. 
            # We can get around this issue thus:

            sal_long_list = [_get_weather_long(
                results_list = res,
                current_day = current_day) for current_day in [
                [int(e) for e in np.linspace(start = 0, stop = 95, num = 96)],
                [int(e) for e in np.linspace(start = 96, stop = 191, num = 96)],
                [int(e) for e in np.linspace(start = 192, stop = 287, num = 96)]
            ][ii]#all_days
            ]

            pkl.dump(sal_long_list, open(file_path, 'wb'))


# %%
# took 7 minutes to run on the front end.
ii = 0
for ii in tqdm.tqdm(range(3)):
    summary_path = '../data/result_intermediates/sal_df_W_summary_part_day'+['0-95', 
                                                                       '96-191', 
                                                                       '192-287'][ii]+'.pkl'
    if os.path.exists(summary_path):    
        sal_long_df_summary = pkl.load(open(summary_path, 'rb'))

    else:
        file_path = '../data/result_intermediates/sal_df_W_long_part_day'+['0-95', 
                                                                           '96-191', 
                                                                           '192-287'][ii]+'.pkl'
        sal_long_list = pkl.load(open(file_path, 'rb'))
        
        # crashes        
        sal_long_df_summary = []
        
        for sal_long_df in tqdm.tqdm(sal_long_list): 
            sal_long_df_summary.extend(
                [sal_long_df.groupby(['Factor', 'Day', 'Rep']
                               ).agg(
                Mean_Salience =('Salience', np.mean),
                Sd_Salience =  ('Salience', np.std),
                Min_Salience = ('Salience', np.min),
                Max_Salience = ('Salience', np.max)
                ).reset_index()])
            
        pkl.dump(sal_long_df_summary, open(summary_path, 'wb'))

# %%
sal_df_W_summary = []
for ii in range(3):
    temp = pkl.load(open('../data/result_intermediates/sal_df_W_summary_part_day'+['0-95', 
                                                                       '96-191', 
                                                                       '192-287'][ii]+'.pkl', 'rb'))
    temp = pd.concat(temp)
    sal_df_W_summary.extend([temp])
        
sal_df_W_summary = pd.concat(sal_df_W_summary)

# %%
sal_long_df_summary = sal_df_W_summary

# %%
# Sort so it plots correctly
sal_long_df_summary['Day'] = sal_long_df_summary['Day'].astype(int)
sal_long_df_summary = sal_long_df_summary.sort_values(['Factor', 'Day'])
sal_long_df_summary

# %%
# Collapse, getting rid of Replicate groups
sal_long_df_summary = sal_long_df_summary.groupby(['Factor', 'Day']
                           ).agg(
    Mean_Salience = ('Mean_Salience', np.mean),
    Sd_Salience = ('Sd_Salience', np.mean)).reset_index()

# %%



# %%
factors = list(sal_long_df_summary.Factor.drop_duplicates())

fig = go.Figure()

for ith_factor in range(len(factors)): 
    def color_from_list(inlist, mode = 'Band'):
        if mode.lower() == 'band':
            return('rgba('+', '.join([str(e) for e in inlist] )+', 0.2)')
        else:
            return('rgba('+', '.join([str(e) for e in inlist] )+', 1)')

    current_factor = factors[ith_factor]
    current_color_list = w_sal_fct_col[current_factor]
    
    mask = sal_long_df_summary['Factor'] == current_factor
    

    

    # add band
    fig.add_trace(
        go.Scatter(x = sal_long_df_summary.loc[mask, 'Day'], 
                   y = sal_long_df_summary.loc[mask, 'Mean_Salience']+sal_long_df_summary.loc[mask, 'Sd_Salience'], 
                   mode = 'lines',
                   marker=dict(color="#444"),
                   line=dict(width=0),
                   showlegend=False,
                  )
    )
    
    if ith_factor == 0:
        fig.add_trace(
            go.Scatter(x = sal_long_df_summary.loc[mask, 'Day'], 
                       y = sal_long_df_summary.loc[mask, 'Mean_Salience']-sal_long_df_summary.loc[mask, 'Sd_Salience'],
                       mode = 'lines',
                       marker=dict(color="#444"),
                       line=dict(width=0),
                       fillcolor= color_from_list(
                           inlist = current_color_list, 
                           mode = 'Band'),
                       fill='tonexty',
                       legendgroup= 'Standard Deviations',
                       name= 'Standard Deviations')
        )
    else:
        fig.add_trace(
            go.Scatter(x = sal_long_df_summary.loc[mask, 'Day'], 
                       y = sal_long_df_summary.loc[mask, 'Mean_Salience']-sal_long_df_summary.loc[mask, 'Sd_Salience'],
                       mode = 'lines',
                       marker=dict(color="#444"),
                       line=dict(width=0),
                       fillcolor= color_from_list(
                           inlist = current_color_list, 
                           mode = 'Band'),
                       fill='tonexty',
                       showlegend=False,
                       legendgroup= 'Standard Deviations',
                       name= 'Standard Deviations')
        )


    fig.add_trace(
        go.Scatter(x = sal_long_df_summary.loc[mask, 'Day'], 
                   y = sal_long_df_summary.loc[mask, 'Mean_Salience'], 
                   mode = 'lines',
                   marker=dict(color= color_from_list(
                       inlist = current_color_list, 
                       mode = 'NotBand')),
                   name=str(current_factor)
        )
    )
        
        
fig.update_layout(yaxis_range=[0,1],
                 title = 'Weather: Salience', template = 'plotly_white')

fig.show()


# %% code_folding=[23, 33, 47, 64]
# rng = np.random.default_rng(354384)
# factors = list(sal_long_df_summary.Factor.drop_duplicates())

# fig = go.Figure()

# for ith_factor in range(len(factors)): 
#     def color_from_list(inlist, mode = 'Band'):
#         if mode.lower() == 'band':
#             return('rgba('+', '.join([str(e) for e in inlist] )+', 0.2)')
#         else:
#             return('rgba('+', '.join([str(e) for e in inlist] )+', 1)')


# #     current_color = px.colors.qualitative.Dark24[ith_factor]
    
# #     current_color_list = list(np.random.randint(low = 0, high = 255, size = 3))
#     current_color_list = list(rng.integers(low=0, high=255, size=3))
#     #['rgba(68, 68, 68, 0.3)'][ith_factor]
#     current_factor = factors[ith_factor]
    
#     mask = sal_long_df_summary['Factor'] == current_factor
    

    

#     # add band
#     fig.add_trace(
#         go.Scatter(x = sal_long_df_summary.loc[mask, 'Day'], 
#                    y = sal_long_df_summary.loc[mask, 'Mean_Salience']+sal_long_df_summary.loc[mask, 'Sd_Salience'], 
#                    mode = 'lines',
#                    marker=dict(color="#444"),
#                    line=dict(width=0),
#                    showlegend=False,
#                   )
#     )
    
#     if ith_factor == 0:
#         fig.add_trace(
#             go.Scatter(x = sal_long_df_summary.loc[mask, 'Day'], 
#                        y = sal_long_df_summary.loc[mask, 'Mean_Salience']-sal_long_df_summary.loc[mask, 'Sd_Salience'],
#                        mode = 'lines',
#                        marker=dict(color="#444"),
#                        line=dict(width=0),
#                        fillcolor= color_from_list(
#                            inlist = current_color_list, 
#                            mode = 'Band'),
#                        fill='tonexty',
#                        legendgroup= 'Standard Deviations',
#                        name= 'Standard Deviations')
#         )
#     else:
#         fig.add_trace(
#             go.Scatter(x = sal_long_df_summary.loc[mask, 'Day'], 
#                        y = sal_long_df_summary.loc[mask, 'Mean_Salience']-sal_long_df_summary.loc[mask, 'Sd_Salience'],
#                        mode = 'lines',
#                        marker=dict(color="#444"),
#                        line=dict(width=0),
#                        fillcolor= color_from_list(
#                            inlist = current_color_list, 
#                            mode = 'Band'),
#                        fill='tonexty',
#                        showlegend=False,
#                        legendgroup= 'Standard Deviations',
#                        name= 'Standard Deviations')
#         )


#     fig.add_trace(
#         go.Scatter(x = sal_long_df_summary.loc[mask, 'Day'], 
#                    y = sal_long_df_summary.loc[mask, 'Mean_Salience'], 
#                    mode = 'lines',
#                    marker=dict(color= color_from_list(
#                        inlist = current_color_list, 
#                        mode = 'NotBand')),
#                    name=str(current_factor)
#         )
#     )
        
        
# fig.update_layout(yaxis_range=[0,1],
#                  title = 'Weather: Salience', template = 'plotly_white')

# fig.show()


# %% [markdown]
# ### Concat model 

# %%

# %%

# %%

# %%




# %% [markdown]
# ## Visualization workspace

# %% code_folding=[]
# Placeholder


# %%


phenotypeGBS = pd.read_pickle('../data/processed/phenotype_with_GBS.pkl')
phenotypeGBS.shape
# %%

(Y.shape, G.shape, S.shape, W.shape, )


phenotype.shape

# #### Full Model

model = tf.keras.models.load_model('../data/atlas/models/3_finalize_model_syr__rep_full/eval/hps_rep_0checkpoint00000100.h5')

saliency = Saliency(model, clone=False)
saliency_map = saliency(score_function, (x_test[0], x_test[1], x_test[2]), keepdims=True)

saliency_map

# ### Generate Predictions



# %%
plt.imshow(saliency_map[0])

# %%
# model.summary()
# tf.keras.utils.plot_model(model, show_shapes=True)
# %%





# model_checkpoint = 'hps_rep_0checkpoint00001000.h5'
path = '../3_finalize_model_syr__rep_full/eval/hps_rep_0checkpoint00000100.h5'#+model_checkpoint # fixme!
path = './eval/hps_rep_0checkpoint00000100.h5'#+model_checkpoint # fixme!
model = tf.keras.models.load_model(path)



saliency = Saliency(model, clone=False)

[   sum(sum(x_test[0] == np.nan)),
    sum(sum(x_test[1] == np.nan)),
sum(sum(sum(x_test[2] == np.nan)))
]

sum(model.predict(x_test) == np.nan)

model.summary()

# %%


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

# %%
# remade_model.predict(x_train[1])


saliency = Saliency(remade_model, clone=False)
saliency_map = saliency(score_function, (x_test[1]), keepdims=True)



# model.get_layer('dense_6').get_weights()
# model.get_layer('dense_6')

# %%
# todo TRY rebuilding the keras model using the trained weights to edit the inputs. Then pass in the right tensor. 
# my suspicision is that haveing inputs that aren't used is what's causing problems -- changes in those tensors has no effect on the gradient.

# %%
# saliency_map = saliency(score_function, (x_test[0], x_test[1], x_test[2]), keepdims=True) # Worked for full network

saliency_map = saliency(score_function, (x_test[0], x_test[1], x_test[2]), keepdims=True)
# %%

len(saliency_map)







# %%

# %%

# %% [markdown]
# # How are model errors related?
# * Do models perform poorly in their own ways?
# * Are there features of the data that are correlated with better or worse performance? 
#

# %%
# Aggregate all the errors for comparison

def mk_error_df(tf_res = tf_res,
                dnn_class = 'G', 
                dnn_rep = '0'):
    temp = pd.DataFrame()
    temp['yHat'] = [e[0] for e in tf_res[dnn_class][dnn_rep]['yHat_test']]
    temp['y']    =  tf_res[dnn_class][dnn_rep]['y_test']
    temp['error'] = temp['yHat'] - temp['y']
    temp['ModelGroup'] = dnn_class
    temp['ModelRep'] = dnn_rep
    temp = temp.reset_index().rename(columns = {'index':'observation'})
    return(temp)



df_accumulator = []
for dnn_class in ['G', 'S', 'W', 'full']:
    df_accumulator.extend( [mk_error_df(tf_res = tf_res, dnn_class = dnn_class, dnn_rep = str(dnn_rep)) for dnn_rep in range(10)] )
df_accumulator = pd.concat(df_accumulator)
df_accumulator

# %%
temp = df_accumulator.loc[:, ['observation', 'error', 'ModelGroup', 'ModelRep']
                  ].pivot_table(
    index = ['ModelGroup', 'ModelRep'], 
    columns = ['observation'], 
    values = ['error']
    )#.reset_index()

pca = PCA(n_components = 3)

pca.fit(temp)

print(pca.explained_variance_ratio_)



# temp
# temp


# temp_pca = temp.loc[:, ]


# temp_pca.reset_index()

# temp_pca.reset_index().loc[:, ['ModelGroup', 'ModelRep']]

temp_pca = pd.DataFrame(pca.transform(temp))
temp_pca['ModelRep'] = temp.reset_index().loc[:, 'ModelRep']
temp_pca['ModelGroup'] = temp.reset_index().loc[:, 'ModelGroup']

temp_pca


px.scatter_3d(temp_pca, 
              x = 0, 
              y = 1, 
              z = 2,
             color = 'ModelGroup')

# %%

# %%
df_corrs = df_accumulator.pivot_table(
    index = 'observation', 
    columns = ['ModelGroup', 'ModelRep'], 
    values = 'error').corr().round(3)

df_corrs.head()

# %%

# %%
df_accumulator['Model'] = df_accumulator['ModelGroup']+'-'+df_accumulator['ModelRep']


df_corrs = df_accumulator.pivot_table(
    index = 'observation', 
    columns = ['Model'], 
    values = 'error').corr().round(3)#.reset_index()

df_corrs.head()

# %%
# Visualizing a Pandas Correlation Matrix Using Seaborn


sns.heatmap(df_corrs#, annot=True
           )
plt.show()

# %%

# %%


fig = px.imshow(df_corrs, aspect="auto", color_continuous_scale='RdBu_r')
fig.show()

# %%
# maybe the way to think about this with the within group sd...


# I want to know 1. How consistent are a group of models?
# How much information can we get by combining models? Are there complements to be had?

# %%
# temp = pd.DataFrame(np.triu(df_corrs.to_numpy()), columns = list(df_corrs))

# temp['variable1'] = list(df_corrs)
# temp = temp.melt(id_vars = 'variable1')
# temp

# %%


# %%
temp = df_accumulator.loc[df_accumulator.ModelGroup == 'G', ].pivot_table(
        index = 'observation', 
        columns = 'ModelRep', 
        values = 'error'
        ).corr(
        ).round(3
        ).reset_index(
        ).drop(columns = 'ModelRep'
        ).melt()

temp = temp.loc[temp.value != 1, ]

px.box(temp, x = "ModelRep", y = 'value', points = 'all')


# %%

# %%

# %%

# %%
tf_res['G']['0'].keys()

# 'y_test', 'yHat_test'

# %%
tf_res.keys()

# %%

# %% [markdown]
# ## What are the high error samples?

# %%
# testIndex

# %%
temp = pd.DataFrame({
    'tensor_index' : testIndex,
    'yHat_test': [e[0] for e in tf_res['cat']['0']['yHat_test']],
    'y_test': tf_res['cat']['0']['y_test']
})


temp['errors'] = temp['yHat_test'] - temp['y_test']


q10 = np.quantile(temp['errors'], .1)
q90 = np.quantile(temp['errors'], .9)
print('Most extreme 20%:'+' <= '+str(q10)+' & >= '+ str(q90))

temp['quantile_group'] = 'q10toq90'
temp.loc[temp.errors <= q10, 'quantile_group'] = 'q10'
temp.loc[temp.errors >= q90, 'quantile_group'] = 'q90'



# x = np.random.randn(1000)
# hist_data = [x]
# group_labels = ['distplot'] # name of the dataset

hist_data = [temp['errors']]
group_labels = ['DNN']

fig = ff.create_distplot(hist_data, group_labels)
fig.show()


# %%
p_copy = phenotype.copy()
p_copy['error_group'] = ''

# what's going on with the predictions where it undershot?
indices = temp.loc[temp.quantile_group == 'q10', 'tensor_index']
mask = [True if e in indices else False for e in p_copy.index]
p_copy.loc[mask, 'error_group'] = 'under'

# what's going on with the predictions where it overshot?
indices = temp.loc[temp.quantile_group == 'q90', 'tensor_index']
mask = [True if e in indices else False for e in p_copy.index]

p_copy.loc[mask, 'error_group'] = 'over'



# filter down to test set

mask = [True if e in testIndex else False for e in p_copy.index]
p_copy = p_copy.loc[mask, ]

# %%
# temp['errors']

# %%

# %%
p_copy['y_test'] = np.nan
p_copy['yHat_test'] = np.nan
p_copy['errors'] = np.nan
p_copy['quantile_group'] = ''

for i in p_copy.index:
    p_copy.loc[i, 'y_test'] =  float(temp.loc[temp.tensor_index == i, 'y_test'])
    p_copy.loc[i, 'yHat_test'] =  float(temp.loc[temp.tensor_index == i, 'yHat_test'])
    p_copy.loc[i, 'errors'] =  float(temp.loc[temp.tensor_index == i, 'errors'])
    p_copy.loc[i, 'quantile_group'] =  list(temp.loc[temp.tensor_index == i, 'quantile_group'])[0]

    
p_copy['GrainYield_scaled'] = (p_copy['GrainYield'] -  YMean)/YStd

# %%
# sanity check. 
px.scatter(p_copy, x = 'GrainYield_scaled', y = 'y_test')

# px.scatter(p_copy, x = 'GrainYield', y = 'y_test')

# %%
np.mean(p_copy['errors'])


sns.kdeplot(p_copy['errors'], shade = True)

# %%
px.scatter(p_copy, x = 'GrainYield_scaled', y = 'errors', #color = 'quantile_group',
           marginal_y="histogram",
          width=800, height=600)



# %%
# X_lognorm = np.random.lognormal(mean=0.0, sigma=1.7, size=500)


qq = stats.probplot(p_copy['errors'], dist='norm', sparams=(1))
x = np.array([qq[0][0][0], qq[0][0][-1]])

fig = go.Figure()
fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers')
fig.add_scatter(x=x, y=qq[1][1] + qq[1][0]*x, mode='lines')
fig.layout.update(showlegend=False)
fig.show()



# %%

# %%
# temp = p_copy.loc[:, ['F', 'M', 'quantile_group', 'errors']].groupby(['F', 'M', 'quantile_group']).count().reset_index()

# px.scatter_3d(p_copy, x = 'F', y = 'M', z = 'errors', color = 'quantile_group')


# %%

# %%
# temp = p_copy.loc[:, ['ExperimentCode', 'Year', 'quantile_group', 'errors']].groupby(['ExperimentCode', 'Year', 'quantile_group']).count().reset_index()

# px.scatter_3d(p_copy, x = 'ExperimentCode', y = 'Year', z = 'errors', color = 'quantile_group')


# %%
# px.scatter(p_copy, x = 'GrainYield_scaled', y = 'errors', color = 'quantile_group')

# %%
# What do we want to compare? 
# Pedigree (F/M)
# ExperimentCode
# Year
# Yield


# %%

p_copy

# %%
p_copy["Year"] = p_copy.Year.astype(str)



# %%
p_copy["Exp_Year"] = p_copy["ExperimentCode"]+"_"+p_copy["Year"]

p_copy = p_copy.sort_values(by = "Exp_Year")

# %%

p_copy['abs_errors'] = np.abs(p_copy['errors'])

figg=px.box(p_copy, x = "Exp_Year", #x = 'Exp_Year', 
       y = 'abs_errors'#, color = 'ExperimentCode'#, points = "all"
#           width=800, height=600
      )


figg.write_image("../../../../Desktop/quickplt.svg")

# %%
px.scatter(p_copy, x = "y_test", y = "errors")


# %%

# %%
px.scatter(p_copy, x = "Pedigree", y = "abs_errors")

# %%
px.box(p_copy, x = "F", y = "abs_errors")

# %%
px.box(p_copy, x = "M", y = "abs_errors")

# %%

# %% [markdown]
# # How would one maximize or minimize yield?

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

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
