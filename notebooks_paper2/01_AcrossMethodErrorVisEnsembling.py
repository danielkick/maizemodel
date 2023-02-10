# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
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
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tf_keras_vis
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize

# for simple MLP for ensembling
from sklearn.neural_network import MLPRegressor

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
pio.templates.default = "plotly_white"

# %%
"../data/atlas/models/3_finalize_model_syr__rep_G/"
"../data/atlas/models/3_finalize_model_syr__rep_S/"
"../data/atlas/models/3_finalize_model_syr__rep_W/"
"../data/atlas/models/3_finalize_model_syr__rep_full/"


# %% [markdown]
# # Compare Goodness of Fit Across and Within Models

# %% [markdown]
# ## Preparation

# %% code_folding=[1, 50, 116]
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


# %% code_folding=[2]
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
bglr_res = bglr_results(project_folder="../models/3_finalize_model_BGLR_BLUPS")
bglr_res.retrieve_results()
bglr_res = bglr_res.results


# %% code_folding=[]
ml_res = ml_results(project_folder = "../models/3_finalize_classic_ml_10x/")
ml_res.retrieve_results()
ml_res = ml_res.results

# %%
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

# %% code_folding=[]
lmlike_res = lmlike_results(project_folder="../models/3_finalize_lm/")
lmlike_res.retrieve_results()
lmlike_res = lmlike_res.results


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


# %% [markdown]
# ### For final figs

# %%
# tmp = summary_df
# tmp.head()

# %%
# tmp2 = tmp

# tmp2.loc[tmp2.model == 'rnr', 'model'] =  "Radius NR"
# tmp2.loc[tmp2.model == 'knn', 'model'] =  "KNN"
# tmp2.loc[tmp2.model == 'rf', 'model'] =  "Rand.Forest"
# tmp2.loc[tmp2.model == 'svrl', 'model'] =  "SVR (linear)"

# tmp2.loc[tmp2.model == 'lm', 'model'] =  "LM"

# tmp2.loc[tmp2.annotation == 'Intercept Only', 'model'] =  "Training Mean"

# %%
# tmp2.to_csv("../output/r_performance_across_models.csv")


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
summary_df.head()

# %% [markdown]
# # Load data

# %% [markdown]
# ## Set up data -- used in salinence, pulling observations

# %%
phenotype = pd.read_csv('../data/processed/tensor_ref_phenotype.csv')
soil = pd.read_csv('../data/processed/tensor_ref_soil.csv')
weather = pd.read_csv('../data/processed/tensor_ref_weather.csv')

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

# %%
y_test.shape


# %% [markdown]
# # Visualization

# %% [markdown]
# ## How does aggregation change the predictive accuracy?

# %%
def get_mod_multi_yyhat(model_type = 'lm', rep_n = '0', split = 'train'):
    rep_n = str(rep_n)
    rep_n_report = rep_n
    
    if model_type == 'lm':
        temp_res = lmlike_res['Multi'][model_type]
        rep_n = 'rep'+rep_n 
    elif model_type == 'bglr':
        temp_res = bglr_res['Multi']['BLUP']
        rep_n = 'rep'+rep_n 
    elif True in [model_type == e for e in ['knn', 'rf', 'rnr', 'svrl']]:
        temp_res = ml_res['All'][model_type]
        rep_n = 'rep'+rep_n 
    elif True in [model_type == e for e in ['full', 'cat']]:
        temp_res = tf_res[model_type]
    else:
        print("temp_res must be one of 'lm', 'bglr', 'knn', 'rf', 'rnr', 'svrl', 'full', 'cat'")
        
    temp_res = temp_res[rep_n]
            
    if split not in ['train', 'test']:
        print("split must be 'train' or 'test'")
    else:
        temp = pd.DataFrame()
        temp['y'] = temp_res['y_'+split]
        if True in [model_type == e for e in ['full', 'cat']]:
            temp['yHat'] = [e[0] for e in temp_res['yHat_'+split]]
        else:
            temp['yHat'] = temp_res['yHat_'+split]
        temp['type'] = model_type
        temp['rep']  = rep_n_report
        temp['split']= split
        temp = temp.loc[:, ['type', 'rep', 'split', 'y', 'yHat']]
    return(temp)
    


# %%
# get_mod_multi_yyhat(model_type = 'lm', rep_n = '0', split = 'train')

# %%
# be able to generate a list of lists with all the models and reps for a split
# [['lm', '0', 'train'],
#  ...
#  ['cat', '9', 'train']]

def mk_mod_comb_list(
    mods_list = ['lm', 'bglr', 'knn', 'rf', 'rnr', 'svrl', 'full', 'cat'],
    reps_list = [str(irep) for irep in range(10)],
    split = 'train'):
    out = list()
    for imod in mods_list:
        for irep in reps_list:
            out = out+[[imod, irep, split]]
    return(out)


mods_list = ['lm', 'bglr', 'knn', 'rf', 'rnr', 'svrl', 'full', 'cat']
reps_list = [str(irep) for irep in range(10)]

mod_comb_list_train = mk_mod_comb_list(
    mods_list = mods_list,
    reps_list = reps_list,
    split = 'train')

mod_comb_list_test = mk_mod_comb_list(
    mods_list = mods_list,
    reps_list = reps_list,
    split = 'test')

# %%
# Create a np array with all of the models' predictions
mod_train_yhats = pd.concat([
    get_mod_multi_yyhat(
        model_type = e[0], 
        rep_n = e[1], 
        split = e[2]).loc[:, ['yHat']
                    ].rename(columns = {'yHat': e[0]+'_'+e[1]})
     for e in mod_comb_list_train], axis = 1)

mod_test_yhats = pd.concat([
    get_mod_multi_yyhat(
        model_type = e[0], 
        rep_n = e[1], 
        split = e[2]).loc[:, ['yHat']
                    ].rename(columns = {'yHat': e[0]+'_'+e[1]})
     for e in mod_comb_list_test], axis = 1)

# %%
mod_train_y = get_mod_multi_yyhat(
    model_type = 'lm',
    rep_n = '0',
    split = 'train').loc[:, ['y']]


mod_test_y = get_mod_multi_yyhat(
    model_type = 'lm',
    rep_n = '0',
    split = 'test').loc[:, ['y']]

# %%
# Fill in missings with 0

for e in list(mod_train_yhats):
    mask = mod_train_yhats[e].isna()
    mod_train_yhats.loc[mask, e] = 0

for e in list(mod_test_yhats):
    mask = mod_test_yhats[e].isna()
    mod_test_yhats.loc[mask, e] = 0


# %%
def get_rmse(observed, predicted):
    MSE = mean_squared_error(observed, predicted)
    RMSE = math.sqrt(MSE)
    return(RMSE)


# %%

# %%

# %% [markdown]
# ## Setup train/test predictions and covariates

# %%
mod_train = pd.concat([
    phenotype.loc[trainIndex, ['Pedigree', 'F', 'M', 'ExperimentCode', 'Year']
            ].reset_index().drop(columns = 'index'),
    mod_train_y,
    mod_train_yhats
], axis = 1)

mod_test = pd.concat([
    phenotype.loc[testIndex, ['Pedigree', 'F', 'M', 'ExperimentCode', 'Year']
            ].reset_index().drop(columns = 'index'),
    mod_test_y,
    mod_test_yhats
], axis = 1)


# %%
# get standard deviation of a set of models
# How much do models vary over replicates?
def get_mod_rep_std(mod_list = ['lm_0', 'lm_1'],
                    df = mod_train):
    return(df.loc[:, mod_list].std(axis = 1).mean())

def get_mod_std(mod_list = ['lm_0', 'lm_1'],
                    df = mod_train):
    return(list(df.loc[:, mod_list].std(axis = 0)))
    

# get_mod_rep_std(mod_list = ['lm_0', 'bglr_1'],
#                 df = mod_train)

# %%
# How much variability is there within each model type (in train set?)
mod_types = ['lm', 'bglr', 'knn', 'rf', 'rnr', 'svrl', 'full', 'cat']


replicate_variability_summary = pd.DataFrame([
    [e, get_mod_rep_std(
        mod_list = [e1 for e1 in list(mod_train) if re.findall(e+'_.+', e1)], 
        df = mod_train)
    ] for e in mod_types], columns = ['mod', 'std'])

replicate_variability_summary

# %%
# Based on the above, I'm limiting the number of replicates of the models that would be expected to converge
# and have std on the order of e-17 
# - lm
# - knn
# - rnr

# drop all but the 0th rep for these models
for lst in [[e+'_'+str(9-i) for i in range(9)] for e in ['lm', 'knn', 'rnr']]:
    mod_train = mod_train.drop(columns=lst)
    mod_test = mod_test.drop(columns=lst)


# %% code_folding=[]
def get_mod_cols(
    df = mod_train,
    mod_type = 'lm'):
    e = mod_type
    col_list = [e1 for e1 in list(df) if re.findall(e+'_.+', e1)]
    return(col_list)

def get_all_mod_cols(
    df = mod_train,
    mod_types = ['bglr', 'cat']):
    col_list = [get_mod_cols(
        df = df,
        mod_type = e) for e in mod_types]
    # flatten list
    out = []
    for i in range(len(col_list)):
        out += col_list[i]
    return(out)


# %%
# calculate several weighting options based on TRAINING set
all_mod_cols = get_all_mod_cols(df = mod_train, mod_types = mod_types)

# uniform weighting ------------------------------------------------------------
uniform_weights = [1/len(all_mod_cols) for i in range(len(all_mod_cols))]

# uniform weighting wrt model --------------------------------------------------
# take n model types and convert to weight per replicate
uniform_by_type_weights = [1/len(mod_types) for i in range(len(mod_types))]
n_rep_by_type = [len(get_mod_cols(df = mod_train, mod_type = e)) for e in mod_types]
# apply to column names
uniform_by_type_weights = [uniform_by_type_weights[i]/n_rep_by_type[i] for i in range(len(uniform_by_type_weights))] 
# convert to model types
temp_types = [mod_col.split('_')[0] for mod_col in all_mod_cols]
#convert to index
temp_idxs = [[i for i in range(len(mod_types)) if mod_types[i] == mod_col][0] for mod_col in temp_types]
# convert to weigths
uniform_by_type_weights = [uniform_by_type_weights[i] for i in temp_idxs]


# weigting wrt rmse ------------------------------------------------------------
# take inverse of rmse and convert to sum to one
temp = [1/get_rmse(mod_train.y, mod_train[e]) for e in all_mod_cols]
inv_rmse_weights = [e/np.sum(temp) for e in temp]

# weigting wrt std -------------------------------------------------------------
temp = get_mod_std(mod_list = all_mod_cols, df = mod_train)
temp = [1/e for e in temp]
inv_std_weights = [e/np.sum(temp) for e in temp]


# %%
def weighted_sum_cols(mod_list = ['lm_0', 'cat_0'],
                      mod_weights = [0.5, 0.5],
                      df = mod_train):
    # ensure weights sum to 1
    mod_weights = mod_weights/np.sum(mod_weights) 
    temp = df.loc[:, mod_list]
    temp = (temp * mod_weights).sum(axis = 1)
    return(temp)


# %%
def ftrain(weight_list):
    temp = weighted_sum_cols(
        mod_list = all_mod_cols,
        mod_weights = weight_list,
        df = mod_train)
    out = get_rmse(
        observed = mod_train['y'], 
        predicted = temp)
    return(out)

def ftest(weight_list):
    temp = weighted_sum_cols(
        mod_list = all_mod_cols,
        mod_weights = weight_list,
        df = mod_test)
    out = get_rmse(
        observed = mod_test['y'], 
        predicted = temp)
    return(out)


# %%
list_of_weight_lists = [uniform_weights,
                        uniform_by_type_weights,
                        inv_rmse_weights,
                        inv_std_weights]
ens_train_rmses = [ftrain(e) for e in list_of_weight_lists]
ens_test_rmses  = [ftest(e) for e in list_of_weight_lists]

# %%
# stack model (no validation) --------------------------------------------------
regr = MLPRegressor(random_state=1, hidden_layer_sizes = (100, 100), max_iter=500)
regr.fit(mod_train.loc[:, all_mod_cols], mod_train.loc[:, 'y'])

# %%
ens_train_rmses += [get_rmse(
    observed = mod_train['y'], 
    predicted = regr.predict(mod_train.loc[:, all_mod_cols]))]

ens_test_rmses += [get_rmse(
    observed = mod_test['y'], 
    predicted = regr.predict(mod_test.loc[:, all_mod_cols]))]

# %%
pd.DataFrame(zip([
    'uniform_weights',
    'uniform_by_type_weights',
    'inv_rmse_weights',
    'inv_std_weights',
    'mlp_predictions'],
    ens_train_rmses,
    ens_test_rmses), columns = ['EnsembleMethod', 'Train', 'Test'])

# %%
# Is there a small MLP that would have performed well?
# Yes, but none are stelar.

# def f(sizes = (100), iters = 500):
#     regr = MLPRegressor(random_state=1, hidden_layer_sizes = sizes, max_iter=iters)
#     regr.fit(mod_train.loc[:, all_mod_cols], mod_train.loc[:, 'y'])
#     out = [get_rmse(
#             observed = mod_train['y'], 
#             predicted = regr.predict(mod_train.loc[:, all_mod_cols])), 
#            get_rmse(
#             observed = mod_test['y'], 
#             predicted = regr.predict(mod_test.loc[:, all_mod_cols]))]
#     return(out)

#  params = [(x, y) for x in [
#      1, 3, 30, 60, 90,
#      (1, 1), (3, 3), (30, 30), (60, 60), (90, 90)                      
#                            ] for y in [1, 3, 30, 60, 90, 300]]
    
# rmses = [f(e[0], e[1]) for e in params]

# rmses_summary = pd.concat([
#     pd.DataFrame(params, columns = ['Neurons', 'Epochs']), 
#     pd.DataFrame(rmses, columns = ['TrainRMSE', 'TestRMSE'])], axis = 1)

# Neurons	Epochs	TrainRMSE	TestRMSE
# (3, 3)	1	    0.585326	0.942080
# 30    	1	    4.322395	0.960786
# (90, 90)	3	    1.301509	1.014392
# (3, 3)	3	    0.456046	1.022075

# %%

# %%
# look for saturation within a model 

# pass in a mod list so the same code can be reused for all models in addition to a single type




def resample_model_averages(individual_mods = ['cat_'+str(i) for i in range(10)],
                            n_iter = 5,
                            p = None):
    # get performance for all reps individually
    # from 2 -> max rep -1
    # get iter rmses by randomly combining the replicates
    n_mods = len(individual_mods)

    # one model ----
    individual_rmses = [
        get_rmse(observed = mod_test['y'], 
                 predicted= mod_test[e]) for e in individual_mods]
    individual_rmses = pd.DataFrame(zip(
        [1 for i in range(n_mods)],
         [[individual_mods[i]] for i in range(n_mods)],
        individual_rmses), columns = ['n_mods', 'drawn_mods', 'rmse'])

    # 2 to n-1 models ---- 
    def f(individual_mods = individual_mods, draw_mods = 2, n_iter = 5, p = None):
        if type(p) != list:
            p = [1/len(individual_mods) for i in range(len(individual_mods))]

        drawn_mods_list = [list(np.random.choice(individual_mods, draw_mods, p = p)) for i in range(n_iter)]
        drawn_mods_rmse = [get_rmse(observed = mod_test['y'], 
                                    predicted= weighted_sum_cols(
                                        mod_list = drawn_mods,
                                        mod_weights = [1/draw_mods for i in range(draw_mods)],
                                        df = mod_test) 
                                   ) for drawn_mods in drawn_mods_list]
        out = pd.DataFrame(zip(
            [draw_mods for i in range(n_iter)],
            drawn_mods_list,
            drawn_mods_rmse
        ), columns = ['n_mods', 'drawn_mods', 'rmse'])
        return(out)

    out_mid = pd.concat([f(individual_mods = individual_mods, 
                           draw_mods = i,
                           n_iter = n_iter, 
                           p = p
                           ) for i in range(2, (n_mods))], 
                           axis = 0)
    # all models ----
    all_reps_rmses = [
        get_rmse(
            observed = mod_test['y'], 
            predicted= weighted_sum_cols(
                mod_list = individual_mods,
                mod_weights = [1/n_mods for i in range(n_mods)],
                df = mod_test) )]

    all_reps_rmses = pd.DataFrame(zip(
        [n_mods for i in range(n_mods)],
        [individual_mods for i in range(n_mods)],
        all_reps_rmses), columns = ['n_mods', 'drawn_mods', 'rmse'])

    out = pd.concat([individual_rmses,
                     out_mid,
                     all_reps_rmses])
    out = out.reset_index().drop(columns = 'index')
    return(out)


# %% [markdown] heading_collapsed=true
# ### Averaging within a model type

# %% hidden=true
temp = resample_model_averages(
    individual_mods = ['cat_'+str(i) for i in range(10)],
    n_iter = 50,
    p = None)
temp.head()

# %% hidden=true
px.scatter(temp, x = 'n_mods', y = 'rmse', trendline='lowess')

# %% hidden=true
temp = resample_model_averages(
    individual_mods = ['bglr_'+str(i) for i in range(10)],
    n_iter = 50,
    p = None)
temp.head()

# %% hidden=true
px.box(temp, x = 'n_mods', y = 'rmse')

# %% [markdown] heading_collapsed=true
# ### Averaging Across model types (weighting by model type)

# %% hidden=true

# %% hidden=true
# all models
temp = resample_model_averages(
    individual_mods = all_mod_cols,
    n_iter = 50,
    p = uniform_by_type_weights)
temp.head()

# %% hidden=true
px.box(temp, x = 'n_mods', y = 'rmse')

# %% hidden=true
px.scatter(temp, x = 'n_mods', y = 'rmse', trendline = 'lowess')

# %% hidden=true

# %% hidden=true
# best few models
temp = resample_model_averages(
    individual_mods = ['bglr_'+str(i) for i in range(10)] + ['cat_'+str(i) for i in range(10)],
    n_iter = 50#,
#     p = uniform_by_type_weights
)
temp.head()

# %% hidden=true
# px.violin(temp, x = 'n_mods', y = 'rmse'#, box=True#, points="all"
#          )

px.box(temp, x = 'n_mods', y = 'rmse'#, box=True#, points="all"
         )

# %% [markdown]
# ### draw 2 models

# %%
temp = [[x, y] for x in all_mod_cols for y in all_mod_cols]
temp = pd.DataFrame(temp, columns=['Model1', 'Model2'])
temp['RMSE'] = np.nan
temp.head()

# %%
i = 0
for i in tqdm.tqdm(range(temp.shape[0])):
    temp.loc[i, 'RMSE'] = get_rmse(
        observed = mod_test['y'], 
        predicted = weighted_sum_cols(
               mod_list = [temp.loc[i, 'Model1'], temp.loc[i, 'Model2']],
            mod_weights = [0.5, 0.5],
                     df = mod_test))

# %%
temp.head()

# %%
px.scatter(temp, x = "Model1", y = "Model2", color = "RMSE")

# %%
temp_wide = temp.pivot(index='Model1', columns='Model2', values='RMSE').reset_index()
temp_wide.index = temp_wide['Model1']
temp_wide = temp_wide.drop(columns='Model1')

# %%
px.imshow(temp_wide)

# %%

# %%

# %%
# px.scatter_3d(temp, 
#               x = "Model1", y = "Model2", z = "RMSE", 
#               color = "RMSE", 
# #          size = "RMSE"
#              )

# %%

# %%

# %%
years = [2022-i for i in range(9)]
len(years)

# %%
y_cmb = [[x, y, z] for x in years for y in years for z in years]
len(y_cmb)


# %% [markdown]
# ### Training Set
# Now that we have all the predictions what would the ideal mix of these models be?

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
# need to calculate modelwise varaiance 
# note that the std of the yhats (across replicates wrt to observation) will be equal to the same of error
def model_err_stats(
    df = mod_train_yhats,
    ys = mod_train_y,
    mod_type = 'cat'):

    select_cols = [e for e in list(df) if re.match(mod_type+"_\d+", e)]

    yhat_mat = np.matrix( df.loc[:, select_cols])


    yhat_stds = np.std(yhat_mat, axis = 1)

    # convert to residuals
    yhat_mat = (yhat_mat - np.matrix(ys['y']).T)
    errs_mean = np.mean(yhat_mat, axis = 1)

    out = pd.DataFrame(
        np.concatenate([
            yhat_stds,
            errs_mean], axis = 1), 
        columns = ['yhat_stds', 
                   'err_mean'])
    return(out)




# %%
mod_list = list(set( [e.split('_')[0] for e in list(df)]))

mod_err_stats_list = [
    model_err_stats(
        df = mod_train_yhats,
        ys = mod_train_y,
        mod_type = mod) for mod in mod_list]


# %%
def f(i, mod_list, mod_err_stats_list):
    out = mod_err_stats_list[i]
    out = pd.DataFrame(np.mean(out, 0)).T
#     out['rmse'] = np.sqrt(np.mean((mod_err_stats_list[i].loc[:, 'err_mean'])**2))
    out['mod'] = mod_list[i]
    return(out)

mod_err_stats_df = pd.concat([f(i, mod_list, mod_err_stats_list) for i in range(len(mod_list))])
    
mod_err_stats_df

# %%
# Now we can try different weighting schemes:




# %%

# %%

# %%
px.scatter(mod_err_stats_list[5], x = 'yhat_stds', y = 'err_mean', trendline='ols')

# %%
get_rmse(ys, pd.DataFrame(np.mean(yhat_mat, axis = 1)))

# %%
ys

# %%
summary_df

px.scatter(summary_df, x = "key1", y = "train_rmse")

# %%
px.scatter(summary_df, x="key1", y="test_rmse")

# %% [markdown]
# ### Test Set

# %%
# selecting a subset of columns:
# mod_test_yhats.loc[:, [e for e in list(mod_test_yhats) if re.match('cat_.+', e)]]

# %%
get_rmse(mod_test_y, mod_test_yhats['lm_0'])

# %%
# What happens as we ensemble more and more models of a type?
# rmse ensemble mods 0 -> 9; group by model type
# 

# %%
get_rmse(mod_test_y, np.mean(mod_test_yhats, axis = 1))

# %%

# %%
get_rmse(mod_test_y, mod_test_yhats['lm_0'])




# %%

# %%
# get rmses for grouping by replicate
rmse_by_rep = pd.DataFrame(
    [ [i, get_rmse(
        mod_test_y, 
        np.mean(
            mod_test_yhats.loc[:, [e for e in list(mod_test_yhats) if re.match('.+_'+str(i), e)]], 
            axis = 1))] for i in reps_list]
).rename(columns = {0:'Group', 1:'RMSE'})

rmse_by_rep

# %%
# get rmses for grouping by model type
rmse_by_mod = pd.DataFrame(
    [ [i, get_rmse(
        mod_test_y, 
        np.mean(
            mod_test_yhats.loc[:, [e for e in list(mod_test_yhats) if re.match(str(i)+'_.+', e)]], 
            axis = 1))] for i in mods_list]
).rename(columns = {0:'Group', 1:'RMSE'})

rmse_by_mod

# %%

pd.concat([rmse_by_rep, rmse_by_mod])

# %%
# what about pairwise model ensembles?

# i = 0
# mod1 = 'cat'
# mod2 = 'bglr'
# [e for e in list(mod_test_yhats) if (re.match(mod1+'_'+str(i), e) or re.match(mod2+'_'+str(i), e))]


# %%

# %%
rmse_matrix = np.zeros(shape = [
    len(list(mod_test_yhats)), 
    len(list(mod_test_yhats))])

ij_list = list()
for i in range(len(list(mod_test_yhats))):
    for j in range(i, len(list(mod_test_yhats))):
        ij_list += [[i, j]]
        

for ijs in tqdm.tqdm(ij_list):
    i = ijs[0]
    j = ijs[1]
    rmse_matrix[i, j] =  get_rmse(
            mod_test_y, 
            np.mean(
                mod_test_yhats.loc[:, [list(mod_test_yhats)[i], 
                                       list(mod_test_yhats)[j]] ], 
                axis = 1))

# %%
px.imshow(rmse_matrix)

# %%
rmse_matrix = pd.DataFrame(rmse_matrix, columns = list(mod_test_yhats))

# %%
rmse_matrix['mod1'] = list(mod_test_yhats)
rmse_matrix = rmse_matrix.melt(id_vars=['mod1'], value_vars=[e for e in list(rmse_matrix) if e != 'mod1'])
rmse_matrix = rmse_matrix.rename(columns = {'variable':'mod2', 'value':'RMSE'})
# drop empty values
rmse_matrix = rmse_matrix.loc[rmse_matrix.RMSE != 0, ]

# %%
# rmse_matrix['only_mod_1'] = rmse_matrix.mod1.str.replace('_.', '', regex = True)
# rmse_matrix['only_mod_2'] = rmse_matrix.mod2.str.replace('_.', '', regex = True)

# %%
rmse_matrix['pairing'] = rmse_matrix.mod1.str.replace('_.', '', regex = True) +'_'+ rmse_matrix.mod2.str.replace('_.', '', regex = True)

# %%
rmse_matrix

# %%
rmse_sum_stats = rmse_matrix.groupby('pairing'
                           ).agg(MeanRMSE = ('RMSE', np.mean),
                                 StdRMSE = ('RMSE', np.std)
                           ).reset_index()

# %%
rmse_matrix = rmse_matrix.merge(rmse_sum_stats)

# %%
rmse_matrix = rmse_matrix.sort_values('MeanRMSE')

# %%
px.box(rmse_matrix, x = 'pairing', y = 'RMSE')

# %%

rmse_matrix.mod1

# %%
pd.DataFrame(mod_comb_list_test)

# %%

# %%

# %% [markdown]
# ## What performance would be reasonable?
# If I use the training data to create ensemble weights, how would these ensembles do?
#

# %%

# %%

# %% [markdown]
# ### Ensembling Performance

# %%

# %%

# %%

# %% [markdown]
# ### Model Stacking

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

# %% [markdown]
# # Setup -- Migrated from top

# %%

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
