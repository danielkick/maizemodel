# -*- coding: utf-8 -*-
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
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

import json  # for reading in dictionaries


# import umap
import sklearn
from sklearn import mixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tqdm

import matplotlib as mpl
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# %%
# kForGeno = 4
# useGenoCluster = False # we decided that practically there won't be a situation where a crop's genetics are fully unknown or unknowable.
# # Pragmatically, not trying to account for genetic grouping increases the available sample size considerably for training purposes

# # Which method should be used in setting up test/train splits?
# """
# Method 1 --- 
#     Randomly draw site/year combinations and add them to the test set. 
#     Remove matching sites from the possible training set. 
#     Stop when target train/test split is reached.
    
# Method 2 ---
#     Randomly draw a given fraction of sites   
#     Upsample or downsample observations within those sites to reach a target split.

# """
# makeTestTrainSplitWith = 'Method2'

# %%
# read in production datasets
metadata  = pd.read_pickle('../data/processed/metadata.pkl') # for double checking expeirment code independence
phenotype = pd.read_pickle('../data/processed/phenotype.pkl')
weather   = pd.read_pickle('../data/processed/weather.pkl')
soil      = pd.read_pickle('../data/processed/soil.pkl')
# genoPCA   = pd.read_pickle('../data/processed/genoPCA.pkl')

phenotypeGBS = pd.read_pickle('../data/processed/phenotype_with_GBS.pkl')


# %% code_folding=[0]
# this function is to help sort out a problem wherein we have nas in the output of the GPCA tensor.
# I've traced main source of this issue to the assumption that if there is a genome lookup code there is a genome to be looked up.
# This assumption is faulty because TASSEL filtered out some taxa based on % coverage. 
# Sources of NAs 
# 1. Main source is filtering in Tassel
# 2. Minor source tbd

def check_n_obs_matching_pca(
    inputdf = phenotypeGBS,
    outPCAPath = '../data/raw/G2FGenomes/pca_in_out/PC_Imputed_mH_with_pr_Filter_Filter.txt'):

    outPCA = pd.read_table(outPCAPath, skiprows=2)

    inputdf = inputdf.copy()
    inputdf = inputdf.loc[:, ['F_MatchBest', 'M_MatchBest']]

    inputdf = inputdf.merge(
        pd.DataFrame(outPCA.Taxa).rename(columns = {'Taxa':'F_MatchBest'}).assign(F_Found = True), 
        how = 'left')
    inputdf = inputdf.merge(
        pd.DataFrame(outPCA.Taxa).rename(columns = {'Taxa':'M_MatchBest'}).assign(M_Found = True), 
        how = 'left')

    # In table supplied to TASSEL
    mask = ((inputdf.F_Found) & (inputdf.M_Found))

    print("""
    Initial samples ----- """+str(inputdf.shape[0])+"""
    Non-NA  samples ----- """+str(inputdf.loc[(inputdf.F_MatchBest.notna() & inputdf.M_MatchBest.notna()), : ].shape[0])+"""
    PCA Matched samples - """+str(inputdf.loc[mask, ].shape[0])+"""
    """)
    
    return(inputdf.loc[mask, ])

temp_matchedDf = check_n_obs_matching_pca(
    inputdf = phenotypeGBS,
    outPCAPath = '../data/raw/G2FGenomes/pca_in_out3/PCs.txt') # <-  new, following reworking merger of input data. 

# %%
# make any groupwise reductions here. For example, we're dropping onterio (ONH2)
phenotype = phenotype.loc[phenotype.ExperimentCode != 'ONH2', ]

# %%
phenotype= phenotype.reset_index().drop(columns = ['index'])

# %%
phenotype.groupby('Year').count()


# %%
# make any groupwise reductions here. For example, we're dropping onterio (ONH2)
phenotypeGBS = phenotypeGBS.loc[phenotypeGBS.ExperimentCode != 'ONH2', ]

# %%
phenotypeGBS= phenotypeGBS.reset_index().drop(columns = ['index'])

# %%
phenotypeGBS.groupby('Year').count()

# %% code_folding=[0]
# We'll use this later to pair up the indicies in phenotype with those in the reduced dataset (i.e. after we filter)
phenotypeGBSForIndices = phenotypeGBS.copy()
phenotypeGBS = phenotypeGBS.loc[phenotypeGBS.HasGenomes, ].reset_index().drop(columns = 'index').copy()

# %% [markdown]
# ### make numeric pedigree

# %%
temp = phenotype.loc[:, ['Pedigree', 'F', 'M']]
temp = temp.drop_duplicates().reset_index().drop(columns = ['index'])

possibleParents = pd.DataFrame(temp.loc[:, 'F']).merge(pd.DataFrame(temp.loc[:, 'M']).rename(columns = {'M': 'F'}), how = 'outer').drop_duplicates().reset_index().drop(columns = ['index'])

#####
# apply any desired family wise groupings here.
#####

# %%
"""
The plan here is to use regexs to create familywise groupings. 
Some parents are themselves crosses so we'll split those and apply the regexs again.
After this is done looking at the unique values of the twice cleaned grand parents 
(where an inbred B73 is treated as B73xB73) act as a lookup table to convert the parents into the cleaned grand parents.

start with parent (F), F -> FAdj -> (FSplitL, FSplitL) -> (FSplitLAdj, FSplitLAdj)

After we have this for both parents we'll add the 0.25 into the corresponding grand parent and use that as the matrix for the G component.
"""

# %%

# %% code_folding=[13]
import re

# drop some by exact match:
removeTheseParents = ['(CML442-B_CML343-B-B-B-B-B-B)-B-B-1-1-B-B-B-1-B12-1-B19', 
                      '(LAMA2002-35-2-B-B-B-B_CG44)-1-3-B-1-1-B24-B5-B16', 
                      '(TX739);LAMA2002-10-1-B-B-B-B3-B7ORANGE-B6'
                     ]
for rmParent in removeTheseParents:
    possibleParents = possibleParents.loc[possibleParents.F != rmParent, ]


possibleParents = possibleParents.reset_index().drop(columns = ['index'])

genoGroupDict = {
                       '^A\d+$':'Addd',    # <- a little different
#  '^B[0-6 | 8-9][0-2 | 4-9]\d*$': 'Bdd',     # any non-b73 B##s
             '^B(?![7][3])\d*$': 'Bdd',     # any non-b73 B##s
                   'BGEM-\d+-N':'BGEM-dddd-N',
                   'BSSSC0_\d+':'BSSSC0_ddd',
                      '^CG\d+$':'CGddd',    # <- a little different
                      '^CML\d+$':'CMLddd', 
                  '^DKC61-\d+$':'DKC61-dd',
                  '^DKC62-\d+$':'DKC62-dd',
                  '^DKC63-\d+$':'DKC63-dd',
                  '^DKC65-\d+$':'DKC65-dd',
                  '^DKC66-\d+$':'DKC66-dd',
                  '^DKC67-\d+$':'DKC67-dd',
                  '^DKC69-\d+$':'DKC69-dd', # <- a little different
                     'GEMN-\d+':'GEMN-dddd',
                     'GEMS-\d+':'GEMS-dddd',
    
                       '^M\d+$':'Mdddd', # <- a little different
                    'MBNILB\d+':'MBNILBddd', 
                       '^N\d+$':'Nddd',
                      '^NC\d+$':'NCdd',    
           'NILASQ4G11I\d+S\d+':'NILASQ4G11Iddsd',
           'NILASQ4G21I\d+S\d+':'NILASQ4G21Iddsd',
           'NILASQ4G31I\d+S\d+':'NILASQ4G31Iddsd',
           'NILASQ4G41I\d+S\d+':'NILASQ4G41Iddsd',
           'NILASQ4G51I\d+S\d+':'NILASQ4G51Iddsd',
           'NILASQ4G61I\d+S\d+':'NILASQ4G61Iddsd',
           'NILASQ4G71I\d+S\d+':'NILASQ4G71Iddsd',    
                      'NYH-\d+':'NYH-ddd',
                     'P11\d+AM':'P11ddAM.', 
                   'P11\d+AMXT':'P11ddAMXT', 
                        'P\d+.':'Pdddd.', 
    'PHG(?![3][9] | [4][7])\d+':'PHGdd', # any non phg47 , non phg39
                       'PHJ\d+':'PHJdd', 
                       'PHK\d+':'PHKdd', 
                       'PHM\d+':'PHMdd', 
             'PHN(?![1][1])\d+':'PHNdd', # any non phn11 
                       'PHP\d+':'PHPdd', 
                       'PHR\d+':'PHRdd', 
             'PHT(?![6][9])\d+':'PHTdd', # any non pht69
                       'PHV\d+':'PHVdd', 
             'PHW(?![6][5])\d+':'PHWdd', # any non phw65
                         'R\d+':'Rddd', 
                      'S832\d+':'S832d', 
                        'TX\d+':'TXddd.',       
                   'W10001_\d+':'W10001_dddd',
                   'W10002_\d+':'W10002_dddd',
#                  'W10001_\d+':'W10001_dddd',
                   'W10004_\d+':'W10004_dddd',
                   'W10005_\d+':'W10005_dddd',
                   'W10006_\d+':'W10006_dddd',
#                  'W10001_\d+':'W10001_dddd',
                   'W10008_\d+':'W10008_dddd',
                     'Z013E\d+':'Z013Edddd',
                     'Z022E\d+':'Z022Edddd',
                     'Z029E\d+':'Z029Edddd',
                     'Z030E\d+':'Z030Edddd',
                     'Z031E\d+':'Z031Edddd',
                     'Z032E\d+':'Z032Edddd',
                     'Z033E\d+':'Z033Edddd',
                     'Z034E\d+':'Z034Edddd',
                     'Z035E\d+':'Z035Edddd',
                     'Z036E\d+':'Z036Edddd',
                     'Z037E\d+':'Z037Edddd',
                     'Z038E\d+':'Z038Edddd',
    # One off substitutions
                          'CG_333':'CGddd',
                     '\?\?\?TX205':'TX205',
    
    # Crosses that will need to be split
        'B73_NC230-\d+-1-1-1-1':'B73_NC230-ddd-1-1-1-1',
                'B73_PHG39-\d+':'B73_PHG39-dd',    
               'LH82_PH207-\d+':'LH82_PH207-dd',
               'LH82_PHG47-\d+':'LH82_PHG47-dd',
               'MO44_LH145_\d+':'MO44_LH145_dddd',
               'MO44_PHW65_\d+':'MO44_PHW65_dddd',
    'MOG_LH123HT-\d+-1-1-1-1-B':'MOG_LH123HT-ddd-1-1-1-1-B',
                'MOG_LH145_\d+':'MOG_LH145_dddd',   
       'MOG_MO45-\d+-1-1-1-1-B':'MOG_MO45-ddd-1-1-1-1-B',
      'MOG_NC230-\d+-1-1-1-1-B': 'MOG_NC230-ddd-1-1-1-1-B',
                 'MOG_OH43_\d+':'MOG_OH43_dddd',
      'MOG_PHG83-\d+-1-1-1-1-B':'MOG_PHG83-224-1-1-1-1-B',
               'MO44_PHG47_\d+':'MO44_PHG47_dddd',
              'PH207_PHG47-\d+':'PH207_PHG47-dd',
              'PHN11_LH145_\d+':'PHN11_LH145_dddd',
               'PHN11_OH43_\d+':'PHN11_OH43_dddd',
              'PHN11_PHG47_\d+':'PHN11_PHG47_dddd',
              'PHN11_PHW65_\d+':'PHN11_PHW65_dddd',
                'PHW65_MOG_\d+':'PHW65_MOG_dddd',
    
    # Things that look like crosses 
                    'B73XPHP02':'B73_PHP02', 
                       'CGX333':'CGddd_333', 
                    'FS5424XRR':'FS5424_RR', 
                  'LH119XPHP02':'LH119_PHP02', 
                  'LH132XPHP02':'LH132_PHP02', 
                  'LH195XPHP02':'LH195_PHP02',
                  'PHG39XPHP02':'PHG39_PHP02',
    
    # allow for XX and XX_dd, XX-dd to be grouped together
    }



# Retain these codes separately
# B73 # -------------------------------- ✓
# LH82 # ------------------------------- 
# LH123HT-ddd-1-1-1-1-B # -------------- 
# LH145_dddd # ------------------------- 
#           # <- LH195 # ---------------     
# MO44      # <- MO44  # --------------- na
# MOG       # <- MOG   # --------------- na
# # MOG_dddd # ------------------------- na
# MO45-ddd-1-1-1-1-B # ----------------- 
# NC230-ddd-1-1-1-1 # ------------------ 
# NC230-ddd-1-1-1-1-B # ---------------- 
# OH43_dddd # -------------------------- 
# PH207-dd # --------------------------- 
# PHG47 # ------------------------------  ✓
# # PHG47_dddd # ----------------------- 
# PHG47-dd # --------------------------- 
# PHG39-dd # --------------------------- ✓
# PHG83-224-1-1-1-1-B # ---------------- 
# PHN11    # <- PHN11 # ---------------- ✓
#          # <- PHT69 # ---------------- ✓
# PHW65    # <- PHW65 # ---------------- ✓
# # PHW65_dddd # ----------------------- 



# fill in F adjusted for each entry 
possibleParents['FAdj'] = ''

for i in range(possibleParents.shape[0]):
    entry = possibleParents.loc[i, 'F']
    for key in genoGroupDict.keys():        
        if re.match(key, str(entry)): 
            possibleParents.loc[i, 'FAdj'] = genoGroupDict[key]
                      
            

# %%
# possibleParents['F'] = possibleParents['FAdj']

possibleParents.loc[possibleParents.FAdj == '', 'FAdj'] = possibleParents.loc[possibleParents.FAdj == '', 'F']


# %%
possibleParents['FSplitL'] = ''
possibleParents['FSplitR'] = ''
possibleParents

# %%
# Still have to deal with crosses of crosses... 
# split, check if the last split has only numbers if not print the split to confirm it's okay. 

for i in range(possibleParents.shape[0]):
    entry = possibleParents.loc[i, 'FAdj']
    entrySplit = entry.split(sep = '_', maxsplit = 1)
    if len(entrySplit) == 1:
        possibleParents.loc[i, 'FSplitL'] = entry
        
    if len(entrySplit) == 2: # false positive _ separating id and index not parents
        if re.search('^d+', entrySplit[1]):
            possibleParents.loc[i, 'FSplitL'] = entry
        else:
            possibleParents.loc[i, 'FSplitL'] = entrySplit[0]
            possibleParents.loc[i, 'FSplitR'] = entrySplit[1]

possibleParents

# %%
semiProcessedGenoGroup = pd.DataFrame(pd.concat([possibleParents.loc[:, 'FSplitL'], 
                                                     possibleParents.loc[:, 'FSplitR']]), 
                                          columns = ['Entry']
                                         ).drop_duplicates().reset_index().drop(columns = ['index'])

semiProcessedGenoGroup['EntryAdj'] = ''

# use the previous dict to clean up these names too.
for i in range(semiProcessedGenoGroup.shape[0]):
    entry = semiProcessedGenoGroup.loc[i, 'Entry']
    for key in genoGroupDict.keys():        
        if re.match(key, str(entry)): 
            semiProcessedGenoGroup.loc[i, 'EntryAdj'] = genoGroupDict[key]
            
semiProcessedGenoGroup.loc[semiProcessedGenoGroup.EntryAdj == '', 'EntryAdj'
                          ] = semiProcessedGenoGroup.loc[semiProcessedGenoGroup.EntryAdj == '', 'Entry']

# %%
# These twice processed names are what we'll use for the g matrix.
GenoGroups = list(semiProcessedGenoGroup['EntryAdj'].drop_duplicates())

GenoGroups[0:5]

# %%
# This will be the lookup table for converting parents into weights.
# We'll use this for each parent and we go back as far as grand parents so each
# match in FSplit_Adj will add +0.25 to the final matrix.

parentMatchUp = possibleParents.merge(
    semiProcessedGenoGroup.rename(
        columns = {'Entry':'FSplitL',
                   'EntryAdj':'FSplitLAdj'
                   }), 
    how  = 'left').merge(
    semiProcessedGenoGroup.rename(
        columns = {'Entry':'FSplitR',
                   'EntryAdj':'FSplitRAdj'
                   }), 
    how  = 'left')

# Carry non adjusted values forward
parentMatchUp.loc[parentMatchUp.FSplitLAdj == '', 'FSplitLAdj'
                 ] = parentMatchUp.loc[parentMatchUp.FSplitLAdj == '', 'FSplitL']
parentMatchUp.loc[parentMatchUp.FSplitRAdj == '', 'FSplitRAdj'
                 ] = parentMatchUp.loc[parentMatchUp.FSplitRAdj == '', 'FSplitR'] 
# For non-crosses carry that info forward (i.e. the Left gets copied to the right)
parentMatchUp.loc[parentMatchUp.FSplitRAdj == '', 'FSplitRAdj'
                 ] = parentMatchUp.loc[parentMatchUp.FSplitRAdj == '', 'FSplitLAdj']

parentMatchUp

# %%
# use GenoGroups, parentMatchUp, and Pedigrees to make numerical pedigree

# set up output
res = pd.DataFrame(np.zeros((temp.shape[0], len(GenoGroups)+2)),
                  columns = ['Pedigree', 'Ready']+list(GenoGroups) )

res['Pedigree'] = temp['Pedigree']
res['Ready'] = False

import tqdm
for i in tqdm.tqdm(range(temp.shape[0])):
    for parent in ['F', 'M']:
#         print(i)        
        if temp.loc[i, parent] not in removeTheseParents:
            currentParent = parentMatchUp.loc[parentMatchUp.F == temp.loc[i, parent] , ]
            res.loc[i, list(currentParent['FSplitLAdj'])[0]] = res.loc[i, list(currentParent['FSplitLAdj'])[0]] +0.25
            res.loc[i, list(currentParent['FSplitRAdj'])[0]] = res.loc[i, list(currentParent['FSplitRAdj'])[0]] +0.25

res
# Some of these will lack 



# %%
res.loc[:, 'Ready'] = res.loc[:, GenoGroups].sum(axis = 1)

res.loc[res.Ready == 1.0,  'Ready'] = True
res.loc[res.Ready != True, 'Ready'] = False 

res

# %%
# --------------------------------------------------------------------------- #
# Clean up phenotype before writing (some of the genotype data is unmatched)
# --------------------------------------------------------------------------- #
phenotype = pd.DataFrame(res.loc[res.Ready == True, 'Pedigree']).merge(phenotype)

res = res.loc[res.Ready == True].drop(columns = ['Ready'])
numericPedigree = res



# %%

# %%

# %% [markdown]
# # Write out dataset 0 (numeric pedigree)

# %%
trainSamples = phenotype.shape[0]
trainTimeSteps = (75+1+212) # preplant + plant + postplant
# trainFeaturesWeather = 19
trainFeaturesWeather = weather.shape[1] - 3 # ExperimentCode	Year	Date
# trainFeaturesSoil = 22
trainFeaturesSoil = soil.shape[1] - 2 # ExperimentCode	Year
# trainFeaturesGeno = 370
trainFeaturesGeno = numericPedigree.shape[1]-1

# %%
# Set up np arrays 
Y = np.zeros((trainSamples, 1))
G = np.zeros((trainSamples,                 trainFeaturesGeno))
S = np.zeros((trainSamples,                 trainFeaturesSoil))
W = np.zeros((trainSamples, trainTimeSteps, trainFeaturesWeather))

# %%

# %%
# Fill Response Variable
Y = phenotype.GrainYield.to_numpy()
Y = Y.astype(float)

# Overwrite Genetics 
    # PCA transformed genome
# G = pd.DataFrame(phenotype.loc[:, 'GBS']).merge(genoPCA).drop(columns = ['GBS']).to_numpy()
    # Numeric pedigree
G = pd.DataFrame(phenotype.loc[:, 'Pedigree']).merge(numericPedigree).drop(columns = ['Pedigree']).to_numpy()

# Overwrite Soil
S = pd.DataFrame(phenotype.loc[:, ['ExperimentCode', 'Year'] ]).merge(soil).drop(columns = ['ExperimentCode', 'Year']).to_numpy()


# %%
# to make this easier later:
GNumNames = list(pd.DataFrame(phenotype.loc[:, 'Pedigree']).merge(numericPedigree).drop(columns = ['Pedigree']))

with open('../data/processed/GNumNames.txt', 'w') as convert_file:
      convert_file.write(json.dumps(GNumNames))

# %%
# Fill in Weather. Note that leapyears can mess up the number of days if using timedelta. 
# To get around this we're sorting and using the index here.
uniquePlantings = phenotype.loc[:, ['ExperimentCode', 'Year', 'DatePlanted']
                               ].drop_duplicates().reset_index().drop(columns = ['index'])

# for i in tqdm.tqdm(range(len(uniquePlantings))):   
for i in range(len(uniquePlantings)):
    mask = (weather.ExperimentCode == uniquePlantings.loc[i, 'ExperimentCode']
       ) & (weather.Year == uniquePlantings.loc[i, 'Year']
       )

    temp = weather.loc[mask, ]#.drop(columns = ['ExperimentCode', 'Year', 'Date']).to_numpy().copy()
    temp = temp.sort_values('Date')

    temp = temp.reset_index().drop(columns = ['index']).reset_index()

    datePlantedIndex = temp.loc[temp.Date == uniquePlantings.loc[i, 'DatePlanted'], 'index']
    datePlantedIndex = int(datePlantedIndex)

    temp = temp.loc[(temp.index >= (datePlantedIndex -  75)) &
             (temp.index <= (datePlantedIndex + 212)) , ]
    
    if sum([sum(temp[col].isna()) for col in list(temp)]) > 0:
            print('i', i, 'missing', sum([sum(temp[col].isna()) for col in list(temp)]))
    
    temp = temp.drop(columns = ['ExperimentCode', 'Year', 'Date', 'index']).to_numpy().copy()


    mask = (phenotype.ExperimentCode == uniquePlantings.loc[i, 'ExperimentCode']
           ) & (phenotype.Year == uniquePlantings.loc[i, 'Year']
           ) & (phenotype.DatePlanted == uniquePlantings.loc[i, 'DatePlanted']
           )

    updateSamples = list(phenotype.loc[mask, ].reset_index().loc[:, 'index'])

    if sum(sum(np.nan == temp)) != 0:
        print(i, 'missing:', sum(sum(np.nan == temp)))

    if (temp.shape == (288,19)):
        W[updateSamples, :, :] = temp
    else:
        print(i, 'had shape', temp.shape)


# %%
[x.shape for x in [Y, G, S, W]]

# %%
# write out for easy reference
phenotype.to_csv('../data/processed/tensor_ref_phenotype.csv')
soil.to_csv('../data/processed/tensor_ref_soil.csv')
weather.to_csv('../data/processed/tensor_ref_weather.csv')


# %%
# np.save('../data/processed/Y.npy', Y)
# np.save('../data/processed/G.npy', G)
# np.save('../data/processed/S.npy', S)
# np.save('../data/processed/W.npy', W)

# phenotype.to_pickle('../data/processed/phenotype_for_keras.pkl')


# %% [markdown]
# # Write out dataset 1 (G == PCA)

# %%
# making lemonade of lemons here we'll use the existing PCA transform and get the average value for each point.
PCParent = pd.read_table('../data/raw/G2FGenomes/pca_in_out3/PCs.txt', skiprows=2)

# %%
import os
os.listdir("../data/")




# %%
import re

# e  = list(PCParent.Taxa)[0]
# [e for e in range(len(list(PCParent.Taxa))) if re.match(".*GEMS.*", list(PCParent.Taxa)[e]) ]


PCParent.loc[PCParent.index >= 910, ]

# %%

# %%
import plotly.express as px

fig = px.scatter_3d(PCParent, x='PC1', y='PC2', z='PC3', opacity=0.3)

zoomOut = 1.5 
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=zoomOut, y=zoomOut, z=zoomOut)
)

fig.update_layout(scene_camera=camera)
fig.write_image("DemoPCAParents.svg")
fig.show()
# list(PCParent.Taxa)

# %%
temp = pd.read_table('../data/raw/G2FGenomes/pca_in_out3/Eigenvalues.txt')




pc_for_25 = min(temp.loc[temp['cumulative proportion'] > .25, 'PC'])
pc_for_50 = min(temp.loc[temp['cumulative proportion'] > .50, 'PC'])
pc_for_75 = min(temp.loc[temp['cumulative proportion'] > .75, 'PC'])

plt.vlines(pc_for_25, ymin = 0, ymax = temp.loc[temp.PC == pc_for_25, 'cumulative proportion'], color = 'grey')
plt.vlines(pc_for_50, ymin = 0, ymax = temp.loc[temp.PC == pc_for_50, 'cumulative proportion'], color = 'grey')
plt.vlines(pc_for_75, ymin = 0, ymax = temp.loc[temp.PC == pc_for_75, 'cumulative proportion'], color = 'grey')


plt.plot(temp.PC, temp['cumulative proportion'])
# plt.plot(temp.loc[temp.PC < 3, 'PC'], temp.loc[temp.PC < 3,'cumulative proportion'], color = 'red')
# plt.xscale('log')
plt.savefig('DemoPCAScreePlot.svg',dpi=350)



# %%

# %%

# %%

# %%
# trainSamplesGBS = phenotypeGBS.shape[0] # <- replaced
trainFeaturesGenoGBS = PCParent.drop(columns = 'Taxa').shape[1]

# %%
addInGBSCodes = phenotypeGBSForIndices.loc[:, ['Pedigree', 'F_MatchBest', 'M_MatchBest', 'ExperimentCode', 'Year']].drop_duplicates()

addInGBSCodes.loc[:, 'Year'] = addInGBSCodes.loc[:, 'Year'].astype(str)

# %%
temp = phenotype.merge(addInGBSCodes, how = 'left')

# %%
# get all the indices that need to be removed (They contain nans in the genomic data)
mask = ((temp.F_MatchBest.isna()) | (temp.M_MatchBest.isna())
   ) | ((temp.F_MatchBest == '') | (temp.M_MatchBest == ''))

# We'll use this at the end.
rmTheseIndices = list(temp.loc[mask, ].reset_index()['index'])

# %%
temp

# %%
# make sure that everthing here is in the pca
F_MatchNotInPCA = pd.DataFrame(temp.F_MatchBest.drop_duplicates()).rename(columns = {'F_MatchBest':'Taxa'})
M_MatchNotInPCA = pd.DataFrame(temp.M_MatchBest.drop_duplicates()).rename(columns = {'M_MatchBest':'Taxa'})
presentTaxa = pd.DataFrame(PCParent.Taxa).assign(InPCA = True)

F_MatchNotInPCA = pd.merge(F_MatchNotInPCA, presentTaxa, how = 'outer')
M_MatchNotInPCA = pd.merge(M_MatchNotInPCA, presentTaxa, how = 'outer')

print("The following are not available in PCA for `F_MatchBest`")
mask = (F_MatchNotInPCA.InPCA != True) & (F_MatchNotInPCA.Taxa.notna())
print(F_MatchNotInPCA.loc[mask, ])
nParentsMissing = F_MatchNotInPCA.loc[mask, ].shape[0]
if nParentsMissing != 0:
    raise Exception('There should be no parents that are absent in the PCA but '+str(nParentsMissing)+' were found.')
    
print("\n")
print("The following are not available in PCA for `M_MatchBest`")
mask = (M_MatchNotInPCA.InPCA != True) & (M_MatchNotInPCA.Taxa.notna())
print(M_MatchNotInPCA.loc[mask, ])
nParentsMissing = M_MatchNotInPCA.loc[mask, ].shape[0]
if nParentsMissing != 0:
    raise Exception('There should be no parents that are absent in the PCA but '+str(nParentsMissing)+' were found.')

# %%
# Set up np arrays 
G = np.zeros((trainSamples, trainFeaturesGenoGBS))


# %%
# It seems that we're good on the matching front. 
# Despite dropping some taxa without table values in tassel we have agreement
# between the non-na samples and the pca matching samples.  

# What's the difference between phenotype gbs and temp?
pGBS_matchedDf = check_n_obs_matching_pca(
    inputdf = phenotypeGBS,
    outPCAPath = '../data/raw/G2FGenomes/pca_in_out3/PCs.txt')

temp_matchedDf = check_n_obs_matching_pca(
    inputdf = temp,
    outPCAPath = '../data/raw/G2FGenomes/pca_in_out3/PCs.txt')

# %%

# %%
# Use the merge method to get he right indexing without looping
Gf = pd.DataFrame(temp.loc[:, 'F_MatchBest']).rename(columns = {'F_MatchBest':'Taxa'})
Gf = Gf.merge(PCParent, how = 'left')

Gm = pd.DataFrame(temp.loc[:, 'M_MatchBest']).rename(columns = {'M_MatchBest':'Taxa'})
Gm = Gm.merge(PCParent, how = 'left')

# %%
# Average the female and male parent's matricies to get a hybrid's value
Gf = Gf.drop(columns =['Taxa']).to_numpy()
Gm = Gm.drop(columns =['Taxa']).to_numpy()
G = ((Gf + Gm) /2)

# %%
# Check that the rows that we expect to have no entries do in fact have none.
G[96136, 0:10]

# %%
# Just for fun, let's look at the loadings

# plt.scatter(G[:, 0], G[:, 1])

Gplt = pd.DataFrame(G)


import plotly.express as px
fig = px.scatter_3d(Gplt, x=0, y=1, z=2, opacity = 0.03)

zoomOut = 1.5 
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=zoomOut, y=zoomOut, z=zoomOut)
)

fig.update_layout(scene_camera=camera,
                 scene=dict(
        xaxis_title='PC1',
        yaxis_title='PC2',
        zaxis_title='PC3',
    ))
fig.write_image("DemoPCASimOffspring.svg")

fig.show()



# %%
# How many parents were we unable to find in the parents file?
# How many were expected?
# has Pedigree
hasPedigreeCount = temp.Pedigree.notna().sum()

# has both matches found.
mask = ((temp.F_MatchBest.notna()) & (temp.M_MatchBest.notna()))
hasMatchesCount = temp.loc[mask].shape[0]

# both matches are actually in PCParent.Taxa
checkMatchesInTaxa = temp.loc[mask].copy()
checkMatchesInTaxa

part1 = pd.DataFrame(PCParent.loc[:, 'Taxa']).rename(columns = {'Taxa':'F_MatchBest'}).merge(checkMatchesInTaxa, how = 'left')
part2 = pd.DataFrame(PCParent.loc[:, 'Taxa']).rename(columns = {'Taxa':'M_MatchBest'}).merge(checkMatchesInTaxa, how = 'left')

hasBothTaxaCount = part1.merge(part2, how = 'inner').shape[0]

print('Entries with non-NA pedigree:        ', hasPedigreeCount, 
     '\nEntries with codes expected to match:', hasMatchesCount,
     '\nEntries with codes that _do_ match:  ', hasBothTaxaCount, 'i.e. non-qced')

# How many entries are in PCParent.Taxa but not in the dataset?

# %%
ParentsInData = pd.DataFrame({'ObsParents': (list(temp.M_MatchBest)+list(temp.F_MatchBest))}).drop_duplicates()

ParentsNotInPC =   [entry for entry in list(ParentsInData.ObsParents) if entry not in list(PCParent.Taxa)]
ParentsNotInData = [entry for entry in list(PCParent.Taxa) if entry not in list(ParentsInData.ObsParents)]

print('Unique codes not in PCA: ', len(ParentsNotInPC), '/', len(list(ParentsInData.ObsParents)),
      '\nUnique codes not in Data ', len(ParentsNotInData), '/', len(list(PCParent.Taxa))
     )

# %%
G

# %%
# this is to ensure we have the right number of non matched pedigees in the BLUP code.
G_PCA_Parents = temp.loc[:, ['F_MatchBest', 'M_MatchBest']]

mask = (G_PCA_Parents.F_MatchBest.isna()) | (G_PCA_Parents.M_MatchBest.isna())

print(
    'Contains at least one NA',
    G_PCA_Parents.loc[mask, ].shape[0], 
    '\nContains at least no NA ',
 G_PCA_Parents.loc[~mask, ].shape[0], 
    '\nAll Entries             ',
 G_PCA_Parents.shape[0])

# %%
# how many na's are in the PCA
np.isnan(G).sum()
# 29421710


# %%
G_pca = G.copy()

# %%
# G_PCA_Parents.to_csv('../data/processed/G_PCA_Parents.csv')

# np.save('../data/processed/G_PCA_1.npy', G)
# phenotypeGBS.to_pickle('../data/processed/phenotype_with_GBS_for_keras.pkl')


# %%

# %% [markdown]
# # Reset data object state:
#
# Splits were generated originally with the parents as floats (no geno pca)
# We'll maintain this and winnow down the sets post hoc.

# %%
Y = np.load('../data/processed/Y.npy')
G = np.load('../data/processed/G.npy')
S = np.load('../data/processed/S.npy')
W = np.load('../data/processed/W.npy')

# %%
[entry.shape for entry in [Y, G, G_pca, S, W]]

# %%
phenotype_reload = phenotype.copy()

# %% [markdown]
# # Create Test/Train/Validation Sets

# %% [markdown]
# ## Get Location Groupings

# %%
temp = phenotype.loc[:, ["ExperimentCode", "Year"]].drop_duplicates()

temp
# effectively the same site:


# %%
# this was defined in 'narrow.py'
gpsGrouping = json.load(open("../data/interim/gpsGrouping.txt"))
# gpsGrouping

# %%
# TODO move these altered groupings back to `narrow`. then rerun to make sure it doesn't add errors. 

gpsGrouping = {'AZI1': ['AZH1', 'AZI1  AZI2'],
 'GAH1': ['GAI2'],
 
 'IAH1': ['IAH1 IAI1', 'IAH1a', 'IAH1b', 'IAH1c', 'IAH3', 'IA(?)3', 'IAH4', 'IA(H4)'], # <-
 
 'IAH2': ['IA(?)2'],
 

 'ILH1': ['ILH1  ILI1  ILH2', 'ILH1 ILI1'],
 'INH1': ['INH1  INI1', 'INH1 INI1'],
 'KSH1': ['KSH1  KSI1', 'KSH2', 'KSH3'],
 'MNH1': ['MN(?)1', 'MNI2'],
 
 'MOH1': ['MOH1  MOI1  MOH2  MOI2', 'MOH1 MOI1', 'MOH2', 'MOH2 MOI2 MOI3'], # <-
 
 'NEH1': ['NEH1  NEH4', 'NEH1 NEI1', 'NEH4', 'NEH4-NonIrrigated'], # <-
 
 'NEH3': ['NEH3-Irrigated'],
 'NYH1': ['NY?', 'NYH1  NYI1', 'NYH1 NYI1', 'NYI1', 'NYH2' , 'NYH3', 'NYH3  NYI2', 'NYH4'], # <- 
 
 'PAI1': ['PAI1  PAI2'],
 'SDH1': ['SDI1'],
 'TXH1': ['TXH1  TXI1  TXI2'],
 'TXH2': ['TXH2  TXI3', 'TXI3'],
 
 
 'TXH1-Condition': ['TXH1-Dry', 'TXH1-Early', 'TXH1-Late'], # <- 
  
 'WIH1': ['WIH1  WII1', 'WIH1 WII1', 'WIH2  WII2', 'WIH2']} # <- 

# %%
temp['ExperimentCodeGroup'] = ''

# apply groupings:
for key in gpsGrouping.keys():
    for value in gpsGrouping[key]:
        if temp.loc[temp.ExperimentCode == value, ].shape[0] > 0:
            print('Updating', value, temp.loc[temp.ExperimentCode == value, ].shape)
            temp.loc[temp.ExperimentCode == value, 'ExperimentCodeGroup'] = key
    
# add groupings that are one to one (i.e. not in dictionary)
temp.loc[temp.ExperimentCodeGroup == '', 'ExperimentCodeGroup'] = temp.loc[temp.ExperimentCodeGroup == '', 'ExperimentCode'] 

# %%
# we'll use this later.
expCodeGroupDF = temp.copy()

# %%
write_me_out = expCodeGroupDF.drop(columns = 'Year').drop_duplicates().reset_index().drop(columns = 'index')
write_me_out.to_csv('./z_expCode_to_expCodeGroup.csv')

# %%
temp2 = pd.DataFrame(temp.ExperimentCodeGroup.drop_duplicates()
                   ).rename(columns = {'ExperimentCodeGroup':'ExperimentCode'}
                   ).merge(metadata.loc[:, ['ExperimentCode', 'ExpLon', 'ExpLat']].drop_duplicates(), how='left'
                   ).groupby(['ExperimentCode']).agg(ExpLon = ('ExpLon', 'mean'),
                                                     ExpLat = ('ExpLat', 'mean')
                   ).reset_index()

# %%
threshold = 0.5 # this is in degrees so 1 ~== 70 miles

nrows = temp2.shape[0]
for i in range(nrows):
    if i+1 != nrows:
        for j in range(i+1, nrows):
#             print(i, j)
            dist = np.sqrt(((temp2.loc[i, 'ExpLon'] - temp2.loc[j, 'ExpLon'])**2
                           ) + ((temp2.loc[i, 'ExpLat'] - temp2.loc[j, 'ExpLat'])**2))

            if dist < threshold:
                print('dist:', round(dist, 3), temp2.loc[i, 'ExperimentCode'], temp2.loc[j, 'ExperimentCode'])
                
print('If no values printed then there are no experiment groups within the threshold of', threshold, 'degrees.')

# %%
plt.scatter(temp2.ExpLon, temp2.ExpLat)

# %% [markdown]
# # Part 3

# %%
# Choose train/test/validation

# %%
# Here's an overview of the object we're using to record acceptable train/test/validate splits.
# We save just the index numbers to make it easy to pull out the relevant entries later.

# indexDictList = [
#     {'Train': [124, 125, 126, ... ], 
#      'Validate': [8639, 8640, 8641, ... ], 
#      'Test': [876, 877, 878, ... ]
#     },     
#     {...},
#     ...
# ]

# %% [markdown]
# ## Generate Test/Training Splits

# %% [markdown]
# Restrict data to those samples containing genotypes

# %%
# Condition on genomic data so that the counts are correct.
discard = phenotype.reset_index().rename(columns = {'index':'BackupIndex'})
discard['Year'] = discard['Year'].astype('int')
phenotypeGBS['Year'] = phenotypeGBS['Year'].astype('int')

print(discard.shape)
discard = discard.merge(phenotypeGBS, how = 'left')
print(discard.shape)
phenotype = discard.loc[discard.HasGenomes == True]

# %% [markdown]
# Check that the index we'll use (`BackupIndex`) retrieves the expected data from the numpy arrays

# %%
# We want to be crystal clear that what we're using is good to go. Here we check to make sure that we 
# can use the backup index 
phenotype.loc[:, 'GrainYield'] = (phenotype.loc[:, 'GrainYield'].astype(float))

def _check_yields_match(i):
    in_p = phenotype.loc[i, 'GrainYield']
    in_y = Y[phenotype.loc[i, 'BackupIndex']]
    return(in_p == in_y)

matches = [_check_yields_match(i) for i in list(phenotype.loc[:, 'BackupIndex'])]

assert (False in matches) == False, 'There were one or more disagreeing y values!'
print("There _NO_ values of BackupIndex found which have disagreeing y values.")

# %% [markdown]
# Incorporate the distance based experiment code grouping (this will result in a more conservative estimate of the model's effectiveness by keeping more similar groups in the same split. 

# %%
# for each year exp.code get n from phenotype
expCodeGroupDF['Year'] = pd.DataFrame(expCodeGroupDF['Year']).astype(int)
p_w_groupings = phenotype.merge(expCodeGroupDF)

# this will ensure that we're working off of site(distance grouped site) x year groups ---|
#                                                                                         v
p_w_groupings['ExperimentCodeGroup'] = p_w_groupings['ExperimentCodeGroup'
                                                    ] +'_'+ p_w_groupings['Year'].astype(str) 

# %%
obsCounts = p_w_groupings.loc[:, ['ExperimentCodeGroup', 'ExperimentCode', 'Year', 'GrainYield']
                        ].groupby(['ExperimentCodeGroup', 'ExperimentCode', 'Year']
                        ).count(
                        ).reset_index(
                        ).rename(columns = {'GrainYield':'Count'})


# %%

# %%
def _mk_groupwise_test_train_split(
    obsCounts,
    max_test_percent = .15,
    min_test_percent = .1,
    min_total_obs = 40000,
    max_cycle = 30,
    randSeed = 16565434,
    always_return_grouping = False,
    print_updates = False
):
#     obsCounts = obsCounts.reset_index().rename(columns = {'index':'i_ExperimentCodeGroup'})
    
    expcodegroup_idx = obsCounts.loc[:, 'ExperimentCodeGroup'].drop_duplicates(
                        ).reset_index(
                        ).drop(columns=['index']).reset_index(
                        ).rename(columns={'index': 'i_ExperimentCodeGroup'})

    obsCounts = expcodegroup_idx.merge(obsCounts)

    # set or reset usable counts:
    obsCounts['Usable'] = obsCounts['Count']
    obsCounts['Partition'] = 'Train'

    # print(randSeed)
    rng = np.random.default_rng(randSeed)

    possibleSites = list(obsCounts.ExperimentCodeGroup.drop_duplicates())
    testSites = []

    for cycle in range(max_cycle):
        # Draw a site, add it to test list, remove from possible sites
        add_testSite = rng.choice(possibleSites, 1)[0]
        testSites.append(add_testSite)
        possibleSites = [entry for entry in possibleSites if entry not in testSites]

        # update test/train partition assignments
        partitionSites = obsCounts.loc[:, ['ExperimentCodeGroup', 'Partition']].drop_duplicates()
        partitionSites['Partition'] = ['Test' if entry in testSites else 'Train' for entry in list(partitionSites.ExperimentCodeGroup)]

        # merge partitions into obs
        obsCounts = obsCounts.drop(columns = ['Partition']).merge(partitionSites)
        # update usable number (no group sitexyear has more obs than the smallest group in test.)
        min_test_obs = np.min(obsCounts.loc[obsCounts.Partition == 'Test', ['Count']])[0]
        mask = (obsCounts['Usable'] > min_test_obs)
        obsCounts.loc[mask, 'Usable'] = min_test_obs

        # update summary statistics:
        summary_stats = obsCounts.groupby(['Partition']).agg(Count = ('Usable', 'sum')).reset_index()
        summary_stats['Total'] = sum(summary_stats['Count'])
        summary_stats['Percent'] = summary_stats['Count'] / summary_stats['Total']

        # check if loop should exit:
        # check if we should return the assignments:
        current_pr_test = summary_stats.loc[summary_stats.Partition == "Test", 'Percent'][0] 
        current_total   = summary_stats.loc[summary_stats.Partition == "Test", 'Total'][0]

        if print_updates:
            print("Cycle ", cycle, "\tTest %:", round(100*current_pr_test, 2), "\tTotal:", current_total)

        above_obs_thresh  = (current_total > min_total_obs)
        above_min_test_pr = (current_pr_test > min_test_percent)
        below_max_test_pr = (current_pr_test < max_test_percent)

        # should the loop exit?
        if above_min_test_pr:
            break

    return_grouping_df = False
    if always_return_grouping:
        return_grouping_df = True
    else:
        return_grouping_df = (above_obs_thresh & below_max_test_pr & above_min_test_pr)

    if return_grouping_df:
        obsCounts['RandSeed'] = randSeed
        obsCounts['TrainPct'] = 1-current_pr_test
        obsCounts['TestPct']  = current_pr_test
        obsCounts['TotalObs'] = current_total

        return(obsCounts)
    else:
        return(np.nan)


# %% [markdown]
# Generate several possible splits. View the observation number of each

# %%
rng2 = np.random.default_rng(1554542341)

split_list = [_mk_groupwise_test_train_split(
    obsCounts,
    max_test_percent = .15,
    min_test_percent = .1,
    min_total_obs = 40000,
    max_cycle = 30,
    randSeed = round(rng2.uniform(0, 1)*100000)) for i in range(10)]

# %%
split_list = [entry for entry in split_list if type(entry) != float] # drop all nans

# %%
obs_in_split_list = [e.TotalObs[0] for e in split_list]

print("Total Observations in splits:", obs_in_split_list)

# %% [markdown]
# Next we need to verify that:
# 1. we can merge groupings into phenotype
# 1. groupings are sensible
# 1. groupings can be extracted 

# %%

# %%
# Select a random index from the list, run tests on it. 
random_index = np.random.choice([i for i in range(len(split_list))])
print('Testing index', random_index, '\n')

temp = phenotype.merge(split_list[random_index])

## Groups in only test or train? ====
test = temp.loc[:, ['ExperimentCodeGroup', 'Partition']
        ].drop_duplicates(
        ).groupby(['ExperimentCodeGroup'
        ]).count().reset_index()
assert not True in list(test.Partition >1 ), "An ExperimentCodeGroup in train and test."
print("No ExperimentCodeGroup in train and test.")

test = temp.loc[:, ['i_ExperimentCodeGroup', 'Partition']
        ].drop_duplicates(
        ).groupby(['i_ExperimentCodeGroup'
        ]).count().reset_index()
assert not True in list(test.Partition >1 ), "An i_ExperimentCodeGroup in train and test."
print("No i_ExperimentCodeGroup in train and test.")

## Exactly one ExperimentCodeGroup per i_ExperimentCodeGroup and reverse?
test = temp.loc[:, ['ExperimentCodeGroup', 'i_ExperimentCodeGroup']
        ].drop_duplicates(
        ).groupby(['ExperimentCodeGroup'
        ]).count().reset_index()
assert not True in list(test.i_ExperimentCodeGroup != 1 ), "Not exactly 1 i_ExperimentCodeGroup per ExperimentCodeGroup."
print("Exactly 1 i_ExperimentCodeGroup per ExperimentCodeGroup.")

test = temp.loc[:, ['i_ExperimentCodeGroup', 'ExperimentCodeGroup']
        ].drop_duplicates(
        ).groupby(['i_ExperimentCodeGroup'
        ]).count().reset_index()
assert not True in list(test.ExperimentCodeGroup != 1 ), "Not exactly 1 ExperimentCodeGroup per i_ExperimentCodeGroup"
print("Exactly 1 ExperimentCodeGroup per i_ExperimentCodeGroup.")


# there can be multiple ExperimentCode s per ExperimentCodeGroup but the converse should only be true if 
# grouping includes year..
test = temp.loc[:, ['ExperimentCodeGroup', 'ExperimentCode']
        ].drop_duplicates(
        ).groupby(['ExperimentCodeGroup'
        ]).count().reset_index()
assert not False in list(test.ExperimentCode >= 1 ), "Not at least one ExperimentCode per ExperimentCodeGroup."
print("At least one ExperimentCode per ExperimentCodeGroup.")

test = temp.loc[:, ['ExperimentCode', 'Year', 'ExperimentCodeGroup']
        ].drop_duplicates(
        ).groupby(['ExperimentCode', 'Year'
        ]).count().reset_index()
assert not False in list(test.ExperimentCodeGroup == 1 ), "Not exactly one ExperimentCodeGroup for each ExperimentCode x Year combination"
print("ExperimentCode x Year maps to exactly one ExperimentCodeGroup")

print('\nAll Tests Passed.')


# %% [markdown]
# If tests have passed then we can transform the list into a json index. 

# %%
# ii = 1
# # for ii in range(len(split_list)):
# print(ii)

# ph_df = phenotype
# split_df = split_list[ii]
# randSeed = 98798698
# downsample = True

def _train_test_transform_to_dict(
    ph_df = phenotype,
    split_df = split_list[0],
    randSeed = 98798698,
    downsample = True
):

    # combine the partition information into the phenotype df.
    # then downsample if needed and add to output dictionary
    rng = np.random.default_rng(randSeed)
    temp = ph_df.merge(split_df)

    """
    Note: Here we're doing something a little unintuitive. Since we're constraining the minimum number of
    observations by site*year there can be different numbers of samples in one ExperimentCodeGroup (if it
    has multiple sites within it). So we need to included ExperimentCode and iterate over that within 
    a loop for i_ExperimentCodeGroup. Because i_ExperimentCodeGroup included year, the same 
    ExperimentCode may be present in multiple i_ExperimentCodeGroup s. 
    """
    temp = temp.loc[:, ['BackupIndex', 'i_ExperimentCodeGroup', 'ExperimentCode', 'Usable', 'Partition']]


    tempDict = {'Train':[], 
                 'Test':[], 
          'TrainGroups':[], 
           'TestGroups':[]}

    # ith_group = 0
    for ith_group in list(temp.i_ExperimentCodeGroup.drop_duplicates()):

        for jth_site in list(temp.loc[temp.i_ExperimentCodeGroup == ith_group, 'ExperimentCode'].drop_duplicates()):
            mask = (temp.i_ExperimentCodeGroup == ith_group) & (temp.ExperimentCode == jth_site)

        #     usable    = temp.loc[temp.i_ExperimentCodeGroup == ith_group, 'Usable'].drop_duplicates()
        #     partition = temp.loc[temp.i_ExperimentCodeGroup == ith_group, 'Partition'].drop_duplicates()
            usable    = temp.loc[mask, 'Usable'].drop_duplicates()
            partition = temp.loc[mask, 'Partition'].drop_duplicates()
        #     # confirm there's only one train/test label to apply and that there's only one downsampling number to use
            assert 1 == len(usable)
            assert 1 == len(partition)

        #     # get and randomize npy indices
        #     ith_idxs  = list(temp.loc[temp.i_ExperimentCodeGroup == ith_group, 'BackupIndex'])
            ith_idxs  = list(temp.loc[mask, 'BackupIndex'])
            rng.shuffle(ith_idxs)

            assert list(partition)[0] in ['Train', 'Test'], 'partition is not in "Train" or "Test"'

            if downsample:    
                if list(partition)[0] == 'Train':
                    tempDict['Train'].extend(ith_idxs[0:int(usable)])
                    tempDict['TrainGroups'].extend([ith_group for i in range(int(usable))])    
                else:
                    tempDict['Test'].extend(ith_idxs[0:int(usable)])
                    tempDict['TestGroups'].extend([ith_group for i in range(int(usable))])
            else: 
                if list(partition)[0] == 'Train':
                    tempDict['Train'].extend(ith_idxs)
                    tempDict['TrainGroups'].extend([ith_group for i in range(len(ith_idxs))])    
                else:
                    tempDict['Test'].extend(ith_idxs)
                    tempDict['TestGroups'].extend([ith_group for i in range(len(ith_idxs))])            

    return(tempDict)


# _train_test_transform_to_dict(
#     ph_df = phenotype,
#     split_df = split_list[0],
#     randSeed = 98798698,
#     downsample = True
# )

# %% [markdown]
# Test that the restructured dictionaries have the expected number of observations:

# %%
for input_split in split_list:
    res = _train_test_transform_to_dict(
        ph_df = phenotype,
        split_df = input_split,
        randSeed = 98798698,
        downsample = True
    )

    # Easy checks -- do the lengths match up within the dict?
    assert len(res['Train']) == len(res['TrainGroups']), 'Train and TrainGroups are different lengths'
    assert len(res['Test']) == len(res['TestGroups']), 'Test and TestGroups are different lengths'

    # Check if the total number of entries matches what we expect
    assert len(res['Train'])+len(res['Test']) == int(input_split['TotalObs'].drop_duplicates()
                               ), 'Total expected observations and observed obserations differ'

    # Check if the total number of entries for train/test match what we expect
    expected_totals = input_split.groupby(['Partition']).agg(total = ('Usable', 'sum')).reset_index()
    assert len(res['Train']) == int(expected_totals.loc[expected_totals.Partition == 'Train', 'total']
                                  ), 'Total observations in Train do not match expectations.'

    assert len(res['Test']) == int(expected_totals.loc[expected_totals.Partition == 'Test', 'total']
                                  ), 'Total observations in Test do not match expectations.'
    
print("tests passed!")

# %%

# %%

# %%

# %%

# %% [markdown]
# Apply across the board

# %%
# Downsampling: 
rng3 = np.random.default_rng(154545238548)
dict_list_downsample = [_train_test_transform_to_dict(
    ph_df = phenotype,
    split_df = entry,
    randSeed = round(rng3.uniform(0, 1)*100000),
    downsample = True) for entry in split_list]

# No downsampling:
rng3 = np.random.default_rng(154545238548)
dict_list_unbalanced = [_train_test_transform_to_dict(
    ph_df = phenotype,
    split_df = entry,
    randSeed = round(rng3.uniform(0, 1)*100000),
    downsample = False) for entry in split_list]

# %%

# %%
for i in tqdm.tqdm(range(len(dict_list_downsample))):
    for value in ['Train', 'Test']:
        downsample = dict_list_downsample[i][value]
        unbalanced = dict_list_unbalanced[i][value] 

        # we expect that unbalanced should contain downsample but not the other way around. 
        # downsample = dict_list_downsample[0]['Test']
        # unbalanced = dict_list_unbalanced[0]['Test'] 
        assert not False in [True if entry in unbalanced else False for entry in downsample
                            ], 'entries in downsample not in unbalanced'

        assert False in [True if entry in downsample else False for entry in unbalanced
                        ], 'No entries in unbalanced that aren\'t also in downsample'
print('tests passed!')

# %%
# check indices against genomic data
# np.isnan(G).sum()

"""
Testing:
1. Are there NaN in the genomic data that should not be so?
2. Do the groups match up with groups in phenotype correctly?
"""

for i in range(len(dict_list_unbalanced)):
    all_idx = dict_list_unbalanced[i]['Train']+dict_list_unbalanced[i]['Test']
    print("Index # ---------", i)
    print("Total indices ---", len(all_idx))
    print("+ctrl NaNs in G -", np.isnan(G_pca).sum())
    print("   In selection -", np.isnan(G_pca[all_idx]).sum())
    print("")
    assert 0 == np.isnan(G_pca[all_idx]).sum(), 'Halted if there were nans found.'

# %%

# %%
# save out:
# indices
with open('../data/processed/indexDictList_syr.txt', 'w') as convert_file:
     convert_file.write(json.dumps(dict_list_downsample))
        
with open('../data/processed/indexDictList_sy.txt', 'w') as convert_file:
     convert_file.write(json.dumps(dict_list_unbalanced))
        
# quick ref for index number
with open('../data/processed/obs_number_syr.txt', 'w') as convert_file:
     convert_file.write(json.dumps([len(e['Train'])+len(e['Test']) for e in dict_list_downsample]))
        
with open('../data/processed/obs_number_sy.txt', 'w') as convert_file:
     convert_file.write(json.dumps([len(e['Train'])+len(e['Test']) for e in dict_list_unbalanced]))

# %%
# [e for e in list(phenotype.loc[:, "Pedigree"].drop_duplicates()) if re.match(".*GEMS.*", e)]
# looking at the groupings do we use any of the embargoed data?

# %%
f = open('../data/processed/indexDictList_syr.txt')       
dat = json.load(f)

# %%
indxtrain = dat[0]['Train']
indxtest = dat[0]['Test']
indxtest[0:3]

# %%
phenotype.shape

# %%
[e for e in list(phenotype.loc[indxtest+indxtrain, "Pedigree"].drop_duplicates()) if re.match(".*GEMS.*", e)]

# %%
