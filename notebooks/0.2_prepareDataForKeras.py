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

import json # for reading in dictionaries


import umap
import sklearn
from sklearn import mixture
from sklearn.metrics import silhouette_score, davies_bouldin_score,v_measure_score
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
kForGeno = 4
useGenoCluster = False # we decided that practically there won't be a situation where a crop's genetics are fully unknown or unknowable.
# Pragmatically, not trying to account for genetic grouping increases the available sample size considerably for training purposes

# Which method should be used in setting up test/train splits?
"""
Method 1 --- 
    Randomly draw site/year combinations and add them to the test set. 
    Remove matching sites from the possible training set. 
    Stop when target train/test split is reached.
    
Method 2 ---
    Randomly draw a given fraction of sites   
    Upsample or downsample observations within those sites to reach a target split.

"""
makeTestTrainSplitWith = 'Method2'

# %%
# read in production datasets
metadata  = pd.read_pickle('../data/processed/metadata.pkl') # for double checking expeirment code independence
phenotype = pd.read_pickle('../data/processed/phenotype.pkl')
weather   = pd.read_pickle('../data/processed/weather.pkl')
soil      = pd.read_pickle('../data/processed/soil.pkl')
# genoPCA   = pd.read_pickle('../data/processed/genoPCA.pkl')

phenotypeGBS = pd.read_pickle('../data/processed/phenotype_with_GBS.pkl')


# %%

# %%
# genoPCA = pd.read_table('../data/raw/G2FGenomes/pca_in_out/Eigenvectors_Imputed_mH_with_pr_Filter_Filter.txt')
# genoPCA2 = pd.read_table('../data/raw/G2FGenomes/pca_in_out/Eigenvalues_Imputed_mH_with_pr_Filter_Filter.txt')

# # plt.plot(genoPCA2.PC, 100*genoPCA2['cumulative proportion'])
# # plt.plot(genoPCA2.PC, 100*genoPCA2['proportion of total'])
# # plt.xlim((0, 20))

# # genoPCA

# %% code_folding=[]


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

# %%
# make any groupwise reductions here. For example, we're dropping onterio (ONH2)
phenotypeGBS = phenotypeGBS.loc[phenotypeGBS.ExperimentCode != 'ONH2', ]

# %%
phenotypeGBS= phenotypeGBS.reset_index().drop(columns = ['index'])

# %%
phenotypeGBS.groupby('Year').count()

# %%



# %%


# We'll use this later to pair up the indicies in phenotype with those in the reduced dataset (i.e. after we filter)
phenotypeGBSForIndices = phenotypeGBS.copy()
phenotypeGBS = phenotypeGBS.loc[phenotypeGBS.HasGenomes, ].reset_index().drop(columns = 'index').copy()

# %%

# %% [markdown] heading_collapsed=true
# ### make numeric pedigree

# %% hidden=true
temp = phenotype.loc[:, ['Pedigree', 'F', 'M']]
temp = temp.drop_duplicates().reset_index().drop(columns = ['index'])

possibleParents = pd.DataFrame(temp.loc[:, 'F']).merge(pd.DataFrame(temp.loc[:, 'M']).rename(columns = {'M': 'F'}), how = 'outer').drop_duplicates().reset_index().drop(columns = ['index'])

#####
# apply any desired family wise groupings here.
#####

# %% hidden=true
"""
The plan here is to use regexs to create familywise groupings. 
Some parents are themselves crosses so we'll split those and apply the regexs again.
After this is done looking at the unique values of the twice cleaned grand parents 
(where an inbred B73 is treated as B73xB73) act as a lookup table to convert the parents into the cleaned grand parents.

start with parent (F), F -> FAdj -> (FSplitL, FSplitL) -> (FSplitLAdj, FSplitLAdj)

After we have this for both parents we'll add the 0.25 into the corresponding grand parent and use that as the matrix for the G component.
"""

# %% hidden=true

# %% code_folding=[13] hidden=true
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
                      
            

# %% hidden=true
# possibleParents['F'] = possibleParents['FAdj']

possibleParents.loc[possibleParents.FAdj == '', 'FAdj'] = possibleParents.loc[possibleParents.FAdj == '', 'F']


# %% hidden=true
possibleParents['FSplitL'] = ''
possibleParents['FSplitR'] = ''
possibleParents

# %% hidden=true
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

# %% hidden=true
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

# %% hidden=true
# These twice processed names are what we'll use for the g matrix.
GenoGroups = list(semiProcessedGenoGroup['EntryAdj'].drop_duplicates())

GenoGroups[0:5]

# %% hidden=true
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

# %% hidden=true
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



# %% hidden=true
res.loc[:, 'Ready'] = res.loc[:, GenoGroups].sum(axis = 1)

res.loc[res.Ready == 1.0,  'Ready'] = True
res.loc[res.Ready != True, 'Ready'] = False 

res

# %% hidden=true
# --------------------------------------------------------------------------- #
# Clean up phenotype before writing (some of the genotype data is unmatched)
# --------------------------------------------------------------------------- #
phenotype = pd.DataFrame(res.loc[res.Ready == True, 'Pedigree']).merge(phenotype)

res = res.loc[res.Ready == True].drop(columns = ['Ready'])
numericPedigree = res



# %% hidden=true

# %% hidden=true

# %% [markdown] heading_collapsed=true
# # Write out dataset 0 (numeric pedigree)

# %% hidden=true
trainSamples = phenotype.shape[0]
trainTimeSteps = (75+1+212) # preplant + plant + postplant
# trainFeaturesWeather = 19
trainFeaturesWeather = weather.shape[1] - 3 # ExperimentCode	Year	Date
# trainFeaturesSoil = 22
trainFeaturesSoil = soil.shape[1] - 2 # ExperimentCode	Year
# trainFeaturesGeno = 370
trainFeaturesGeno = numericPedigree.shape[1]-1

# %% hidden=true
# Set up np arrays 
Y = np.zeros((trainSamples, 1))
G = np.zeros((trainSamples,                 trainFeaturesGeno))
S = np.zeros((trainSamples,                 trainFeaturesSoil))
W = np.zeros((trainSamples, trainTimeSteps, trainFeaturesWeather))

# %% hidden=true

# %% hidden=true
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


# %% hidden=true
# to make this easier later:
GNumNames = list(pd.DataFrame(phenotype.loc[:, 'Pedigree']).merge(numericPedigree).drop(columns = ['Pedigree']))

with open('../data/processed/GNumNames.txt', 'w') as convert_file:
      convert_file.write(json.dumps(GNumNames))

# %% hidden=true
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


# %% hidden=true
[x.shape for x in [Y, G, S, W]]

# %% hidden=true

# %% hidden=true
np.save('../data/processed/Y.npy', Y)
np.save('../data/processed/G.npy', G)
np.save('../data/processed/S.npy', S)
np.save('../data/processed/W.npy', W)

phenotype.to_pickle('../data/processed/phenotype_for_keras.pkl')


# %% [markdown]
# # Write out dataset 1 (G == PCA)

# %%
# making lemonade of lemons here we'll use the existing PCA transform and get the average value for each point.
PCParent = pd.read_table('../data/raw/G2FGenomes/pca_in_out3/PCs.txt', skiprows=2)

# %%


# %%

# %%
import plotly.express as px

fig = px.scatter_3d(PCParent, x='PC1', y='PC2', z='PC3', opacity=0.3)
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
plt.plot(temp.loc[temp.PC < 3, 'PC'], temp.loc[temp.PC < 3,'cumulative proportion'], color = 'red')
plt.xscale('log')


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
G_PCA_Parents.to_csv('../data/processed/G_PCA_Parents.csv')

np.save('../data/processed/G_PCA_1.npy', G)
phenotypeGBS.to_pickle('../data/processed/phenotype_with_GBS_for_keras.pkl')


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

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown] heading_collapsed=true
# ## Demo Possible Genotype Groupings

# %% hidden=true

# %% hidden=true
if useGenoCluster:
    # # center/scaling
    # inDataCol = list(inData)
    # inDataMus = list(range(len(inDataCol)))
    # inDataSds = list(range(len(inDataCol)))
    # for i in range(len(inDataCol)):
    #     col = inDataCol[i]
    #     inDataMus[i] = inData.loc[:, col].mean()
    #     inDataSds[i] = inData.loc[:, col].std()

    #     inData.loc[:, col] = ((inData.loc[:, col] - inDataMus[i]) / inDataSds[i])

    inData = genoPCA.drop(columns = ['GBS'])

# %% hidden=true
if useGenoCluster:
    # https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Clustering_metrics.ipynb

    possibleKs = range(2,12)

    gm_bic= []
    gm_score=[]
    for i in tqdm.tqdm(possibleKs):
        gm = mixture.GaussianMixture(n_components=i,n_init=10,tol=1e-3,max_iter=1000).fit(inData)
    #     print("BIC for number of cluster(s) {}: {}".format(i,gm.bic(inData)))
    #     print("Log-likelihood score for number of cluster(s) {}: {}".format(i,gm.score(inData)))
    #     print("-"*100)
        gm_bic.append(gm.bic(inData))
        gm_score.append(gm.score(inData))

# %% hidden=true
if useGenoCluster:
    pltBIC = pd.DataFrame(zip(possibleKs, 
        gm_bic
    ), columns = ['k', 'bic'])

# %% hidden=true
if useGenoCluster:
    plt.figure(figsize = (6, 4))

    plt.plot(pltBIC.k, pltBIC.bic)
    plt.scatter(x=pltBIC.k, y = pltBIC.bic)

    plt.grid(True)
    plt.title('Cluster Estimation Through GMM')
    plt.xlabel("Cluster Number")
    plt.ylabel("BIC score")
    plt.xticks(pltBIC.k)
    print()

# %% hidden=true

# %% hidden=true
if useGenoCluster:
    # https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Clustering-Dimensionality-Reduction/Clustering_metrics.ipynb

    km_scores= []
    km_silhouette = []
    vmeasure_score =[]
    db_score = []
    for i in possibleKs:
        km = sklearn.cluster.KMeans(n_clusters=i, random_state=0).fit(inData)
        preds = km.predict(inData)

        silhouette = silhouette_score(inData,preds)
        km_silhouette.append(silhouette)
        print("Silhouette score for number of cluster(s) {}: {}".format(i,silhouette))

# %% hidden=true
if useGenoCluster:
    pltSil = pd.DataFrame(zip(possibleKs, 
        km_silhouette
    ), columns = ['k', 'Silhouette'])

# %% hidden=true
if useGenoCluster:
    plt.figure(figsize = (6, 4))

    plt.plot(pltSil.k, pltSil.Silhouette)
    plt.scatter(x=pltSil.k, y = pltSil.Silhouette)

    plt.grid(True)
    plt.title('Cluster Estimation Through Kmeans + Silhouette')
    plt.xlabel("Cluster Number")
    plt.ylabel("Silhouette score")
    plt.xticks(pltBIC.k)
    print()

# %% hidden=true
if useGenoCluster:
    genoCross = phenotype.loc[:, ['Pedigree', 'GBS0', 'GBS1', 'GBS']]

    genoCross[['F', 'M']] = ''
    genoCross= genoCross.reset_index().drop(columns = 'index')
    splitPedigree = genoCross.Pedigree.str.split('/')

    # slow but works
    for i in tqdm.tqdm(range(len(splitPedigree)) ):
        genoCross.loc[i, ['F', 'M']] = splitPedigree[i] 

# %% hidden=true
if useGenoCluster:
    genoCross

# %% hidden=true

# %% hidden=true
if useGenoCluster:
    # get some sort of hclust based genotype grouping to exclude members in test data from training data:

    # How many parents are there?
    allParents = list(genoCross.F.drop_duplicates())
    allParents = allParents + [parent for parent in (genoCross.M.drop_duplicates()) if parent not in allParents]
    print(len(allParents), 'unique parents exist.\nThis includes parents a numeric flag (e.g. PHN11_PHW65_0455)')
    # another way to do this would be to disallow any parents to be shared between training and test.
    # this wouldn't account for similar genotypes or parents of the same line. 
    print(len(genoCross.GBS.drop_duplicates()), 'unique crosses')

    # wards clustering
    genoCluster = sklearn.cluster.AgglomerativeClustering(n_clusters = kForGeno).fit(genoPCA.drop(columns = ['GBS']))

    # creates a catch all cluster of 1826 entries
    # genoCluster = sklearn.cluster.OPTICS(min_samples = 5).fit(genoPCA.drop(columns = ['GBS']))
    # genoCluster = cluster.DBSCAN(eps = 1000.00000001, min_samples = 5).fit(genoPCA.drop(columns = ['GBS']))



# %% hidden=true
if useGenoCluster:
    # pd.concat([
    temp = pd.DataFrame(genoPCA.loc[:, 'GBS'])

    temp['Cluster'] = genoCluster.labels_
    # pd.DataFrame(genoCluster.labels_).rename(columns = {0:'Cluster'})
    #     ])

    list(temp.Cluster.drop_duplicates())
    print('Using', len(temp.Cluster.unique()), 'clusters.')

# %% hidden=true
if useGenoCluster:
    temp = genoCross.loc[:, ['F', 'M', 'GBS']].drop_duplicates().merge(temp)
    temp

# %% hidden=true

# %% hidden=true
if useGenoCluster:
    reducer = umap.UMAP()

# %% hidden=true
if useGenoCluster:
    X = genoPCA.drop(columns = 'GBS') 
    XScaled = X#StandardScaler().fit_transform(X) # <-- Note the visualizaton is using non scaled data.

    embedding = reducer.fit_transform(XScaled)
    embedding.shape

# %% hidden=true

# %% hidden=true
if useGenoCluster:
    np.random.seed(98709887)

    clusterUnique = list(temp.Cluster.unique())

    clusterUnique = pd.DataFrame(zip(
        clusterUnique,
        [mpl.colors.rgb2hex((np.random.random(), np.random.random(), np.random.random())) for clust in clusterUnique]
    )).rename(columns = {0:'ExperimentCode', 1:'color'})

# %% hidden=true
if useGenoCluster:
    pltDf= pd.DataFrame()
    pltDf['x'] = embedding[:, 0]
    pltDf['y'] = embedding[:, 1]

    pltDf['ExperimentCode'] = temp['Cluster']
    pltDf= pltDf.merge(clusterUnique)

# %% hidden=true
if useGenoCluster:
    figure(figsize=(8, 6), dpi=80)

    plt.scatter(
        pltDf['x'],
        pltDf['y'],
        color = pltDf['color']
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the\nPCA transformed genomes', fontsize=24)

# %% hidden=true
if useGenoCluster:
    pltDfCounts = pltDf.groupby('ExperimentCode').count().reset_index()
    pltDfCounts

# %% hidden=true
# temp

# %% hidden=true
if useGenoCluster:
    phenotype= phenotype.merge(temp, how = 'left').rename(columns = {'Cluster':'GenoCluster'})

# %% hidden=true

# %% hidden=true

# %% hidden=true

# %% hidden=true

# %% hidden=true

# %% hidden=true

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
# ## Randomly Generate Test/Training Splits

# %%
# if useGenoCluster != True:
#     print('ding') # let each observation be its own group so the test set construction is identical to not considering it at all
#     phenotype= phenotype.reset_index().rename(columns = {'index':'GenoCluster'})

# %%
phenotype

# %%
obsCounts = phenotype.loc[:, ['ExperimentCode', 'Year', #'GenoCluster', 
                              'GrainYield']
                         ].groupby(['ExperimentCode', 'Year'#, 'GenoCluster'
                                   ]
                         ).count(
                         ).reset_index().rename(columns = {'GrainYield':'Count'})

# for each year exp.code get n from phenotype
temp = expCodeGroupDF.merge(obsCounts, how = 'left')

# grouping by location clusters
temp = temp.groupby(['ExperimentCodeGroup', 'Year'#, 'GenoCluster'
                    ]).agg(total = ('Count', 'sum')).reset_index()
# obsCounts

# %%
consideredGroups = temp.copy()
consideredGroups



# %% [markdown]
# ## Method 1. Randomly add groups to the test set until the desired split is reached.
#
#
# This method works nicely but could introduce correlation into the test set. If a site with a large n is selected first, it is less likely that another large n site is selected as well (i.e. without going over).

# %% [markdown]
# ### Randomly generate test/training splits

# %% code_folding=[7]
def mk_test_train_split(
    percentTarget = 0.1,
    percentTolerance = 0.01,
    maxIter = 100,
    df = temp,
    randSeed = 3354135,
    **kwargs
):
    available = df.copy()
    available = available.reset_index()
    removed   = pd.DataFrame()
    test      = pd.DataFrame()
    percent = 0
    
    nTestList   = []
    nTrainList  = []
    percentList = []
    
    
#     #optional args
#     if 'verbose' not in kwargs.keys():
#         print('Defaulting to verbose.')
#         kwargs['verbose'] = True
#     if 'verbose' in kwargs.keys():
#         if kwargs['verbose'] not in [True, 'True']:
#             print('Boolean required.')
#             print('Defaulting to verbose.')
#             kwargs['verbose'] = True
#         else:
#             print('Set to non-verbose.')
#             kwargs['verbose'] = False
                

    if type(randSeed) != int:
        randSeed = np.random.choice(range(1, 10000))
        print('Seed must be of type int! Using', randSeed, '.')
    rng = np.random.default_rng(randSeed)
    
#     if kwargs['verbose'] == True:
    print('nTest', '\t', 'nTrain', '\t')

    for i in range(maxIter):
        if ((percent > (percentTarget-percentTolerance)) # close enough
           ) | (percent > percentTarget):                # over
            break
        else:
            # ithRow = rng.choice(range(0, len(available)))
            ithRow = rng.choice(list(available.index))
            # Update dataframes ----
            ## Add to test ====
            test = pd.concat([test, pd.DataFrame(available.loc[ithRow, :]).transpose()])

            experimentCodeGroup = available.loc[ithRow, 'ExperimentCodeGroup']

            ## move disqualified rows from available to removed ====
            ### The same genetics group can't be used in train and test ####
            ### The same site can't be used in train and test ####
            mask = ((available.ExperimentCodeGroup == experimentCodeGroup))
#             print("gc", available.loc[ithRow, 'GenoCluster'])
            removed = pd.concat([removed, available.loc[mask, ]])
            available = available.loc[~mask, ]

            ## remove from available ====
            available = available.loc[available.index != ithRow]

            # Update split summaries ----
            nTest       = test['total'].sum()
            nTrain = available['total'].sum()
            percent = nTest/nTrain
            
            nTestList   = nTestList+[nTest]
            nTrainList  = nTrainList+[nTrain]
            percentList = percentList+[percent]
            
#             if kwargs['verbose'] == True:
            print( nTest,  '\t',  nTrain,  '\t', round(percent, 2))

    return({'Train':available,
            'Test':test,
            'Removed':removed,
            'Seed':randSeed,
            'TestPercent':percent,
            'History':pd.DataFrame(zip(nTestList, nTrainList, percentList)
                   ).rename(columns={0:'Test', 1:'Train', 2:'Percent'}
                   ).reset_index()
           })

# %%
if makeTestTrainSplitWith == 'Method1':

    accumulator = pd.DataFrame()

    for i in range(1000):
        result = mk_test_train_split(
            percentTarget = 0.1,
            percentTolerance = 0.01,
            maxIter = 100,
            df = consideredGroups,
            randSeed = '',#3354135
            verbose = False
        )

        temp = result['History']
        temp['Seed'] = result['Seed']
    #     temp

        if accumulator.shape[0] == 0:
            accumulator = temp
        else:
            accumulator = pd.concat([accumulator, temp])

    accumulator

# %%

# %%
if makeTestTrainSplitWith == 'Method1':
    # demo spread of these results
    for seed in list(accumulator.Seed.drop_duplicates()):
        plt.plot(accumulator.loc[accumulator.Seed == (seed), 'index'], accumulator.loc[accumulator.Seed == (seed), 'Percent'], alpha = 0.3)

# %%
if makeTestTrainSplitWith == 'Method1':
    mask = (accumulator.Percent > 0.095) & (accumulator.Percent < 0.105)
    temp = accumulator.loc[mask, ]
    temp.head(3)

# %%
# ValueError: cannot reindex from a duplicate axis
# sns.kdeplot(
#     data=temp, x="Train", y="Percent",
#     fill=True, thresh=0, levels=100, #cmap="mako",
#     cmap = sns.color_palette("viridis", as_cmap=True),
#     cbar = True
# )

# %%
# ValueError: cannot reindex from a duplicate axis
# sns.kdeplot(data=temp, x="Train", y="Percent", color = 'red')

# %% [markdown]
# ### Filter from splits and convert into observation indices.

# %%
if makeTestTrainSplitWith == 'Method1':
    filteredRuns = temp.rename(columns = {'index':'iter'}).reset_index().drop(columns = 'index')
    
    filteredRuns.head(3)

# %%

# %%
if makeTestTrainSplitWith == 'Method1':
    # Get all the test/train from those that we know work.
    accumulatorList = [] #pd.DataFrame()

    for i in range(len(filteredRuns)):
        result = mk_test_train_split(
                percentTarget = 0.1,
                percentTolerance = 0.01,
                maxIter = (int(filteredRuns.loc[i, 'iter'])+1),
                df = consideredGroups,
                randSeed = int(filteredRuns.loc[i, 'Seed']),
                verbose = False
            )
        accumulatorList = accumulatorList + [result]

# %% code_folding=[6]
if makeTestTrainSplitWith == 'Method1':
    def mk_val_index_and_return_indices(
        randSeed = 65576186, # used in picking validation set repeatably
        targetValPercent = 0.05,
        df = phenotype.merge(expCodeGroupDF, how = 'left'),
        accumulatorList = accumulatorList,
        i =0
    ):

        tempTrain = accumulatorList[i]['Train']
        tempTrain = tempTrain.drop(columns = 'index').reset_index().drop(columns = 'index')

        tempTest = accumulatorList[i]['Test']
        tempTest = tempTest.drop(columns = 'index').reset_index().drop(columns = 'index')


        # choose validation js ----
        desiredObs = round(targetValPercent * tempTrain.total.sum())
        trainIndexs = [k for k in range(len(tempTrain))]

        rng = np.random.default_rng(randSeed)
        rng.shuffle(trainIndexs)

        trainIndexSummary = pd.DataFrame(zip(
            trainIndexs, 
            [tempTrain.loc[trainIndex, 'total'] for trainIndex in trainIndexs]), columns = ['TrainIndex', 'Total'])
        trainIndexSummary['CSum'] = np.cumsum(trainIndexSummary.Total)
        trainIndexSummary['Validate'] = False

        trainIndexSummary.loc[trainIndexSummary.CSum <= desiredObs, 'Validate'] = True
        # cover the edge case of the first being greater than the desired obs
        trainIndexSummary.loc[0, 'Validate'] = True

        validationJs = list(trainIndexSummary.loc[trainIndexSummary.Validate == True, 'TrainIndex'])


        df['Split'] = ''
        for j in range(len(tempTrain)):
            mask = (df.ExperimentCodeGroup == tempTrain.loc[j, 'ExperimentCodeGroup']
               ) & (df.Year == str(tempTrain.loc[j, 'Year'])
        #        ) & (df.GenoCluster == tempTrain.loc[j, 'GenoCluster']
               )
            if j in validationJs:
                df.loc[mask, 'Split'] = 'Validate'
            else:
                df.loc[mask, 'Split'] = 'Train'

        for j in range(len(tempTest)):
            mask = (df.ExperimentCodeGroup == tempTest.loc[j, 'ExperimentCodeGroup']
               ) & (df.Year == str(tempTest.loc[j, 'Year'])
        #        ) & (df.GenoCluster == tempTest.loc[j, 'GenoCluster']
               )
            df.loc[mask, 'Split'] = 'Test'

        df = df.reset_index()

        indexDict = {'Train': list(df.loc[df.Split == 'Train', 'index']),
         'Validate': list(df.loc[df.Split == 'Validate', 'index']),
         'Test': list(df.loc[df.Split == 'Test', 'index'])}

        return(indexDict)

# %%
if makeTestTrainSplitWith == 'Method1':

    accumulatorListLen = len(accumulatorList)

    rng = np.random.default_rng(354635434)
    randSeeds = [round(i) for i in list(rng.uniform(1, 100000, accumulatorListLen))]
    dfMerged = phenotype.merge(expCodeGroupDF, how = 'left')

    indexDictList = []
    for i in tqdm.tqdm(range(accumulatorListLen)):
        res = mk_val_index_and_return_indices(
            randSeed = randSeeds[i], # used in picking validation set repeatably
            targetValPercent = 0.05,
            df = dfMerged,
            accumulatorList = accumulatorList,
            i =0
        )
        indexDictList = indexDictList + [res]
        
        len(indexDictList)


# %%

# %%
# with open('../data/processed/indexDictList.txt', 'w') as convert_file:
#      convert_file.write(json.dumps(indexDictList))

# %%
# len(Y)


# %%

# %% [markdown]
# ## Method 2. Randomly select sites and then balance through down/up sampling.
#
#

# %%
consideredGroups

# %%
plt.scatter(consideredGroups.ExperimentCodeGroup, consideredGroups.Year, c = consideredGroups.total, linewidths=2)
plt.colorbar()
plt.xticks(rotation=90)
print()

# %%

# %%
# Strict Site
# here we're going to select based on sites and pull all the data across the years for that site.

df = consideredGroups#.groupby(['ExperimentCodeGroup']).agg(total = ('total', 'sum')).reset_index()
df



# Site

# %%
sitewiseCounts = df.groupby(
    ['ExperimentCodeGroup']).agg(
    minObs = ('total', 'min'),
    medObs = ('total', 'median'),
    maxObs = ('total', 'max')
    ).reset_index()

siteYearCounts = df.agg(
    minObs = ('total', 'min'),
    medObs = ('total', 'median'),
    maxObs = ('total', 'max')
    ).reset_index()

sitewiseCounts

siteYearCounts

# %%
# Alter phenotype for easier selection of experiment code groups.
temp = pd.DataFrame(phenotype.loc[:, 'ExperimentCode']).drop_duplicates()
temp['ExperimentCodeGroup'] = temp['ExperimentCode']
for key in gpsGrouping: #The dictionary from way back
    for value in gpsGrouping[key]:
        temp.loc[temp.ExperimentCode == value, 'ExperimentCodeGroup'] = key

        
phenotype = phenotype.merge(temp, how = 'left')
phenotype['ExperimentCode'] = phenotype['ExperimentCodeGroup']# added
phenotype.head()

# moved from below

# %% code_folding=[]

# start with x groups
# balance the observations in those x groups by site and year
# add groups until a desired threshold is hit

def mk_test_train_split_method2(
    df = consideredGroups,
    ExperimentSite = 'ExperimentCodeGroup',
    nSites = 3,
    testPercentRange = (0.09, 0.11),
    valPercentRange =  (0.05, 0.06),
    maxIter = 2,
    randSeed = 3354135,
    verbose = False
    
):
    # df = consideredGroups
    # ExperimentSite = 'ExperimentCodeGroup'
    # nSites = 3
    # testPercentRange = (0.09, 0.11)
    # valPercentRange =  (0.05, 0.06)
    # maxIter = 2
    # randSeed = 3354135

    # balanceBy = 'downsample' #upsample
    # Get Test Set -----------------------------------------------------------------
    possibleSites = df[ExperimentSite].drop_duplicates()

    if type(randSeed) != int:
        randSeed = np.random.choice(range(1, 10000))
        if verbose:
            print('Seed must be of type int! Using', randSeed, '.')
    rng = np.random.default_rng(randSeed)

    if verbose:
        print('--- Define Test Set ---\n')
    testPercent = 0
    for i in range(maxIter):
        if testPercent >= testPercentRange[0]:
            break
        else:
            if i == 0:
                # Draw initial test sites
                testSites = rng.choice(possibleSites, nSites)
            else:
                testSites = rng.choice(possibleSites, 1)

            # Update possible test sites
            possibleSites = [entry for entry in possibleSites if entry not in testSites]


            # update test/train index in dataframe
            df['Test'] = [True if site not in possibleSites else False for site in df[ExperimentSite]]

            # update metrics about test/train split
            # reset the usable total. It may change based on balancing differences.
            # i.e. we'll permit groups smaller than the smallest test group in the training data.
            df['useTotal'] = df['total']

            siteYearMin = df.loc[df.Test, 'total'].min()
            siteYearMax = df.loc[df.Test, 'total'].max()

            if verbose:
                print(siteYearMin, siteYearMax)

            # downsample test 
            df.loc[df.Test, 'useTotal'] = siteYearMin

            # balance down only those groups in training that are greater than the smallest group in test
            mask = ((~df.Test) & (df.total > siteYearMin))
            df.loc[mask, 'useTotal'] = siteYearMin

            # reporting
            splitSummary = df.groupby('Test').agg(totalObs = ('useTotal', 'sum')).reset_index()
            splitSummary['Percent'] = splitSummary['totalObs'] / splitSummary.totalObs.sum()

            if verbose:
                print('Iteration:', i)
                print(splitSummary)

            testPercent = list(splitSummary.loc[splitSummary.Test, 'Percent'])[0]

    # if test/train is set we can select a validation set --------------------------
    if verbose:
        print('--- Define Val. Set ---\n')
    # setup validation search
    temp = df.loc[~df.Test, ].copy()
    # temp = temp.groupby(['ExperimentCodeGroup']).agg(total = ('useTotal', 'sum')).reset_index()

    valPossibleSites = list(temp.loc[:, 'ExperimentCodeGroup'].drop_duplicates())
    valPossibleSites
    temp['valTotal'] = 0
    temp.loc[:, 'valTotal'] = temp.loc[:, 'useTotal']
    temp['Val'] = False

    valPercent = 0
    for i in range(maxIter):
        if valPercent >= valPercentRange[0]:
            break
        else:
            valSite = rng.choice(valPossibleSites, 1)
            valSite = valSite[0]

            temp.loc[temp[ExperimentSite] == valSite, 'Val'] = True

#             valSiteYearMin = temp.loc[temp.Val, 'useTotal'].min() # <- Don't reduce down the validation set. Using CV
            # balance validation set
#             temp.loc[temp.Val, 'valTotal'] = valSiteYearMin

            splitSummary = temp.groupby(['Val']).agg(totalObs = ('valTotal', 'sum')).reset_index()
            splitSummary['Percent'] = splitSummary['totalObs'] / splitSummary.totalObs.sum()

            if verbose:
                print('Iteration:', i)
                print(splitSummary)

            valPercent = list(splitSummary.loc[splitSummary.Val, 'Percent'])[0]


    # check if we're still okay ----------------------------------------------------
    df = df.merge(temp, how = 'outer')


    # validation will have missing values for the test set.
    df.loc[df.Val.isna(), 'Val'] = False

    df['Split']              = 'Train'
    df.loc[df.Test, 'Split'] = 'Test'
    df.loc[df.Val,  'Split'] = 'Val'


    df['Total'] = 0

    df.loc[df.Split == 'Train', 'Total'] = df.loc[df.Split == 'Train', 'useTotal']
    df.loc[df.Split == 'Test',  'Total'] = df.loc[df.Split == 'Test',  'useTotal']
    df.loc[df.Split == 'Val',   'Total'] = df.loc[df.Split == 'Val',   'valTotal']


    # internal check -- are the groups getting reduced correctly?
    # df.groupby(['Split']).agg(
    #     max = ('Total', 'max'),
    #     mean = ('Total', 'mean'),
    #     min = ('Total', 'min')).reset_index()


    overAllSummary = df.groupby(['Split']).agg(totalObs = ('Total', 'sum')).reset_index()
    overAllSummary['Percent'] = overAllSummary['totalObs'] / splitSummary.totalObs.sum()

    overAllSummary['TrainingPercent'] = 0
    mask = (overAllSummary.Split != 'Test')
    overAllSummary.loc[mask, 'TrainingPercent'] = overAllSummary.loc[mask, 'totalObs'] / overAllSummary.loc[mask, 'totalObs'].sum()

    if verbose:
        print('Overall\n')
        print(overAllSummary)
        print('\nTotal Obs:', overAllSummary.totalObs.sum())


    testPercent = list(overAllSummary.loc[overAllSummary.Split == 'Test', 'Percent'])[0]
    valPercent =  list(overAllSummary.loc[overAllSummary.Split == 'Val', 'TrainingPercent'])[0]


    if (testPercent >= testPercentRange[0]):
        if (testPercent <= testPercentRange[1]):
            if verbose:
                print('Test Percent within bounds.')
        else:
            if verbose:
                print('Test Percent too high.', round(testPercent, 2))        
    else:
        if verbose:
            print('Test Percent too low.', round(testPercent, 2))

    if (valPercent >= valPercentRange[0]):
        if (valPercent <= valPercentRange[1]):
            if verbose:
                print('Val. Percent within bounds.')
        else:
            if verbose:
                print('Val. Percent too high.', round(valPercent, 2))        
    else:
        if verbose:
            print('Val. Percent too low.', round(valPercent, 2))




    df.Total = df.Total.astype(int)

    return({'split': df.loc[:, ['ExperimentCodeGroup', 'Year', 'Split', 'Total']],
            'summary': overAllSummary,
            'randSeed': randSeed
           })

# %%
consideredGroups

# %%
# Randomly create possible splits ----------------------------------------------

totalTrials = 2000
testPercentRange = (0.09, 0.11)
# valPercentRange =  (0.05, 0.06)
valPercentRange =  (0.000001, 0.999999) # Altered to effectively remove constraint of validattion set.

minTestSize = 4000 # seems to be a cluster around 2000 and above 4000. Choose ~4k as cutoff?


# outputList = []
outputDict = {} # switched to dict to make retrival by seed number easier.
distSummary = pd.DataFrame()
rng = np.random.default_rng(564848)
randomSeeds = [rng.choice(range(0, 100000)) for i in range(totalTrials)]


for i in tqdm.tqdm(range(totalTrials)):
    randSeed = randomSeeds[i]

    res = mk_test_train_split_method2(
        df = consideredGroups,
        ExperimentSite = 'ExperimentCodeGroup',
        nSites = 3,
        testPercentRange = testPercentRange,
        valPercentRange =  valPercentRange,
        maxIter = 20, # 2 produces an even 100 observations but I intended this to be higher
        randSeed = int(randSeed) # must be int here or it will draw a random seed
        # randSeed = ''#3354135    
    )
    
    outputDict.update({res['randSeed']: res})
    
    temp = res['summary']
    temp['Seed'] = res['randSeed']

    if distSummary.shape[0] == 0:
        distSummary = temp
    else:
        distSummary = pd.concat([distSummary, temp])



# %%

# %%
# add indexing to find the successful splits -----------------------------------
# Flag passing test/val seed values
distSummary['TestOkay'] = False
mask = (distSummary.Split == 'Test'
       ) & (distSummary.Percent >= testPercentRange[0]
       ) & (distSummary.Percent <= testPercentRange[1])
distSummary.loc[mask, 'TestOkay'] = True


distSummary['ValOkay'] = False
mask = (distSummary.Split == 'Val'
       ) & (distSummary.TrainingPercent >= valPercentRange[0]
       ) & (distSummary.TrainingPercent <= valPercentRange[1])
distSummary.loc[mask, 'ValOkay'] = True

# Expand flag to all rows of a given seed value
seedOkayTest = list(distSummary.loc[distSummary.TestOkay, 'Seed'])
for seed in seedOkayTest:
    distSummary.loc[distSummary.Seed == seed, 'TestOkay'] = True
    

seedOkayVal = list(distSummary.loc[distSummary.ValOkay, 'Seed'])
for seed in seedOkayVal:
    distSummary.loc[distSummary.Seed == seed, 'ValOkay'] = True
    
    
# and post hoc add in a check for the size of the test group.
distSummary['TestSizeOkay'] = False
mask = (distSummary.Split == 'Test') & (distSummary.totalObs >= minTestSize)
distSummary.loc[mask, 'TestSizeOkay'] = True


# %%

# %%

# %%
temp = distSummary.loc[((distSummary.Split == 'Test') & (distSummary.TestOkay) & (distSummary.ValOkay)),]

plt.hist(temp.loc[:, 'totalObs'], bins = round((max(temp.totalObs) - min(temp.totalObs)) / 100))
print()

# %%

# %%

# %%
# TODO add in TestSizeOkay to the selection 

# Provide some summary information about the run -------------------------------
print('Out of', int(distSummary.shape[0]/3), 'trials with')
print('Test Range:   ', testPercentRange,
      '\nVal. Range:   ', valPercentRange,
      '\nMin Test obs: ', minTestSize, sep = '')

temp = distSummary.loc[distSummary.Split == 'Test', ['Seed', 'TestOkay', 'ValOkay', 'TestSizeOkay']
                      ].groupby(['TestOkay', 'ValOkay', 'TestSizeOkay']
                               ).count().reset_index()

temp['Percent'] = temp['Seed']
temp['Percent'] = temp['Percent']/sum(temp['Percent'])
temp['Percent'] = round(temp['Percent']*100, 2)

temp = temp.drop(columns = ['Seed'])
# temp = pd.pivot(temp, index = 'TestOkay', columns= 'ValOkay')
temp

# %%
# moved earlier and overwrote ExperimentCode so that the initial groupings apply to the groups not the indivdual sites.

# # Alter phenotype for easier selection of experiment code groups.
# temp = pd.DataFrame(phenotype.loc[:, 'ExperimentCode']).drop_duplicates()
# temp['ExperimentCodeGroup'] = temp['ExperimentCode']
# for key in gpsGrouping: #The dictionary from way back
#     for value in gpsGrouping[key]:
#         temp.loc[temp.ExperimentCode == value, 'ExperimentCodeGroup'] = key

        
# phenotype = phenotype.merge(temp, how = 'left')
# phenotype.head()

# %%
# Use the seed values to find the successful splits ----------------------------

mask = (distSummary.TestOkay & distSummary.ValOkay & distSummary.TestSizeOkay)

# sort to make sure that the ordering is identical across runs.
retrieveEntries = list((pd.DataFrame(distSummary.loc[mask, 'Seed']
                                    ).drop_duplicates(
                                    ).sort_values(['Seed']
                                    ).loc[:, 'Seed']))


# for each seed

# randomly draw indexes
# set up seed values up front for this.
rng = np.random.default_rng(1861834)
downsampleObsSeeds = [int(round(entry, 0)) for entry in rng.uniform(0, 1, len(retrieveEntries))*10000]



# Downsample groups where needed and -------------------------------------------
# Format into a list of dictionaries for future use ----------------------------
indexDictList = []
# each successful split
i = 0
for i in tqdm.tqdm(range(len(downsampleObsSeeds))):
    tempDict = {'Train': [],
                'Val': [], # <- this does not match the desired output so that we can use one string to get both the right location here and above
                           # We'll need to rename it.
                'Test': []
               }

    temp = outputDict[retrieveEntries[i]]['split']

    # pre make all the seed values we need.
    rng = np.random.default_rng(downsampleObsSeeds[i])
    tempSeeds = pd.DataFrame(
        {'Train': [int(round(entry, 0)) for entry in rng.uniform(0, 1, temp.shape[0]*3 )*10000],
         'Val':   [int(round(entry, 0)) for entry in rng.uniform(0, 1, temp.shape[0]*3 )*10000],
         'Test':  [int(round(entry, 0)) for entry in rng.uniform(0, 1, temp.shape[0]*3 )*10000]})

    # each row (exp year set)
    # j = 0 
    for j in range(temp.shape[0]):

        exCodeGroup =  temp.loc[j, 'ExperimentCodeGroup']
        year =         temp.loc[j, 'Year']
        downsampleTo = temp.loc[j, 'Total']
        split =        temp.loc[j, 'Split']

        mask = (phenotype.Year == year) & (phenotype.ExperimentCodeGroup == exCodeGroup)

        sampleTheseIndices = list(phenotype.loc[mask, ].reset_index().loc[:, 'index'])

        rng = np.random.default_rng(tempSeeds.loc[j, split])
        sampledIndices = list( rng.choice(sampleTheseIndices, downsampleTo) )
        tempDict[split] = tempDict[split] + sampledIndices                    

    # shuffle to break up any information based on order of the index
    # e.g., is index 2 more similar to index 3 than to index 300?
    rng = np.random.default_rng(downsampleObsSeeds[i])
    rng.shuffle(tempDict['Train'])
    rng.shuffle(tempDict['Val'])
    rng.shuffle(tempDict['Test'])


    # rename and add to list
    tempDict['Validate'] = tempDict['Val']
    del tempDict['Val']
    # force these to be ints (not numpy int 32) so they can be converted to json

    tempDict['Train'] =    [int(entry) for entry in tempDict['Train']]
    tempDict['Validate'] = [int(entry) for entry in tempDict['Validate']]
    tempDict['Test'] =     [int(entry) for entry in tempDict['Test']]

    indexDictList = indexDictList+[tempDict]

# %%
print('We have', len(indexDictList), 'training/testing splits ready.')

# %%
#TODO We need the groupings or a set of groupwise CV folds. 
# This could be done by providing the group indexes within the train test. 
# we could merge together the train/validate groups, then use phenotype to get the relevant groupings and write that out as a
# separate key/value pair. 

newIndexDictList = []

uniqExpCodeGroups = list(phenotype.loc[:, 'ExperimentCodeGroup'].drop_duplicates())
uniqExpCodeGroups = pd.DataFrame(zip(uniqExpCodeGroups, 
                                     [i for i in range(len(uniqExpCodeGroups))]
                                    ), columns = ['ExperimentCodeGroup', 'ExperimentCodeGroupNum'])

rng = np.random.default_rng(651683)
i = 0
for i in range(len(indexDictList)):
    tempDict = {} # make sure the previous dict is removed
    tempDict = {'Train':indexDictList[i]['Train']+indexDictList[i]['Validate'],
                'Test':indexDictList[i]['Test']
               }

    # The validate set is all one group, make the distribution of groups uniform
    rng.shuffle(tempDict['Train'])
    rng.shuffle(tempDict['Test'])

    # Get Groupings
    trainExpGroups = phenotype.loc[tempDict['Train'], ["ExperimentCodeGroup"]].merge(uniqExpCodeGroups, how  = 'left')
    testExpGroups = phenotype.loc[tempDict['Test'], ["ExperimentCodeGroup"]].merge(uniqExpCodeGroups, how  = 'left')
    # Add
    tempDict.update({'TrainGroups':list(trainExpGroups.ExperimentCodeGroupNum)}) # If not converted to a list, the data will be a series of numpy ints and be non json serializable.
    tempDict.update({'TestGroups': list(testExpGroups.ExperimentCodeGroupNum)})
    # Store
    newIndexDictList = newIndexDictList + [tempDict]



# %%
# Deprecated, from when we weren't using groupwise CV. 
# with open('../data/processed/indexDictList.txt', 'w') as convert_file:
#      convert_file.write(json.dumps(indexDictList))
        
with open('../data/processed/indexDictList.txt', 'w') as convert_file:
     convert_file.write(json.dumps(newIndexDictList))

# %%

# %% [markdown]
# # Create splits which match non-orginal datasets
#
# This is where we reduce the `indexDictList` to contain only entries that we have genomic data for. 

# %%

# %% [markdown]
# ## Make training set in agreement with the genome pca set (set1)

# %%

# %%
# phenotypeGBS

# %%
# phenotype

# %%
# sharedCols = [entry for entry in list(phenotypeGBS) if entry in list(phenotype)]

# %%

# %%

# %%

# %%
# # merge the two dfs on shared columns, then get the indices for the rows which were _not_ present in 
# temp = phenotypeGBS.loc[:, sharedCols]
# temp["HasGenome"] = True
# temp = temp.reset_index().rename(columns = {"index":"PhenoGBSIndex"})

# temp2 = phenotype.reset_index().rename(columns = {"index":"PhenoIndex"})

# temp2 = temp2.merge(temp, how = 'outer')


# # disallowed index
# rmTheseIndices = temp2.loc[temp2.HasGenome.isna()].reset_index().loc[:, 'PhenoIndex']
# rmTheseIndices = [int(entry) for entry in list(rmTheseIndices)]
# print("Example indices to be removed:", rmTheseIndices[1:5])

# %%
# temp2.loc[:, ['PhenoIndex', 'PhenoGBSIndex']]

# %%
indexDictList = json.load(open('../data/processed/indexDictList.txt'))

# %%

# indexDictList[i].keys()

# %%

# %%
# This is _painfully_ inefficient, but it only has to be done once.

# Go through each dictionary in the list. 
# For each list of sample indices
# Check if it's in the disallowed index set and don't return it if that's the case.
# update a new list of dicts that contain none of the disallowed samples

# note:
# `rmTheseIndices` is defined up in "Write out dataset 1 (G == PCA)"


indexDictListCleaned = []

# for dict in list
for i in tqdm.tqdm(range(len(indexDictList))):
    # for List type in Train/Test
    # for entry in list
    okTrainIndices = [j for j in range(len(indexDictList[i]['Train'])
                                      ) if indexDictList[i]['Train'][j] not in rmTheseIndices]
    
    replaceTrain =       [indexDictList[i]['Train'][k] for k in okTrainIndices]
    replaceTrainGroups = [indexDictList[i]['TrainGroups'][k] for k in okTrainIndices]

    okTestIndices = [j for j in range(len(indexDictList[i]['Test'])
                                     ) if indexDictList[i]['Test'][j] not in rmTheseIndices]
    replaceTest =       [indexDictList[i]['Test'][k] for k in okTestIndices]
    replaceTestGroups = [indexDictList[i]['TestGroups'][k] for k in okTestIndices]

    indexDictListCleaned = indexDictListCleaned+[{'Train':replaceTrain, 
                                                   'Test':replaceTest, 
                                            'TrainGroups':replaceTrainGroups, 
                                             'TestGroups':replaceTestGroups}]

# %%
with open('../data/processed/indexDictList_PCA.txt', 'w') as convert_file:
     convert_file.write(json.dumps(indexDictListCleaned))

# %%

# %%



##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##




# %%
# phenotypeGBSForIndices

# %%
# phenotype

# %%
