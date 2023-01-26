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

# %%
# Specified Options -----------------------------------------------------------

## tslearn ====
demoClustering = False
compareClusterInput = False

compareSilhouettes = False 

## General ====
## Tuning ====
splitIndex = 0

kfolds =     10

cvSeed = 646843

needGNum = True
needGPCA = True
needS    = True
needW    = True

# %%
# Set up indice sets -----------------------------------------------------------
# path     = '../../'         #atlas version
path     = '../data/atlas/' #local version
pathData = path+'data/processed/'

indexDictList = json.load(open(pathData+'indexDictList_PCA_syr.txt'))
trainIndex =  indexDictList[splitIndex]['Train']
testIndex =   indexDictList[splitIndex]['Test']
trainGroups = indexDictList[splitIndex]['TrainGroups']
testGroups =  indexDictList[splitIndex]['TestGroups']


# G ===========================================================================
if needGPCA:
    G = np.load(pathData+'G_PCA_1.npy') # <--- Changed to PCA
    GStd = np.apply_along_axis(np.nanstd, 0, G[trainIndex])
    GMean = np.apply_along_axis(np.nanmean, 0, G[trainIndex])
    GNorm = ((G - GMean) / GStd)    

    GNorm = GNorm.astype('float32')
    GNormPCA = GNorm.copy()
        


# %%
np.sum(GNormPCA)

# %%
print('If this is not nan, there are no nas in the test/train spilt selected')
    
np.sum(GNormPCA[trainIndex+testIndex ])

# %%
G = np.load(pathData+'G_PCA_1.npy') # <--- Changed to PCA

# %%
import tqdm

print("Checking for Nas in GPCA Training/Testing splits:")
indicesWithNans = []

posControl = np.isnan(np.sum(G))
if posControl:
    print("Positive Control works --", posControl)
else:
    raise Exception("Positive Control failed.")

for splitIndex in tqdm.tqdm( range(len(indexDictList)) ):
    trainIndex =  indexDictList[splitIndex]['Train']
    testIndex =   indexDictList[splitIndex]['Test']

    nanFound = np.isnan( np.sum(G[trainIndex+testIndex ]) )

    if nanFound:
        indicesWithNans = indicesWithNans+[splitIndex]
        # print("Nas found in index", splitIndex)

if len(indicesWithNans) != 0:
    print("NAs found for indices:", indicesWithNans)
    raise Exception("NAs found in indices.")    
else:
    print("No NAs found!")   

# %%
