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

# %%
import os
import json

import numpy as np
import tslearn
from tslearn.clustering import TimeSeriesKMeans

import pickle # save/load clustering results
import math # for ceil()
import tqdm
import matplotlib.pyplot as plt

# %%
# Specified Options -----------------------------------------------------------

## tslearn ====
demoClustering = False
compareClusterInput = False
compareSilhouettes = False 

remakeClustering = False # Should clustrings be made or loaded?
                         # This takes about 4 hours to complete. 




seed = 564564684


## General ====
## Tuning ====
splitIndex = 0

kfolds =     10

cvSeed = 646843

needGNum = True
needGPCA = True
needS    = True
needW    = True

# %% code_folding=[]
# Set up indice sets -----------------------------------------------------------
# path     = '../../'         #atlas version
path     = '../data/atlas/' #local version
pathData = path+'data/processed/'




# indexDictList = json.load(open(pathData+'indexDictList_PCA.txt'))
# trainIndex =  indexDictList[splitIndex]['Train']
# testIndex =   indexDictList[splitIndex]['Test']
# trainGroups = indexDictList[splitIndex]['TrainGroups']
# testGroups =  indexDictList[splitIndex]['TestGroups'] # Don't actually need this until we evaluate the tuned model.



# Y ===========================================================================
Y = np.load(pathData+'Y.npy')
    
# G ===========================================================================
G = np.load(pathData+'G_PCA_1.npy') # <--- Changed to PCA
        
# S ===========================================================================
S = np.load(pathData+'S.npy')

# W ===========================================================================    
WNorm = np.load(pathData+'W.npy')

# %%

# %%
phenotype = pd.read_pickle('../data/processed/phenotype_for_keras.pkl') # < --- These are the versions to use
# phenotypeGBS = pd.read_pickle('../data/processed/phenotype_with_GBS_for_keras.pkl')

# %%

# %%
# GNumNames = json.load(open("../data/processed/GNumNames.txt"))
# GNumNames

# %%
# GPCANames = ['PC'+str(i+1) for i in range(GNormPCA.shape[1])]
# GPCANames

# %% [markdown]
#  The data we want for R is phenotype/covariate df (with indexing)
#  * filter indices
#  * y
#  * soil covariates
#  * weather/management covariates
#      * cluster all
#      * cluster 1-n
#  * G PCA

# %%

# %% [markdown]
# ## Set up a df to merge data into. 

# %%
## Indexing _without_ the requirement of genome data =============================
indexDictList = json.load(open(pathData+'indexDictList.txt'))
trainIndex =  indexDictList[splitIndex]['Train']
testIndex =   indexDictList[splitIndex]['Test']
trainGroups = indexDictList[splitIndex]['TrainGroups']
testGroups =  indexDictList[splitIndex]['TestGroups'] # Don't actually need this until we evaluate the tuned model.


# %%
phenotype.loc[testIndex[0]:50639, ]

# %%

# We'll use `phenotype` as the basis for the object. 
dataForR = phenotype.copy()

# previous issues with merging -- proceeding with a highly inefficient but effective way. 
# Groupings without requiring genomic data
indexDictList = json.load(open(pathData+'indexDictList.txt'))
trainIndex    = indexDictList[splitIndex]['Train']
testIndex     = indexDictList[splitIndex]['Test']
trainGroups   = indexDictList[splitIndex]['TrainGroups']
testGroups    = indexDictList[splitIndex]['TestGroups'] 

dataForR['SplitGnum'] = ''
dataForR['SplitGnumGroups'] = np.nan

for i in tqdm.tqdm(range(len(trainIndex))):
    index = trainIndex[i]
    dataForR.loc[index, 'SplitGnum'] = 'Train'
    dataForR.loc[index, 'SplitGnumGroups'] = trainGroups[i]
    
for i in tqdm.tqdm(range(len(testIndex))):
    index = testIndex[i]
    dataForR.loc[index, 'SplitGnum'] = 'Test'
    dataForR.loc[index, 'SplitGnumGroups'] = testGroups[i]
    

# Groupings with genomic data
indexDictList = json.load(open(pathData+'indexDictList_PCA.txt'))
trainIndex    = indexDictList[splitIndex]['Train']
testIndex     = indexDictList[splitIndex]['Test']
trainGroups   = indexDictList[splitIndex]['TrainGroups']
testGroups    = indexDictList[splitIndex]['TestGroups'] 

dataForR['SplitGpca'] = ''
dataForR['SplitGpcaGroups'] = np.nan

for i in tqdm.tqdm(range(len(trainIndex))):
    index = trainIndex[i]
    dataForR.loc[index, 'SplitGpca'] = 'Train'
    dataForR.loc[index, 'SplitGpcaGroups'] = trainGroups[i]
    
    
for i in tqdm.tqdm(range(len(testIndex))):
    index = testIndex[i]
    dataForR.loc[index, 'SplitGpca'] = 'Test'
    dataForR.loc[index, 'SplitGpcaGroups'] = testGroups[i]
    
    

# %%

# %%

# %%

# %% [markdown]
# ## Soil

# %%
soil = pd.read_pickle('../data/processed/soil.pkl')
soilDataColNames = [entry for entry in soil.columns.to_list(
) if entry not in ['ExperimentCode', 'Year', 'DatePlanted']]

# %%
# pd.concat([phenotype, pd.DataFrame(S, columns= soilDataColNames)], axis = 1)

# %%
nRowOld = dataForR.shape[0]

dataForR = pd.concat([dataForR, pd.DataFrame(S, columns= soilDataColNames)], axis = 1)
if nRowOld != dataForR.shape[0]:
    raise Exception('Expected consistent number of rows but found '+str(nRowOld)+' -> '+str(dataForR.shape[0]))

# %%

# %% [markdown]
# ## Weather New Method -- Data for weather ERM matrix
# Switching from clustering weather data as needed for the Classic ML methods to building an environmental relationship matrix as in Washburn et al 2021.
# As a result, we now need all the weather measurements.

# %%
weather   = pd.read_pickle('../data/processed/weather.pkl')

# %%
mask = (weather.ExperimentCode == "DEH1") & (weather.Year == "2019")
weather.loc[mask, :].drop_duplicates().shape

# %% [markdown]
# ## Weather Deprecated Method -- Weather Environment clusters

# %% [markdown]
# ### Weather clustering prep.

# %%
weather   = pd.read_pickle('../data/processed/weather.pkl')

# %%
# I think dynamic time warping is probably what we want to use.

# input numpy array is 
# samples, timesteps, variables (1)


# make a lookup so we can use W or WNorm instead of recreating it or using all the samples with duplicate locations.
# Adapted from prepareDataForKeras.py
uniqPlantToIndexDict = {}

# recreate unique plantings
uniquePlantings = phenotype.loc[:, ['ExperimentCode', 'Year', 'DatePlanted']
                               ].drop_duplicates(
                               ).reset_index(
                               ).drop(columns = ['index'])

for i in range(len(uniquePlantings)):
    mask = (phenotype.ExperimentCode == uniquePlantings.loc[i, 'ExperimentCode']
           ) & (phenotype.Year == uniquePlantings.loc[i, 'Year']
           ) & (phenotype.DatePlanted == uniquePlantings.loc[i, 'DatePlanted']
           )

    updateSamples = list(phenotype.loc[mask, ].reset_index().loc[:, 'index'])

    uniqPlantToIndexDict.update({i:updateSamples})  
#     print(i, updateSamples)

# %%

# create and fill in a deduplicated 3d array
WMin = np.zeros((len(list(uniqPlantToIndexDict.keys())), WNorm.shape[1], WNorm.shape[2]))
for i in list(uniqPlantToIndexDict.keys()):
    idx = uniqPlantToIndexDict[i][0]    
    WMin[i, :, :] = WNorm[idx, :, :]

    # uniqPlantToIndexDict[]

# %% [markdown]
# ### Demonstrate `tslearn` use on weather 

# %%
if demoClustering == True:

    # now we can slice across the last (and smallest) axis
    # to get all combinations for a single measure

    print(list(weather)[3:])


    i = 5

    x_temp = WMin[:, :, i]
    print(x_temp.shape)

    plt.imshow(x_temp)

# %%
if demoClustering == True:    
    km = TimeSeriesKMeans(n_clusters=3,
                              n_init=2,
                              metric="dtw",
    #                          verbose=True,
                              n_jobs = 2, 
                              max_iter_barycenter=10,
                              random_state=seed)

    y_pred = km.fit_predict(x_temp)

    print('example cluster predictions', y_pred[0:5])

# %%
if demoClustering == True:
    plotme = pd.DataFrame(x_temp)
    plotme['cluster'] = y_pred
    plotme = plotme.sort_values('cluster')

    plt.imshow(plotme.drop(columns = ['cluster']))

# %%
if demoClustering == True:
    # example search
    max_k = 10

    search_k = pd.DataFrame([i+2 for i in range(max_k) if i+1 < max_k], columns = ['k'])
    search_k['Silhouette'] = np.nan

    import tqdm
    for i in tqdm.tqdm(range(search_k.shape[0])):

        km = TimeSeriesKMeans(n_clusters=int(search_k.loc[i, 'k']),
                                  n_init=2,#4,
                                  metric="dtw",
        #                          verbose=True,
                                  n_jobs = 4, 
                                  max_iter_barycenter=10,
                                  random_state=seed)

        y_pred = km.fit_predict(x_temp)

        search_k.loc[i, 'Silhouette'] = tslearn.clustering.silhouette_score(x_temp, y_pred, metric = 'dtw')

    plt.plot(search_k.k, search_k.Silhouette)

# %% [markdown] heading_collapsed=true
# ### Demonstrate use of multiple measures at the same time

# %% hidden=true
# compare clustering on one or all measures:

if compareClusterInput == True:
    
    clusterResults = pd.DataFrame(np.zeros(shape = (WMin.shape[0], WMin.shape[2]+1)), 
                 columns = [str(i) for i in range(WMin.shape[2]+1)])
    clusterResults = clusterResults.rename(columns = {list(clusterResults)[-1]:'all'})

    for i in range(WMin.shape[2]+1):
        km = TimeSeriesKMeans(n_clusters=6,
                      n_init=2,
                      metric="dtw",
                      # verbose=True,
                      n_jobs = 2, 
                      max_iter_barycenter=10,
                      random_state=seed)
        print(i)
        if i != (WMin.shape[2]):        
            clusterResults.loc[:, str(i)] = km.fit_predict(WMin[:, :, i])

        else:
            clusterResults.loc[:, 'all'] = km.fit_predict(WMin[:, :, :])

# %% hidden=true
if compareClusterInput == True:
    plt.figure(figsize=(2, 30), dpi=80)
    plt.imshow(clusterResults)
    plt.set_cmap('Accent')

# %% [markdown] heading_collapsed=true
# ### Demonstrate how to find the optimal k

# %% hidden=true

# %% code_folding=[0] hidden=true
if compareSilhouettes:
    ## Using all the metrics available, what's the best k we can get?
    max_k = 17
    
    possible_ks = [i+2 for i in range(max_k) if i+1 < max_k]
    # for storing clustering assigments
    # so we don't have to remake them
    clusterResults = pd.DataFrame(np.zeros(shape = (WMin.shape[0], len(possible_ks))), 
                                  columns = [str(nom) for nom in possible_ks])
    
    
    # for choosing the right k
    silhouetteComparison = pd.DataFrame(possible_ks, columns = ['k'])
    silhouetteComparison['Silhouette'] = np.nan

    for i in tqdm.tqdm(range(silhouetteComparison.shape[0])):

        considered_k = int(silhouetteComparison.loc[i, 'k'])
        
        km = TimeSeriesKMeans(n_clusters=considered_k,
                                  n_init= 4,
                                  metric="dtw",
        #                          verbose=True,
                                  n_jobs = 4, 
                                  max_iter_barycenter=10,
                                  random_state=seed)

        y_pred = km.fit_predict(WMin)
        
        clusterResults.loc[:, str(considered_k)] = y_pred
        

        silhouetteComparison.loc[i, 'Silhouette'] = tslearn.clustering.silhouette_score(WMin, y_pred, metric = 'dtw')


    silhouetteComparison.to_csv('../data/processed/silhouetteComparison.csv')    
    clusterResults.to_csv('../data/processed/clusterResults.csv')

# %% code_folding=[] hidden=true
if compareSilhouettes == False:
    silhouetteComparison = pd.read_csv('../data/processed/silhouetteComparison.csv', usecols= ['k', 'Silhouette'])
    clusterResults = pd.read_csv('../data/processed/clusterResults.csv')    

# %% code_folding=[] hidden=true
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

# %% hidden=true
# plot as a sanity check
plt.plot(silhouetteComparison.k, silhouetteComparison.Silhouette)
plt.scatter(x=bestK, 
            y = list(silhouetteComparison.loc[silhouetteComparison.k == bestK, 'Silhouette'])[0], 
            color = 'red')

# %% hidden=true
weatherClusters = clusterResults.loc[:, str(bestK)]

# %% hidden=true
# Apply to all observations in phenotype. 


# format in the same way as WNorm
WCluster = np.zeros(shape = (WNorm.shape[0], 1))

# going to be slow but only has to be run once per clustering.
for i in range(weatherClusters.shape[0]):
    for j in list(uniqPlantToIndexDict[i]):
        WCluster[j] = weatherClusters.loc[i]

WCluster = [str(int(entry[0])) for entry in WCluster]
WCluster = np.array(WCluster)
WCluster


# %% hidden=true

# %% hidden=true

# %% [markdown] heading_collapsed=true
# ### Scan for best cluster number and best clustering for each and all weather variables.
#
# We might as well get all the covariates we can from weather. 
# ```
# for each weather variable (and all together);
#     find best cluster number
#     return best cluster assignments
# format and merge into phenotype so that one or all can be used as a covariate
# ```
#
#

# %% code_folding=[0, 3, 23, 26] hidden=true
def get_tscluster_results(
    df = WMin[:, :, 1],
    considered_k = 2
):
    km = TimeSeriesKMeans(n_clusters=considered_k,
                          n_init= 4,
                          metric="dtw",
                          #verbose=True,
                          n_jobs = 4, 
                          max_iter_barycenter=10,
                          random_state=seed)

    y_pred = km.fit_predict(df)

    # silhouetteComparison.loc[i, 'Silhouette'] = 
    silhouette_score = tslearn.clustering.silhouette_score(df, y_pred, metric = 'dtw')

    return({
        'yHat': y_pred,
        'silhouette': silhouette_score
    })


def scan_tscluster_results(
    df =  WMin[:, :, 1],
    kList = [2, 3]
):

    yHats = []
    silhouettes = []
    for i in kList:
        kRes = get_tscluster_results(
            df = df,
            considered_k = i
        )

        silhouettes += [kRes['silhouette']]
        yHats.append([kRes['yHat'].tolist()])
    return({
        'silhouettes':silhouettes,
        'yHats':yHats
    })


# %% hidden=true
# WMin2 = WMin[:, 50:75, :]

# %% code_folding=[11] hidden=true
# here we'll store all the output from each 
# DataSlice [ k [cluster, ], ]

# overallList
# |- # entry for slice 1
# |  |- silhouettes: [k_min  ... k_max]
# |  |- yHats: [k_min  ... k_max]
# |
# |- # entry for slice 2
# | ...

if remakeClustering:
    overallList = []
    kList = [2+i for i in range(17-2) ]

    # For each individual column
    for i_slice in tqdm.tqdm(range(WMin.shape[2])):
        slice_results = scan_tscluster_results(
            df =  WMin[:, :, i_slice], #
            kList = kList
        )
        overallList.append(slice_results)

    # and for all data columns
    slice_results = scan_tscluster_results(
        df =  WMin[:, :, :], 
        kList = kList
    )
    overallList.append(slice_results)

# %% code_folding=[0] hidden=true
if remakeClustering:
    bestClusterings = []  # this holds the best clustering found for a given

    for temp in overallList:

        temp_sil = temp['silhouettes']
        temp_sil_next = temp_sil[1:]+[0]

        silhouetteComparison = pd.DataFrame(
            {'Silhouette': temp_sil,
             'NextSil': temp_sil_next,
             'Delta': 0,
             'Allow': True}
        )
        silhouetteComparison['Delta'] = silhouetteComparison['Silhouette'] - silhouetteComparison['NextSil'] 
        silhouetteComparison['k'] = kList

        # silhouetteComparison = silhouetteComparison.reset_index().rename(columns = {'index': 'k'})


        # find the first k where the silhouette score degrades.
        # disallow all values of k greater than that.
        stopK = silhouetteComparison.loc[silhouetteComparison.Delta < 0, 'k']
        if stopK.shape[0] > 0:
            silhouetteComparison.loc[silhouetteComparison.k > min(stopK), 'Allow'] = False
        # use the largest k before it regresses
        bestK = max(silhouetteComparison.loc[silhouetteComparison.Allow, 'k'])

        print("The best k found is", bestK)

        indx = [i for i in range(len(kList)) if kList[i] == bestK][0]

        out = {'bestK': bestK,
               'bestCluster': temp['yHats'][indx],
               'silhouetteDF':silhouetteComparison}

        bestClusterings.append(out)

# %% code_folding=[0, 3] hidden=true
if remakeClustering:
    print('Warning! On 2021-12-7 this took >3:47 to complete!')
    pickle.dump( bestClusterings, open( "weatherTimeSeriesClusterings.pkl", "wb" ) )
else:
    bestClusterings = pickle.load( open( "weatherTimeSeriesClusterings.pkl", "rb" ) )

# %% code_folding=[0] hidden=true
# plot the clustering results
print('Here are the clustering results for each and all the time series data:')

# Get the inputs variable names for each of the clusterings
weatherVariables = weather.keys().tolist()
plotTitles = weatherVariables[(-1*(len(bestClusterings)-1)):] # use len-1 because the last clustering is on "all"
plotTitles.extend(['All'])


nfigs = len(bestClusterings)
rfigs = int(round(np.sqrt(nfigs)))
cfigs = math.ceil(nfigs/rfigs)

fig, axs = plt.subplots(rfigs, cfigs)

counter = 0
for i in  range(rfigs):
    for j in range(cfigs):
        if counter < nfigs:
#             print(i)
            df = bestClusterings[i]['silhouetteDF']
            bestK = bestClusterings[i]['bestK']
        
            axs[i, j].vlines(x=[i+1 for i in range(5)], 
                             ymin = min(df.Silhouette), 
                             ymax = max(df.Silhouette), 
                             color = "lightgrey")
            axs[i, j].plot(df.k, df.Silhouette)
            axs[i, j].scatter(x=bestK,
                              y = list(df.loc[df.k == bestK, 'Silhouette'])[0], 
                              color = 'red')
            axs[i, j].set_title(plotTitles[counter])
            counter += 1
            
fig.set_size_inches(15, 14)

# %% code_folding=[0, 7] hidden=true
# For each we need to get the best clustering and then back apply to all the conditions

# we'll merge these back in using the ['ExperimentCode', 'Year', 'DatePlanted'] columns. 
# These were what we used to determing the unique plantings to begin with.
plotTitles.extend(['ExperimentCode', 'Year', 'DatePlanted']) 

# Make an empty placeholder df
wClustersforMerge = pd.DataFrame(
    np.empty((
        len(uniqPlantToIndexDict.keys()),
        (len(bestClusterings)+3)
    )),
    columns=plotTitles
)
wClustersforMerge[:] = np.NaN


# add clusterings in
for i in range(len(bestClusterings)):
    wClustersforMerge.loc[:, plotTitles[i]
                          ] = bestClusterings[i]['bestCluster'][0]

# add metadata in for joining with phenotype df
# use the lookup dictionary we built earlier.
refCols = ['ExperimentCode', 'Year', 'DatePlanted']

for i in list(uniqPlantToIndexDict.keys()):
    idx = uniqPlantToIndexDict[i][0]
    
    wClustersforMerge.loc[i, refCols] = phenotype.loc[idx, refCols]
    
    
# rename columns with prepended "WC_" for "Weather Cluster"
renameDict = {}

for oldVal in [title for title in plotTitles if title not in refCols]:
    renameDict.update({oldVal:"WC_"+oldVal})
    
wClustersforMerge = wClustersforMerge.rename(columns = renameDict)


# make sure that this is a datetime so that it merges correctly. 
wClustersforMerge['DatePlanted'] = pd.to_datetime(wClustersforMerge['DatePlanted'])

# %% hidden=true
# FIXME!

# now can incorperate correctly 
# phenotype.merge(wClustersforMerge, how = 'outer')

# %% hidden=true
nRowOld = dataForR.shape[0]

# dataForR = pd.concat([dataForR, pd.DataFrame(G, columns = ['PC'+str(i+1) for i in range(G.shape[1])]) ], axis = 1)
dataForR = dataForR.merge(wClustersforMerge, how = 'outer')
if nRowOld != dataForR.shape[0]:
    raise Exception('Expected consistent number of rows but found '+str(nRowOld)+' -> '+str(dataForR.shape[0]))

# %% hidden=true

# %% hidden=true

# %% [markdown]
# ## Genome PCA

# %%
# pd.concat([phenotype, pd.DataFrame(G, columns = ['PC'+str(i+1) for i in range(G.shape[1])]) ], axis = 1)

# %%
nRowOld = dataForR.shape[0]

dataForR = pd.concat([dataForR, pd.DataFrame(G, columns = ['PC'+str(i+1) for i in range(G.shape[1])]) ], axis = 1)
if nRowOld != dataForR.shape[0]:
    raise Exception('Expected consistent number of rows but found '+str(nRowOld)+' -> '+str(dataForR.shape[0]))

# %%

# %%

# %%

# %%
dataForR.to_feather('../data/processed/covariatesForR.feather')
