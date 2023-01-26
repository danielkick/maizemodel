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
import os
import re
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# %% code_folding=[]
# Data prep adapted from model.py
# Specified Options -----------------------------------------------------------
## Tuning ====
splitIndex = 0
numEpochs  = 500
kfolds     = 10
cvSeed     = 646843
max_pseudorep = 10 # <-- New. How many times should the model be run to account for random initialization?

## Model ====
# set needed data
needGpca = True
needGnum = False
needS    = True
needW    = True


# %% code_folding=[0]
# Index prep -------------------------------------------------------------------
path     = '../../' #atlas version
pathData = path+'data/processed/'

# get the right index set automatically.
if needGpca == True:
    indexDictList = json.load(open(pathData+'indexDictList_PCA.txt')) # <--- Changed to PCA
else:
    indexDictList = json.load(open(pathData+'indexDictList.txt'))

trainIndex    = indexDictList[splitIndex]['Train']
trainGroups   = indexDictList[splitIndex]['TrainGroups']
testGroups    = indexDictList[splitIndex]['TestGroups'] 
testIndex     = indexDictList[splitIndex]['Test']


# Data prep -------------------------------------------------------------------
# Y ===========================================================================
Y = np.load(pathData+'Y.npy')
YStd = Y[trainIndex].std()
YMean = Y[trainIndex].mean()
YNorm = ((Y - YMean) / YStd)

# G ===========================================================================
if needGpca:
    G = np.load(pathData+'G_PCA_1.npy') # <--- Changed to PCA
    GStd = np.apply_along_axis(np.nanstd, 0, G[trainIndex])
    GMean = np.apply_along_axis(np.nanmean, 0, G[trainIndex])
    GNorm = ((G - GMean) / GStd)
#     GNorm = G
    GNormPCA = GNorm.astype('float32') 
    
## G NUM
if needGnum:
    G = np.load(pathData+'G.npy')
    # No work up here because they're already ready.
    GNorm = G

    GNormNUM = GNorm.astype('int32')

# S ===========================================================================
if needS:
    S = np.load(pathData+'S.npy')
    SStd = np.apply_along_axis(np.std,  0, S[trainIndex])
    SMean = np.apply_along_axis(np.mean, 0, S[trainIndex])
    SNorm = ((S - SMean) / SStd)

    SNorm = SNorm.astype('float32')

# W =========================================================================== 
if needW:
    W = np.load(pathData+'W.npy')
    # we want to center and scale _but_ some time x feature combinations have an sd of 0. (e.g. nitrogen application)
    # to get around this we'll make it so that if sd == 0, sd = 1.
    # Then we'll be able to proceed as normal.
    WStd = np.apply_along_axis(np.std,  0, W[trainIndex])
    WStd[WStd == 0] = 1
    WMean = np.apply_along_axis(np.mean, 0, W[trainIndex])
    WNorm = ((W - WMean) / WStd)

    WNorm = WNorm.astype('float32')
    #WNorm = WNorm.astype('float64') # <- this was set, unclear if it was functional. 

# %%
# # mimicing model.py We set up the model to be run (or the text for it in this case) and the data for it together.    
# # Model defination ------------------------------------------------------------    
# if   readyModelGPCA:
#     ## G PCA
#     data_list = [GNormPCA[trainIndex]]
#     data_list2 = [GNormPCA[testIndex]]    
# elif readyModelGNUM:
#     ## G NUM   
#     data_list = [GNormNUM[trainIndex]]
#     data_list2 = [GNormNUM[testIndex]]    
# elif readyModelS:
#     ## S
#     data_list = [SNorm[trainIndex]]
#     data_list2 = [SNorm[testIndex]]    
# elif readyModelWmlp:
#     ## W MLP
#     data_list = [WNorm[trainIndex]]
#     data_list2 = [WNorm[testIndex]]
# elif readyModelWc1d:
#     ## W C1D 
#     data_list = [WNorm[trainIndex]]
#     data_list2 = [WNorm[testIndex]]    
# elif readyModelWc2d:
#     ## W C2d       
#     data_list = [WNorm[trainIndex]]
#     data_list2 = [WNorm[testIndex]]    
# else: 
#     raise NameError("No model has been selected to load.")

# %%
# load_path = '../Gpca_finalize_model/'

# dir_contents = os.listdir(load_path)
# models_to_load = [entry for entry in dir_contents if re.match('trained_model_*', entry)]

# %%
# models_to_load

# %%

# %%
Gpca_model = keras.models.load_model('../Gpca_finalize_model/trained_model_rep0')
Gpca_model.trainable = False

S_model = keras.models.load_model('../S_finalize_model/trained_model_rep0')
S_model.trainable = False

# W_model = keras.models.load_model('../Wmlp_finalize_model/trained_model_rep0') # <--- ! could be any of the Ws
W_model = keras.models.load_model('../Wc1d_finalize_model/trained_model_rep0') # <--- ! could be any of the Ws
# W_model = keras.models.load_model('../Wc2d_finalize_model/trained_model_rep0') # <--- ! could be any of the Ws
W_model.trainable = False

# %%

# %%

import matplotlib.pyplot as plt

for i in range(len(Gpca_model.layers)):
    print(i, Gpca_model.layers[i])

    


# %%
Gpca_model.layers[1].get_weights()[0].shape

# %%

# %%
plt.imshow(Gpca_model.layers[1].get_weights()[0])

# %%
plt.imshow(Gpca_model.layers[4].get_weights()[0])

# %%
plt.imshow(Gpca_model.layers[7].get_weights()[0])

# %%
plt.imshow(Gpca_model.layers[10].get_weights()[0])

# %%
plt.imshow(Gpca_model.layers[13].get_weights()[0])

# %%
plt.imshow(Gpca_model.layers[16].get_weights()[0])

# %%
plt.plot(Gpca_model.layers[16].get_weights()[0])

# %%

# %%
# Maybe we can use summary statistics as a proxy for layer importance
print("i  Min       Sd        Max")
for i in range(5):
    vals = Gpca_model.layers[1+(3*i)].get_weights()[0]
    print(i, 
          np.min(vals),
          np.std(vals),
          np.max(vals)
         )

# %%
Gpca_model.summary()

# %%
Gpca_model.layers()

# %%
# for layer in Gpca_model.layers:
for layer in W_model.layers:
    print(layer.name)

# %%

# %%
# Todo
# [x] remove last layer and replace it
# [x] concatenate two sublayers
# [ ] Add connections to arbitrary layers within a subnetwork
gIn = tf.keras.Input(shape=(1725), dtype='float32', name = 'gIn')
sIn = tf.keras.Input(shape=(21), dtype='float32', name = 'sIn')
wIn = keras.Input(shape=(288, 19, 1), dtype = 'float32', name = 'wIn')

# Demo arbitrary connection --
# Use the first and last layers as input for the interaction stack
Gpca_subnetwork_lvl0 = keras.models.Model(
    inputs=Gpca_model.input, 
    outputs=Gpca_model.get_layer('dense').output)

S_subnetwork_lvl0 = keras.models.Model(
    inputs=S_model.input, 
    outputs=S_model.get_layer('dense').output)

W_subnetwork_lvl0 = keras.models.Model(
    inputs=W_model.input, 
    outputs=W_model.get_layer('max_pooling1d').output)


gX = Gpca_subnetwork_lvl0(gIn)
sX = S_subnetwork_lvl0(sIn)
wX = W_subnetwork_lvl0(wIn)
wX = layers.Flatten()(wX)


Gpca_subnetwork = keras.models.Model(
    inputs=Gpca_model.input, 
    outputs=Gpca_model.layers[-2].output)

S_subnetwork = keras.models.Model(
    inputs=S_model.input, 
    outputs=S_model.layers[-2].output)

W_subnetwork = keras.models.Model(
    inputs=W_model.input, 
    outputs=W_model.layers[-2].output)

gOut = Gpca_subnetwork(gIn)
sOut = S_subnetwork(sIn)
wOut = W_subnetwork(wIn)

x = keras.layers.concatenate([gOut, sOut, wOut,
                              gX, sX, wX ])
x = keras.layers.Dense(100, activation = 'relu')(x)

yHat = layers.Dense(1, activation='linear', name='yHat')(x)

model = keras.Model(inputs=[gIn, sIn, wIn], outputs=[yHat])

# %%
model.summary()

# %%

adam_learning_rate = 0.01
adam_learning_beta1 = 0.9064524478168201
adam_learning_beta2 = 0.9115209599210423

opt = tf.keras.optimizers.Adam(
        learning_rate = adam_learning_rate,
        beta_1 = adam_learning_beta1, 
        beta_2 = adam_learning_beta2  
    )

model.compile(
    optimizer= opt,
    loss='mse',
    metrics = ['mae', 'mse']
)


# %%
model.fit([GNormPCA[trainIndex], SNorm[trainIndex], WNorm[trainIndex] ], 
          YNorm[trainIndex],
          epochs = 2)

# %%

# %%
# Trying to combine processing a what a model finds important with the layers that might be most useful to the full convoluted network

# Gpca_model.predict([GNorm[trainIndex]])

# %%
GNorm[trainIndex].shape

# %%

# %%
import numpy as np


# %%
nObs = 20000

fake_GNorm = np.random.normal(0, 1, nObs*1725).reshape(nObs, 1725)

# %%
fake_yhat = Gpca_model.predict([fake_GNorm])

# %%
plt.hist(fake_yhat)

# %%
import pandas as pd
fake_GNorm_df = pd.DataFrame(fake_GNorm)
fake_GNorm_df['yhat'] = fake_yhat

# %%
fake_GNorm_df = fake_GNorm_df.sort_values('yhat')

# %%
plt.imshow(fake_GNorm_df.loc[fake_GNorm_df['yhat'] > 0.05, ])

# %%
plt.imshow(fake_GNorm_df.loc[fake_GNorm_df['yhat'] < -0.25, ])

# %%
fake_GNorm_df = fake_GNorm_df.reset_index().drop(columns = 'index')

print("Plot 'data' for 500 highest and 500 lowest yhats")
plt.imshow(pd.concat([fake_GNorm_df.loc[0:499, ], fake_GNorm_df.loc[1501:2000, ]]))


# %%

# %%
asdef

# %%
# Works on one!]

model2 = keras.models.Model(inputs=Gpca_model.input, 
                           outputs=Gpca_model.layers[-2].output)

gX = model2(gIn)

gX = keras.layers.Dense(1)(gX)
outputs = keras.layers.Dense(1)(gX)

model = keras.Model(gIn, outputs)



adam_learning_rate = 0.01
adam_learning_beta1 = 0.9064524478168201
adam_learning_beta2 = 0.9115209599210423

opt = tf.keras.optimizers.Adam(
        learning_rate = adam_learning_rate,
        beta_1 = adam_learning_beta1, 
        beta_2 = adam_learning_beta2  
    )

model.compile(
    optimizer= opt,
    loss='mse',
    metrics = ['mae', 'mse']
)


model.summary()

# %%
model.fit(data_list, 
          YNorm[trainIndex],
          epochs = 2)

# %%
model2.predict(data_list)

# %%

# %%

# %%

# %%

# %%
gIn = tf.keras.Input(shape=(1725), dtype='float32', name = 'gIn')

gX = Gpca_model.layers[-2](gIn, training = False)

gX = keras.layers.Dense(1)(gX)
outputs = keras.layers.Dense(1)(gX)

model = keras.Model(gIn, outputs)



adam_learning_rate = 0.01
adam_learning_beta1 = 0.9064524478168201
adam_learning_beta2 = 0.9115209599210423

opt = tf.keras.optimizers.Adam(
        learning_rate = adam_learning_rate,
        beta_1 = adam_learning_beta1, 
        beta_2 = adam_learning_beta2  
    )

model.compile(
    optimizer= opt,
    loss='mse',
    metrics = ['mae', 'mse']
)

# %%
Gpca_model.summary()

# %%
model.summary()

# %%
model.fit(data_list, epochs = 2)

# %%
model = keras.models.load_model('../Gpca_finalize_model/trained_model_rep0')

yhat_train = model.predict(data_list)
y_train = YNorm[trainIndex]
yhat_test = model.predict(data_list2)
y_test = YNorm[testIndex]

# %%

# %%

# %%

# %%
dir_contents = os.listdir()
models_to_load = [entry for entry in dir_contents if re.match('trained_model_*', entry)]

out_dict = {}
for model_to_load in models_to_load:
    rep_id = model_to_load.split('_')[-1]

    model = keras.models.load_model('./'+model_to_load)

    yhat_train = model.predict(data_list)
    y_train = YNorm[trainIndex]
    yhat_test = model.predict(data_list2)
    y_test = YNorm[testIndex]

    temp_dict = {
        rep_id: {
        "y_train"   : list(y_train),
        "yhat_train": [entry[0] for entry in yhat_train.tolist()],
        "y_test"    : list(y_test),
        "yhat_test" : [entry[0] for entry in yhat_test.tolist()]
        }}
    
    out_dict.update(temp_dict)
    
with open('all_model_res.json', 'w') as f:
    json.dump(out_dict, f)

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
