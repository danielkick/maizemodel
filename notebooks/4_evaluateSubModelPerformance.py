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
import json

import pandas as pd
import numpy as np
# import tensorflow as tf
# import keras

# import keras_tuner as kt

# import tqdm
# import matplotlib.pyplot as plt


# %% [markdown]
# Getting the losses is done by a script in each model directory (`hpSearchEvaluation.py`). Here we'll retrieve the json output from those and visualize it.

# %%
model_dirs = ['S_2', 'Gpca_2', 'Gnum_2']

# the models that had to be run without cv modified tuner
w_prefixes = ['W_mlp', 'W_c1d']#, 'W_c2d']
w_ks = [0]
w_postfix = '_atlas'

w_dirs = []
for w_prefix in w_prefixes:
    for w_k in w_ks:
        w_dirs += [model_prefix+'_k'+str(model_k)+model_postifix]

model_dirs = model_dirs+w_dirs

# %%
# Transform the dictionaries into a tidy dataframe.
all_losses = pd.DataFrame()

for model_dir in model_dirs:
    tempPath = '../data/atlas/models/'+model_dir+'/hpSearchBestLoss.json'
    print(tempPath)
    if os.path.exists(tempPath):
        if (os.path.getsize(tempPath) > 0) :
            with open(tempPath) as f:
                read_data = f.read()

                temp = json.loads(read_data)
                ii = temp.keys()
                ii = [entry for entry in list(ii) if entry not in ['model_name']]

                acc_losses = pd.DataFrame()
                for i in ii:
                    temp_losses = pd.DataFrame().from_dict(temp[i]['performance']).reset_index().rename(columns = {'index':'k'})
                    temp_losses['model_place'] = i
                    if acc_losses.shape[0] == 0:
                        acc_losses = temp_losses
                    else:
                        acc_losses = acc_losses.merge(temp_losses, how = 'outer')

                acc_losses['model_name'] = temp['model_name']
                acc_losses['prefix'] = temp['model_name'].split('_k')[0]

                if all_losses.shape[0] == 0:
                    all_losses = acc_losses
                else:
#                     all_losses = all_losses.merge(acc_losses, how = 'outer')
                    all_losses = pd.concat([all_losses, acc_losses])

# %%

# %%
all_losses

# %%
import matplotlib.pyplot as plt

for prefix in list(set(all_losses.prefix)):
    for place in range(10):
        place = str(place)
        
        



# %%
place = '0'
prefix = 'S_2'

temp = all_losses.loc[((all_losses.prefix == prefix) & (all_losses.model_place == place)), ]

plt.scatter(temp.model_place, temp.cv_maes)
plt.scatter(temp.model_place, temp.naive_maes)

# %%

# %%
from plotnine import *

# %%

# %%
all_losses['model_place'] = all_losses['model_place'].astype(str)

ggplot(all_losses#.loc[all_losses.prefix == 'Gpca_2', ]
)+geom_point(aes(x = 'model_place', y = 'cv_maes')
)+geom_point(aes(x = 'model_place', y = 'naive_maes'), color = 'red', alpha = 0.05
# )+scale_y_continuous(trans='log10'
# )+facet_wrap('~ prefix', nrow = 1, scales = 'free_y')
)+ylim([0.5, 1.2]
)+facet_wrap('~ prefix', nrow = 1)




# %%

# %%
all_losses

# %%
evaluate_models = ['W_mlp', 'W_c1d', 'W_c2d']

modelRes = pd.DataFrame()
for model in evaluate_models:
    print(model)
    tempPath = '../data/atlas/models/'+model+'/hpSearchBestLoss.json'
    if os.path.exists(tempPath):
    if (os.path.getsize(tempPath) > 0) :
        temp = pd.read_json(tempPath)
        temp = temp.assign(model = model)

        if modelRes.shape[0] == 0:
            modelRes = temp
        else:
            modelRes = pd.concat([modelRes, temp])

# %%
# model = 'W_mlp'

evaluate_models = ['W_mlp', 'W_c1d', 'W_c2d']

modelRes = pd.DataFrame()
for model in evaluate_models:
    print(model)
    tempPath = '../data/atlas/models/'+model+'/hpSearchBestLoss.json'
    if os.path.exists(tempPath):
    if (os.path.getsize(tempPath) > 0) :
        temp = pd.read_json(tempPath)
        temp = temp.assign(model = model)

        if modelRes.shape[0] == 0:
            modelRes = temp
        else:
            modelRes = pd.concat([modelRes, temp])


# %%
modelRes

# %%
import matplotlib.pyplot as plt


plt.scatter(x = 'model', y = 'cv_losses', data = modelRes)
plt.hlines(xmin = -.1, xmax = 0.1,  y = 'mean_naive_loss', data = modelRes, color = 'red')
plt.hlines(xmin = -.1, xmax = 0.1,  y = 'mean_cv_loss', data = modelRes, color = 'red')

# %%

# %% [markdown]
# Data prep taken from classic ml

# %% code_folding=[]
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

# %% code_folding=[]
# Set up indice sets -----------------------------------------------------------
# path     = '../../'         #atlas version
path     = '../data/atlas/' #local version
pathData = path+'data/processed/'

indexDictList = json.load(open(pathData+'indexDictList.txt'))
trainIndex =  indexDictList[splitIndex]['Train']
testIndex =   indexDictList[splitIndex]['Test']
trainGroups = indexDictList[splitIndex]['TrainGroups']
testGroups =  indexDictList[splitIndex]['TestGroups'] # Don't actually need this until we evaluate the tuned model.



# indexDictList = json.load(open(pathData+'indexDictList_PCA.txt'))
# trainIndex =  indexDictList[splitIndex]['Train']
# testIndex =   indexDictList[splitIndex]['Test']
# trainGroups = indexDictList[splitIndex]['TrainGroups']
# testGroups =  indexDictList[splitIndex]['TestGroups'] # Don't actually need this until we evaluate the tuned model.



# Y ===========================================================================
Y = np.load(pathData+'Y.npy')
YStd = Y[trainIndex].std()
YMean = Y[trainIndex].mean()
YNorm = ((Y - YMean) / YStd)


# G ===========================================================================
if needGNum:
    G = np.load(pathData+'G.npy')
    # No work up here because they're already ready.
    GNorm = G

    GNorm = GNorm.astype('int32')
    GNormNum = GNorm.copy()
    
# G ===========================================================================
# if needGPCA:
#     G = np.load(pathData+'G_PCA_1.npy') # <--- Changed to PCA
#     GStd = np.apply_along_axis(np.nanstd, 0, G[trainIndex])
#     GMean = np.apply_along_axis(np.nanmean, 0, G[trainIndex])
#     GNorm = ((G - GMean) / GStd)    

#     GNorm = GNorm.astype('float32')
#     GNormPCA = GNorm.copy()
        
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
#     WNorm = WNorm.astype('float32')
    WNorm = WNorm.astype('float64')

# %%

# %%
path = '../../'         #atlas version
path = '../data/atlas/' #local version

projectName = 'kt' # this is the name of the folder which will be made to hold the tuner's progress.
# batchSize = hp.Int('batch_size', 32, 128, step=32)
maxTrials = 2

# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# %%
# Based on https://github.com/keras-team/keras-tuner/issues/122
class TunerGroupwiseCV(kt.engine.tuner.Tuner):
    def run_trial(self, trial, x, y,
                  nTestGroups,# = len(set(testGroups)),
                  kfolds,# = 2,     # <- We specify the number of desired folds because
                                  # the number of groups in the training set my not be evenly divsible by
                                  # the number of groups in the true test set.
                  cvSeed,# = 646843,
                  trainGroups,# = trainGroups,

#                   batch_size,#=32,
                  epochs,#=1,
                  *fit_args, **fit_kwargs
                 ):
        # Fold setup goes here ----
        cvGroup = list(set(trainGroups))
        rng = np.random.default_rng(cvSeed)
        valHoldoutDict = {}
        ## make the validation fold group membership for each fold. ====
        for i in range(kfolds):
            rng.shuffle(cvGroup)
            valHoldoutDict.update({str(i):cvGroup[0:nTestGroups]})
        ## This is what will be used for the test/train evaluation ====
        cvFoldIdxDict = {}
        for i in valHoldoutDict.keys():
            foldTrainIdx = [idx for idx in range(len(trainIndex)) if trainGroups[idx] not in valHoldoutDict[i]]
            foldTestIdx =  [idx for idx in range(len(trainIndex)) if trainGroups[idx]     in valHoldoutDict[i]]
            cvFoldIdxDict.update({i:{'Train':foldTrainIdx,'Test':foldTestIdx}})
        val_losses = []

        # Training loop ----
        for i in cvFoldIdxDict.keys():
            # TODO I suppose we should be rescaling all data here instead of outside
            #      to reduce information leaking from validataion -> training.
            #      If we don't it'll potentially cause underperformance instead of
            #      overperformance so this isn't a critical todo.
            x_train, x_test = x[cvFoldIdxDict[i]['Train']], x[cvFoldIdxDict[i]['Test']]
            y_train, y_test = y[cvFoldIdxDict[i]['Train']], y[cvFoldIdxDict[i]['Test']]

            model = self.hypermodel.build(trial.hyperparameters)
            
            fit_kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32) # <---------- 
            model.fit(x_train, y_train,
#                       batch_size= batch_size,
                      epochs=epochs
                     )
            val_losses.append(model.evaluate(x_test, y_test))
        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
        self.save_model(trial.trial_id, model)


# %%
def build_model(hp):
    num_layers = hp.Int("num_layers", 1, 7)
    
    adam_learning_rate = hp.Choice('learning_rate', [0.1, 0.01, 0.001, 0.0001]) 
    adam_learning_beta1 = hp.Float('beta1', 0.9, 0.9999)
    adam_learning_beta2 = hp.Float('beta2', 0.9, 0.9999) 
    
    sIn = tf.keras.Input(shape=(21), dtype='int32', name = 'sIn')
    sX = sIn    
    for i in range(num_layers):
        sX = layers.Dense(
            units= hp.Int("fc_units_"+str(i), 4, 64),        # <------
            activation = 'relu'
        )(sX)
        sX = layers.BatchNormalization()(sX)
        sX = layers.Dropout(
            rate = hp.Float("fc_dropout_"+str(i), 0, 0.99) # <------
            )(sX)
    x = sX
    yHat = layers.Dense(1, activation='linear', name='yHat')(x)

    
    model = keras.Model(inputs = [sIn], 
                        outputs = yHat)
    
    opt = tf.keras.optimizers.Adam(
        learning_rate = adam_learning_rate,
        beta_1 = adam_learning_beta1, #0.9,   # default = 0.9
        beta_2 = adam_learning_beta2  #0.9999 # default = 0.999
    )
    
    model.compile(
        optimizer= opt, 
        loss='mse', 
        metrics = ['mae']
    )
    return model

# %%
hp = kt.HyperParameters()

model = build_model(hp)

tuner = TunerGroupwiseCV(
    hypermodel=build_model,
    oracle=kt.oracles.BayesianOptimization(
        objective='val_loss',
        max_trials= maxTrials),    
    directory=".",
    project_name=projectName,
#     overwrite=True
)


# %%
tuner.get_best_hyperparameters(1)

# %%

# %%

# %%
# adapted from 9_keras-plumbing-test


# %% [markdown]
# ## To run after copying files from atlas

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
import keras

# %%

# %%

# %%
path = '../data/atlas/atlasOut/'

history=np.load(path+'history.npy',allow_pickle='TRUE').item()
model = keras.models.load_model(path+'test.keras')

val_data     = np.load(path+'val_data.npy',allow_pickle='TRUE')
val_targets  = np.load(path+'val_targets.npy',allow_pickle='TRUE')

test_data    = np.load(path+'test_data.npy',allow_pickle='TRUE')
test_targets = np.load(path+'test_targets.npy',allow_pickle='TRUE')

# %%
print(f"Val. MAE: {model.evaluate(val_data, val_targets)[1]:.2f}")
print(f"Test MAE: {model.evaluate(test_data, test_targets)[1]:.2f}")

# %%
import matplotlib.pyplot as plt
loss = history["mae"]
val_loss = history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss,     color='b', label="Training MAE")
plt.plot(epochs, val_loss, color='k', linestyle='--',label="Val. MAE")
# plt.axhline(y=naiveGuess,  color='r', linestyle='-', label="Naive Guess")

plt.title("Training and Test Mean Absolute Error")
plt.legend()
plt.show()

# %%
centerAndScale= pd.read_pickle(path+'centerAndScale.pkl')

# %%

# %%
predictions = model.predict(train_data)
pltPredictionsTrain= pd.DataFrame(predictions).rename(columns= {0: 'yHat'})
pltPredictionsTrain['y']= list(train_targets)
pltPredictionsTrain['Split'] = 'Train'
pltPredictionsTrain['Color'] = 'gray'

predictions = model.predict(val_data)
pltPredictionsVal= pd.DataFrame(predictions).rename(columns= {0: 'yHat'})
pltPredictionsVal['y']= list(val_targets)
pltPredictionsVal['Split'] = 'Val'
pltPredictionsVal['Color'] = 'lightgray'

predictions = model.predict(test_data)
pltPredictionsTest= pd.DataFrame(predictions).rename(columns= {0: 'yHat'})
pltPredictionsTest['y']= list(test_targets)
pltPredictionsTest['Split'] = 'Test'
pltPredictionsTest['Color'] = '#9b0e03'

pltPredictions = pd.concat([pltPredictionsTrain, pltPredictionsVal, pltPredictionsTest])


# undo center/scale
pltPredictions['y'] = (pltPredictions['y'] * float(centerAndScale.loc[centerAndScale.col == 'GrainYield', 'std'])) + float(centerAndScale.loc[centerAndScale.col == 'GrainYield', 'mean'])
pltPredictions['yHat'] = (pltPredictions['yHat'] * float(centerAndScale.loc[centerAndScale.col == 'GrainYield', 'std'])) + float(centerAndScale.loc[centerAndScale.col == 'GrainYield', 'mean'])


plt.scatter(pltPredictions.y, pltPredictions.yHat, color = pltPredictions.Color, alpha = 0.3)

# fig, axs = plt.subplots(1, 3, figsize=(10,5))

# axs[0].scatter(pltPredictionsTrain.y, pltPredictionsTrain.yHat)
# axs[1].scatter(pltPredictionsVal.y,   pltPredictionsVal.yHat)
# axs[2].scatter(pltPredictionsTest.y,  pltPredictionsTest.yHat)

# axs[0].set_title('Training')
# axs[1].set_title('Validation')
# axs[2].set_title('Test')

# %%
import seaborn as sns
sns.displot(pltPredictions, x="y", hue="Split", element="step")

# %%
pltPredictions['yDiff'] = pltPredictions['yHat'] - pltPredictions['y']
sns.displot(pltPredictions, x="yDiff", hue="Split", element="step")

# %%
sns.kdeplot(data=pltPredictions, x="yDiff", hue="Split", fill=True)



# %%
predictions = model.predict(test_data)
pltPredictions= pd.DataFrame(predictions).rename(columns= {0: 'yHat'})
pltPredictions['y']= list(test_targets)

pltPredictions

# %%
model.predict(test_data)

# %%

# %%

# %%
history1=np.load('history.npy',allow_pickle='TRUE').item()

# %%
history = history1
# .history

# %%
model = keras.models.load_model("test.keras")
print(f"Test MAE: {model.evaluate(val_data, val_targets)[1]:.2f}")
print(f"Test MAE: {model.evaluate(test_data, test_targets)[1]:.2f}")

# %%
model

# %%
history

# %%
import matplotlib.pyplot as plt
loss = history["mae"]
val_loss = history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss,     color='b', label="Training MAE")
plt.plot(epochs, val_loss, color='k', linestyle='--',label="Test MAE")
# plt.axhline(y=naiveGuess,  color='r', linestyle='-', label="Naive Guess")

plt.title("Training and Test Mean Absolute Error")
plt.legend()
plt.show()

# %%
predictions = model.predict(test_data)

pltPredictions= pd.DataFrame(predictions).rename(columns= {0: 'yHat'})

pltPredictions['y']= list(test_targets)

pltPredictions

# %%
plt.scatter(pltPredictions.y, pltPredictions.yHat)

# %%

# %%

# %%

# %%

# %%

# %% code_folding=[]
# load in data and tensorflow
import numpy as np
import tensorflow as tf
import keras

# path = '../data/atlas/atlasIn/'
# path = '../data/atlas/'
path = './'
num_epochs = 3
batch_size = 16

train_targets= np.load(path+'train_targets.npy')
train_data= np.load(path+'train_data.npy')
val_targets= np.load(path+'val_targets.npy')
val_data= np.load(path+'val_data.npy')

# model
model = keras.Sequential([
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1)
])

model.compile(optimizer="rmsprop", 
              loss="mse", 
              metrics=["mae"])

callbacks = [
    keras.callbacks.ModelCheckpoint("test.keras",
#     save_best_only=True)
                                   )
]

history = model.fit(train_data, train_targets, 
                    epochs=num_epochs, 
                    batch_size= batch_size, 
                    verbose=0,
                    validation_data = (val_data, val_targets),
                    callbacks=callbacks)

# save history. We'll get model from .keras file
np.save(path+'history.npy', history.history)

# %% code_folding=[]
# def build_model():
#     model = keras.Sequential([
#         keras.layers.Dense(64, activation="relu"),
#         keras.layers.Dense(64, activation="relu"),
#         keras.layers.Dense(1)
#     ])
#     model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
#     return model

# %% code_folding=[]
# # fit
# model = build_model()
# num_epochs = 10
# model.fit(train_data, train_targets, 
#           epochs=num_epochs, batch_size=16, verbose=0)

# test_mse, test_mae = model.evaluate(test_data, test_targets, verbose=0)
# test_mse

# %%

# %%

# %% [markdown]
# ## To run after copying files from atlas

# %%
history1=np.load('history.npy',allow_pickle='TRUE').item()

# %%
history = history1
# .history

# %%
model = keras.models.load_model("test.keras")
print(f"Test MAE: {model.evaluate(val_data, val_targets)[1]:.2f}")
print(f"Test MAE: {model.evaluate(test_data, test_targets)[1]:.2f}")

# %%
model

# %%
history

# %%
import matplotlib.pyplot as plt
loss = history["mae"]
val_loss = history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss,     color='b', label="Training MAE")
plt.plot(epochs, val_loss, color='k', linestyle='--',label="Test MAE")
# plt.axhline(y=naiveGuess,  color='r', linestyle='-', label="Naive Guess")

plt.title("Training and Test Mean Absolute Error")
plt.legend()
plt.show()

# %%
predictions = model.predict(test_data)

pltPredictions= pd.DataFrame(predictions).rename(columns= {0: 'yHat'})

pltPredictions['y']= list(test_targets)

pltPredictions

# %%
plt.scatter(pltPredictions.y, pltPredictions.yHat)

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
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

## Prepare the data ===================================================
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


## Build the model ====================================================
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

## Train the model ====================================================
batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.save(path)
## Evaluate the trained mode ==========================================
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


# %%
path = '../data/atlas/atlas/'

reconstructed_model = keras.models.load_model(path)

x_train= np.load(path+'x_train.npy')
y_train= np.load(path+'y_train.npy')
x_test= np.load(path+'x_test.npy')
y_test= np.load(path+'y_test.npy')


# np.testing.assert_allclose(
#     model.predict(x_train), reconstructed_model.predict(x_train)
# )

reconstructed_model.predict(x_train)
# reconstructed_model.fit(test_input, test_target)


# %%

# %%

# %%

# %%

# %% [markdown]
# Working with Atlas

# %%
#script
pathPrefix = '../data/atlas/'

import pandas as pd
import matplotlib.pyplot as plt
temp = pd.read_pickle(pathPrefix+'test.pkl')

plt.scatter(temp.GrainYield, temp.NitrogenPerAcre)

savefig(pathPrefix+'test.png', transparent=True)



# move dir 
# local -> atlas
# when repeated this copies within atlas... need to fix that.
# scp -r ~/Documents/BitBucket/MaizeModel/data/atlas/ daniel.kick@atlas-login.hpc.msstate.edu:/project/rover_pgru_wash/daniel/atlas/

# atlas -> local
# scp -r daniel.kick@atlas-login.hpc.msstate.edu:/project/rover_pgru_wash/daniel/atlas/ ~/Documents/BitBucket/MaizeModel/data/atlas/ 



# ssh atlas-login
# # cd /project/rover_pgru_wash/daniel/
# salloc -A scinet --partition=gpu -t 00:30:00 --chdir=/project/rover_pgru_wash/daniel/


# #!/bin/bash

# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH -A scinet
#SBATCH --time=96:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=40   # 20 processor core(s) per node X 2 threads per core
#SBATCH --partition=atlas    # standard node(s)

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load r/4.0.2
R --no-save < 3_ADxEBLUP_by_qgg.R



salloc -A scinet       -t 00:20:00 --chdir=/project/rover_pgru_wash/daniel/atlas/            
srun   -A scinet --pty -t 00:20:00 --chdir=/project/rover_pgru_wash/daniel/atlas/ /bin/bash -l 
module load singularity
singularity instance start ../tensorflow/tensorflow-21.07-tf2-py3.sif tf2py3
# singularity run instance://tf2py3 # No module named 'pandas' >:(
singularity exec instance://tf2py3 python keras_test.py

    
    
python keras_test.py

# salloc -A scinet       -t 00:30:00 --chdir=/project/rover_pgru_wash/daniel/tensorflow/              
# srun   -A scinet --pty -t 00:30:00 --chdir=/project/rover_pgru_wash/daniel/tensorflow/ /bin/bash -l 
# module load singularity 
# singularity instance start tensorflow-21.07-tf2-py3.sif tf2py3

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# Adapt code from AtlasLabNotebook

# after scp from Atlas -> Local
pathData = '../data/atlas/data/processed/'

def quick_vis_atlas(pathData = '../data/atlas/data/processed/',
                    path = '../data/atlas/models/00_G_2L/'
):

    runSettings = json.load(open(path+'runSettings.txt'))
    splitIndex = runSettings['splitIndex']

    # Load relevant data -----------------------------------------------------------
    indexDictList = json.load(open(pathData+'indexDictList.txt'))
    Y = np.load(pathData+'Y.npy')
    G = np.load(pathData+'G.npy')
    # S = np.load(pathData+'S.npy')
    # W = np.load(pathData+'W.npy')

    trainIndex = indexDictList[splitIndex]['Train']
    validateIndex = indexDictList[splitIndex]['Validate']
    testIndex = indexDictList[splitIndex]['Test']

    YStd = np.load(path+'YStd.npy')
    YMean = np.load(path+'YMean.npy')
    GStd = np.load(path+'GStd.npy')
    GMean = np.load(path+'GMean.npy')
    # SStd = np.load(path+'SStd.npy')
    # SMean = np.load(path+'SMean.npy')
    # WStd = np.load(path+'WStd.npy')
    # WMean = np.load(path+'WMean.npy')

    YNorm = ((Y - YMean) / YStd)
    GNorm = ((G - GMean) / GStd)
    # SNorm = ((S - SMean) / SStd)
    # WNorm = ((W - WMean) / WStd)

    # After we have output
    history=np.load(path+'history.npy',allow_pickle='TRUE').item()
    model = keras.models.load_model(path+'callbacks.keras')

    print(f"Val. MAE: {model.evaluate(GNorm[validateIndex], YNorm[validateIndex])[1]:.2f}")
    print(f"Test MAE: {model.evaluate(GNorm[testIndex], YNorm[testIndex])[1]:.2f}")





    yHat = model.predict(GNorm[testIndex])
    y = YNorm[testIndex]

    yHat = (yHat*YStd)+YMean
    y = (y*YStd)+YMean

    plt.scatter(y, yHat)
    x=np.linspace(min([min(y), min(yHat)]), max([max(y), max(yHat)]), 100)
    plt.plot(x, x, '-r', label='y=x')










    naiveGuess = np.mean(abs(YNorm[validateIndex] - YNorm[trainIndex].mean()))

    loss = history["mae"]
    val_loss = history["val_mae"]
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss,     color='b', label="Training MAE")
    plt.plot(epochs, val_loss, color='k', linestyle='--',label="Validation MAE")
    plt.axhline(y=naiveGuess,  color='r', linestyle='-', label="Naive Guess")

    plt.title("Training and Test Mean Absolute Error")
    plt.legend(loc = 'lower right')
    plt.show()

# %%
quick_vis_atlas(path = '../data/atlas/models/S_2/')
