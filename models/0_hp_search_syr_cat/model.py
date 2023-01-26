# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import shutil 
import os
import re
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
# -
# ## Cuda settings:

# +
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
# -


# ## Data Prep:

# +
# Directory prep -------------------------------------------------------------- 
## Automatic file manipulaiton logic goes here ================================       
# clean out logs
# if os.path.exists('./logs/training_log.csv'): 
#     os.remove('./logs/training_log.csv')
# if os.path.exists('./logs/tb_logs'):
#     shutil.rmtree('./logs/tb_logs') # likely has files in it which is why we don't use os.rmdir().

# +
## Load additional options files here =========================================       
# check that everthing is okay. otherwise write out an error file with exception -- naming inconsistency. 
# modelDir = os.getcwd().split(os.path.sep)[-1]
# use_cvFold_idx = int(modelDir.split("_")[-1].strip("k"))

# +
# Specified Options -----------------------------------------------------------
## General ====
projectName = 'kt' # this is the name of the folder which will be made to hold the tuner's progress.
hp = kt.HyperParameters()
## Tuning ====
splitIndex = 0
maxTrials  = 40  
numEpochs  = 1000 
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


# +
# Index prep -------------------------------------------------------------------
path     = '../../' #atlas version
pathData = path+'data/processed/'

indexDictList = json.load(open(pathData+'indexDictList_syr.txt')) 

trainIndex    = indexDictList[splitIndex]['Train']
trainGroups   = indexDictList[splitIndex]['TrainGroups']
testGroups    = indexDictList[splitIndex]['TestGroups'] 
testIndex     = indexDictList[splitIndex]['Test']
# -

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

"""
Additional Testing:
Has the fold definition code messed up these indices? 
"""
for key in cvFoldIdxDict.keys():
    nan_Train = np.isnan(G[cvFoldIdxDict[key]['Train']]).sum()
    nan_Val   = np.isnan(G[cvFoldIdxDict[key]['Test']]).sum()
    print('key', key, ': NaN in Train: ', nan_Train, ', NaN in Test: ', nan_Val, sep = '')
    assert 0 == nan_Train == nan_Val, 'Nans found in dataset'

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


# ## Hypermodel Definition:

# +
def _reload_and_freeze_keras_model(data_type = 'G', trainable = False):
    rel_path = './finalized_models/'+data_type+'/'
    mod_name = [e for e in os.listdir(rel_path) if re.match('.*.h5', e)]
    assert len(mod_name) == 1

    trained_model = keras.models.load_model(rel_path+mod_name[0])
    if not trainable:
        trained_model.trainable = False
    return(trained_model)

Gpca_model = _reload_and_freeze_keras_model(data_type = 'G', trainable = False)
S_model = _reload_and_freeze_keras_model(data_type = 'S', trainable = False)
W_model = _reload_and_freeze_keras_model(data_type = 'W', trainable = False)
# -





# + code_folding=[]
# instead of building the model first then adding kt, let's work out from kt
def build_model(hp):
    ## Global hyperparameters ==================================================
    adam_learning_rate = hp.Choice('learning_rate', [0.1, 0.01, 0.001, 0.0001])
    adam_learning_beta1 = hp.Float('beta1', 0.9, 0.9999)
    adam_learning_beta2 = hp.Float('beta2', 0.9, 0.9999) 

    ## Subnetwork hyperparameters ==============================================
    g_unit_min = 4
    g_unit_max = 256 #1725
    g_dropout_max = 0.3

    s_unit_min = 4
    s_unit_max = 64
    s_dropout_max = 0.3

    w_filter_min = 4
    w_filter_max = 512

    cat_unit_min = 4
    cat_unit_max = 256 #1024
    cat_dropout_max = 0.3
    
    ## Inputs ==================================================================
    gIn = tf.keras.Input(shape=(1725),       dtype='float32', name = 'gIn')
    sIn = tf.keras.Input(shape=(21),         dtype='float32', name = 'sIn')
#     wIn = tf.keras.Input(shape=(288, 19, 1), dtype = 'float32', name = 'wIn')
    wIn = tf.keras.Input(shape=(288, 19), dtype='float32', name = 'wIn')
    # Subnetworks --------------------------------------------------------------
    ## G PCA ===================================================================
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
    """
    Note! This is messier than in previous versions. This is because to have a similar 
    stucture to the full model, each sub model takes all tensors as an input. As a byproduct
    we can't select the penultimate layer because unused input tensors are at the end of the list, 
    just before yHat. Instead we have to look at the last layer (yHat) and find the inbound node 
    name. Then we can look up the output of that node. 
    
    Before we could just use `Gpca_model.layers[-2].output`
    """
    # in each of these this _should_ work because there's only one input to the yHat layer
    Gpca_yHat_input_layer = Gpca_model.get_config()['layers'][-1]['inbound_nodes'][0][0][0]
    Gpca_subnetwork_out = keras.models.Model(
        inputs=Gpca_model.input, 
        outputs=Gpca_model.get_layer(Gpca_yHat_input_layer).output)
    gOut = Gpca_subnetwork_out(inputs=[gIn, sIn, wIn])
    
    ## S =======================================================================
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
    S_yHat_input_layer = S_model.get_config()['layers'][-1]['inbound_nodes'][0][0][0]
    S_subnetwork_out = keras.models.Model(
        inputs=S_model.input, 
        outputs=S_model.get_layer(S_yHat_input_layer).output)
    sOut = S_subnetwork_out(inputs=[gIn, sIn, wIn])    
    
    ## W c1d ===================================================================
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
    W_yHat_input_layer = W_model.get_config()['layers'][-1]['inbound_nodes'][0][0][0]
    W_subnetwork_out = keras.models.Model(
        inputs=W_model.input, 
        outputs=W_model.get_layer(W_yHat_input_layer).output)
    wOut = W_subnetwork_out(inputs=[gIn, sIn, wIn])
    
    ## Interaction layers ======================================================
    x_num_layers = hp.Int("x_num_layers", 1, 7)
    cX = keras.layers.concatenate([gOut, sOut, wOut ])   
    
    for i in range(x_num_layers):
        cX = layers.Dense(
            units= hp.Int("x_fc_units_"+str(i), cat_unit_min, cat_unit_max),
            activation = 'relu'
        )(cX)
        cX = layers.BatchNormalization()(cX)
        cX = layers.Dropout(
                rate = hp.Float("x_fc_dropout_"+str(i), 0.0, cat_dropout_max)   
            )(cX)        
    x = cX        
    yHat = layers.Dense(1, activation='linear', name='yHat')(x) 
    
    ## Setup ====
    model = keras.Model(inputs=[gIn, sIn, wIn], outputs=[yHat])

    opt = tf.keras.optimizers.Adam(
        learning_rate = adam_learning_rate,
        beta_1 = adam_learning_beta1, 
        beta_2 = adam_learning_beta2  
    )

    model.compile(
        optimizer= opt,
        loss='mse',
        metrics = ['mae']
    )
    return model  

model = build_model(hp)


# -
class CVTuner(kt.engine.tuner.Tuner):
    def run_trial(self, trial, 
                  foldwise_data_list,
                  epochs,
                  batch_size,
                  *fit_args, **fit_kwargs
                 ):
        
        model = self.hypermodel.build(trial.hyperparameters)
        fit_kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=16)  
        
        val_losses = []
        add_metrics_dict = {}
        # for i in range(len(foldwise_data_list)): # previously we moved over all the folds. This 10x the training time. 
        i = k_cv_rng.choice([i for i in range(len(foldwise_data_list))], 1)[0] # randomly select a fold to train on    
        print(i, 'fold')
        x_train = foldwise_data_list[i]['train']['x']
        y_train = foldwise_data_list[i]['train']['y']
        x_test  = foldwise_data_list[i]['test' ]['x']
        y_test  = foldwise_data_list[i]['test' ]['y']
        model = self.hypermodel.build(trial.hyperparameters)
        model.fit(x_train, y_train, 
                  batch_size=batch_size, 
                  epochs=epochs
                 )
        model_fold_loss = model.evaluate(x_test, y_test)
        print("mfl", model_fold_loss)
        val_losses.append(model_fold_loss[0]) # loss is the first metic, 
                                              # mae is the second. See the model definition.
        add_metrics_dict.update({"k"+str(i)+"_loss": model_fold_loss[0],
                                 "k"+str(i)+"_mae" : model_fold_loss[1]})
        
        add_metrics_dict.update({'val_loss': np.nanmean(val_losses)}) #using nanmean here just in case something goes off the rails 
        self.oracle.update_trial(trial.trial_id, add_metrics_dict)
        self.save_model(trial.trial_id, model)
# ## Train

# +
tuner = CVTuner(
    hypermodel=build_model,
    oracle=kt.oracles.BayesianOptimization(
        objective='val_loss',
        max_trials=maxTrials
    ),
    # directory="C:/Users/drk8b9/rm_me", #for windows
    directory=".", 
    project_name = projectName,
    overwrite=False
)

tuner.search(
    foldwise_data_list,
    batch_size = 20, # <-  this should be an int, but it'll get overwritten during the training.
    epochs = numEpochs,
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience= 7),
        keras.callbacks.CSVLogger("/training_log.csv", append=True)
    ]
)

# -


