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
import os, re, json
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.express as px

import seaborn as sns
import matplotlib.pyplot as plt


# %%
os.listdir("../data/atlas/models/")

# %%

#TODO mv baseline up here




# %% [markdown]
# Benchmark

# %%
splitIndex = 0
needGpca = True

# Index prep -------------------------------------------------------------------
path     = '../data/atlas/' #atlas version
pathData = path+'data/processed/'

# get the right index set automatically.
# if needGpca == True:
#     indexDictList = json.load(open(pathData+'indexDictList_PCA.txt')) # <--- Changed to PCA
# else:
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

# "Model" with point estimates ------------------------------------------------
out_part0 = pd.DataFrame({'y':YNorm[trainIndex]})
out_part0['model_type'] = 'naive_mean'
out_part0['split'] = 'train'
out_part0['yhat'] = np.mean(YNorm[trainIndex])

out_part1 = pd.DataFrame({'y':YNorm[trainIndex]})
out_part1['model_type'] = 'naive_median'
out_part1['split'] = 'train'
out_part1['yhat'] = np.median(YNorm[trainIndex])


out_part2 = pd.DataFrame({'y':YNorm[testIndex]})
out_part2['model_type'] = 'naive_mean'
out_part2['split'] = 'test'
out_part2['yhat'] = np.mean(YNorm[trainIndex])

out_part3 = pd.DataFrame({'y':YNorm[testIndex]})
out_part3['model_type'] = 'naive_median'
out_part3['split'] = 'test'
out_part3['yhat'] = np.median(YNorm[trainIndex])


baseline_part0 = pd.concat([out_part0, out_part1, out_part2, out_part3])
baseline_part0['YStd']  = YStd
baseline_part0['YMean'] = YMean

# -----------------------------------------------------------------------------
#                           Require genome present                            #
# -----------------------------------------------------------------------------

indexDictList = json.load(open(pathData+'indexDictList_PCA.txt')) # <--- Changed to PCA

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

# "Model" with point estimates ------------------------------------------------
out_part0 = pd.DataFrame({'y':YNorm[trainIndex]})
out_part0['model_type'] = 'Gpca_naive_mean'
out_part0['split'] = 'train'
out_part0['yhat'] = np.mean(YNorm[trainIndex])

out_part1 = pd.DataFrame({'y':YNorm[trainIndex]})
out_part1['model_type'] = 'Gpca_naive_median'
out_part1['split'] = 'train'
out_part1['yhat'] = np.median(YNorm[trainIndex])


out_part2 = pd.DataFrame({'y':YNorm[testIndex]})
out_part2['model_type'] = 'Gpca_naive_mean'
out_part2['split'] = 'test'
out_part2['yhat'] = np.mean(YNorm[trainIndex])

out_part3 = pd.DataFrame({'y':YNorm[testIndex]})
out_part3['model_type'] = 'Gpca_naive_median'
out_part3['split'] = 'test'
out_part3['yhat'] = np.median(YNorm[trainIndex])


baseline_part1 = pd.concat([out_part0, out_part1, out_part2, out_part3])
baseline_part1['YStd']  = YStd
baseline_part1['YMean'] = YMean


baseline = pd.concat([baseline_part0, baseline_part1])
baseline['replicate'] = 'rep0'
baseline['error'] = baseline['y'] - baseline['yhat']

# %%
baseline


# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
# all_model_res.json contains the y/yhat info for training AND testing entries
def _retrieve_tidy_data(path = '../data/atlas/models/Gpca_finalize_model/all_model_res.json'):
    with open(path) as f:
        data = json.load(f)  

    def _tidy_results(repId): #repId = 'rep1'
        temp = data[repId]

        out_part = pd.concat([
            pd.DataFrame(
            {'split' : ['train' for train in range(len(temp['y_train']))],
             'y'     : temp['y_train'],
             'yhat'  : temp['yhat_train']
            }),
        pd.DataFrame(
            {'split' : ['test' for test in range(len(temp['y_test']))],
             'y'     : temp['y_test'],
             'yhat'  : temp['yhat_test']
            })
        ])
        out_part['replicate'] = repId
        return(out_part)

    # make sure that replication 0 is first
    keys_list = list(data.keys())
    keys_list.sort()
    # get as list and concatenate
    tidy_data = pd.concat([_tidy_results(repId) for repId in keys_list])
    tidy_data['error'] = tidy_data['y']-tidy_data['yhat']
    return(tidy_data)


# %%
# # all_model_res.json contains the y/yhat info for training AND testing entries
# def _retrieve_tidy_data(path = '../data/atlas/models/Gpca_finalize_model/all_model_res.json'):
#     with open(path) as f:
#         data = json.load(f)  

#     def _tidy_results(repId): #repId = 'rep1'
#         temp = data[repId]

#         out_part = pd.concat([
#             pd.DataFrame(
#             {'split' : ['train' for train in range(len(temp['y_train']))],
#              'y'     : temp['y_train'],
#              'yhat'  : temp['yhat_train']
#             }),
#         pd.DataFrame(
#             {'split' : ['test' for test in range(len(temp['y_test']))],
#              'y'     : temp['y_test'],
#              'yhat'  : temp['yhat_test']
#             })
#         ])
#         out_part['replicate'] = repId
#         return(out_part)

#     # make sure that replication 0 is first
#     keys_list = list(data.keys())
#     keys_list.sort()
#     # get as list and concatenate
#     tidy_data = pd.concat([_tidy_results(repId) for repId in keys_list])
#     tidy_data['error'] = tidy_data['y']-tidy_data['yhat']
#     return(tidy_data)

path = "../data/atlas/models/7_concat_v0_finalize_model/all_model_res.json"
with open(path) as f:
    data = json.load(f)


# %%

# %%

# %%
model_dir = [
    "3_finalize_model_syr_G",
    "3_finalize_model_syr_S",
    "3_finalize_model_syr_W"
#     "3_Gpca", "3_S", "3_Wmlp", "3_Wc1d", "3_Wc2d",
#     "7_concat_v0"
              ]
# tweaked to allow for non-standard naming
model_types_paths = ['../data/atlas/models/'+model_type+'/all_model_res.json' for model_type in model_dir]
# model_types_paths.append('../data/atlas/models/'+'7_concat_v0_finalize_model_no_drop'+'/all_model_res.json')
# model_types.append('7_concat_v0_no_drop')

# tidy_results = [_retrieve_tidy_data(path = '../data/atlas/models/'+model_type+'_finalize_model/all_model_res.json'
#                                    ).assign(model_type = model_type) for model_type in model_types]

# %%
i = 0

_retrieve_tidy_data(path = model_types_paths[i])

# %%
tidy_results = [_retrieve_tidy_data(path = model_types_paths[i]
                                   ).assign(model_type = model_types[i]) for i in range(len(model_types))]

# %%
print('Format Example')
tidy_results[3]

# %%
# Add baseline info into tidy results
# model_types
baseline_model_types = list(baseline.model_type.drop_duplicates())

model_types.extend(baseline_model_types)

# %%
for b_m_t  in baseline_model_types:
    tidy_results.extend([ baseline.loc[baseline.model_type == b_m_t, ] ])

# %%
# baseline_model_types

# %%
# baseline.loc[baseline.model_type == b_m_t, ]

# %%
# asdf


# %%
for i in range(len(tidy_results)):
    print(i)
    tidy_data = tidy_results[i]
    tidy_data= tidy_data.loc[tidy_data['replicate'] == 'rep0', ]
    sns.set_theme(style="whitegrid")

    # Draw a nested violinplot and split the violins for easier comparison
    sns.violinplot(data=tidy_data, 
                   x="replicate", 
                   y="error", 
                   hue="split",
                   split=True, inner="quart", linewidth=1.5,
                   palette={"train": sns.color_palette("Paired")[0], 
                             "test": sns.color_palette("Paired")[2]}
                  )
    sns.despine(left=True)
#     plt.legend(loc = 2, 
#                bbox_to_anchor = (1,1))
    plt.title(model_types[i])
#     plt.xlim(-4, 4)
    sns.set(rc={"figure.figsize":(5, 3)})
    plt.savefig('./'+str(model_types[i])+'_split_error.svg', dpi=350)
#     plt.show()
    plt.close()

# %%


# %%

# %%
import matplotlib.pyplot as plt

temp =[xx.loc[xx['replicate'] == 'rep0', ] for xx in tidy_results]

for i in range(len(temp)):
    if i == 0:
        out = temp[i]
    else:
        out = out.merge(temp[i], how  = 'outer')
        

mask = out['model_type'].isin([
    'Gpca_naive_mean',
    '3_Gpca',
     '3_S',
     '3_Wmlp',
     '3_Wc1d',
     '3_Wc2d',
     '7_concat_v0',
     '7_concat_v0_no_drop',
])


out = out.loc[mask, :]

out = out.merge(
    pd.DataFrame(zip(
        ['Gpca_naive_mean',    '3_Gpca',    '3_S',    '3_Wmlp',    '3_Wc1d',    '3_Wc2d',    '7_concat_v0', '7_concat_v0_no_drop'], 
        ['Intercept Model', 'Genomic MLP', 'Soil MLP', 'W&M MLP', 'W&M CNN 1D', 'W&M CNN 2D', 'Full Network', 'Full Network\nDropout 0%'],
        [0, 1, 2, 3, 4, 5, 6, 7]), columns = ['model_type', 'Model', 'new_rank'])
    ).sort_values('new_rank')

out = out.rename(columns = {'error':'Error',
                     'split':'Set'})
out.loc[out['Set'] == 'train', 'Set'] = 'Train'
out.loc[out['Set'] == 'test', 'Set'] = 'Test'

out = out.merge(pd.DataFrame(zip(['Train', 'Test'],
                 [1, 2]), columns = ['Set', 'new_rank2']))
out = out.sort_values(['new_rank', 'new_rank2'])


xx = 1.5
sns.set_theme(style="whitegrid")
sns.violinplot(
    data=out, 
    x="Model", 
    y="Error", 
    hue="Set",
    split=True, inner="quart", linewidth=1.5,
    palette={"Train": sns.color_palette("Paired")[0], 
             "Test": sns.color_palette("Paired")[2]}
    ).set(title='Model Performance Comparison')


sns.despine(left=True)
sns.set(rc={"figure.figsize":(xx*5, xx*3)})
plt.xticks(rotation = 45)
plt.legend(loc='lower right')
# plt.set_xticklabels(plt.get_xticklabels(), rotation=45, horizontalalignment='right')
# plt.title('Model Performance Comparison')
plt.savefig('./'+'Overall_split_error.svg', dpi=350)


# %%




# %%



# %%
import matplotlib.pyplot as plt

temp =[xx.loc[xx['replicate'] == 'rep0', ] for xx in tidy_results]

for i in range(len(temp)):
    if i == 0:
        out = temp[i]
    else:
        out = out.merge(temp[i], how  = 'outer')
        

mask = out['model_type'].isin([

     '7_concat_v0',
     '7_concat_v0_no_drop',
])


out = out.loc[mask, :]

out = out.merge(
    pd.DataFrame(zip(
        ['7_concat_v0',    '7_concat_v0_no_drop'], 
        ['Full Network', 'Full Network\nDropout 0%'],
        [0, 1]), columns = ['model_type', 'Model', 'new_rank'])
    ).sort_values('new_rank')

out = out.rename(columns = {'error':'Error',
                     'split':'Set'})
out.loc[out['Set'] == 'train', 'Set'] = 'Train'
out.loc[out['Set'] == 'test', 'Set'] = 'Test'

out = out.merge(pd.DataFrame(zip(['Train', 'Test'],
                 [1, 2]), columns = ['Set', 'new_rank2']))
out = out.sort_values(['new_rank', 'new_rank2'])


xx = 1.5
sns.set_theme(style="whitegrid")
sns.violinplot(
    data=out, 
    x="Model", 
    y="Error", 
    hue="Set",
    split=True, inner="quart", linewidth=1.5,
    palette={"Train": sns.color_palette("Paired")[0], 
             "Test": sns.color_palette("Paired")[2]}
    ).set(title='Effect of Eliminating Dropout')


sns.despine(left=True)
sns.set(rc={"figure.figsize":(xx*5, xx*3)})
plt.xticks(rotation = 45)
plt.legend(loc='lower right')
# plt.set_xticklabels(plt.get_xticklabels(), rotation=45, horizontalalignment='right')
# plt.title('Model Performance Comparison')
plt.savefig('./'+'Dropout_split_error.svg', dpi=350)

# %%

# %%

# %%

# %%

# %% [markdown]
# Benchmark

# %%
# splitIndex = 0
# needGpca = True

# # Index prep -------------------------------------------------------------------
# path     = '../data/atlas/' #atlas version
# pathData = path+'data/processed/'

# # get the right index set automatically.
# # if needGpca == True:
# #     indexDictList = json.load(open(pathData+'indexDictList_PCA.txt')) # <--- Changed to PCA
# # else:
# indexDictList = json.load(open(pathData+'indexDictList.txt'))

# trainIndex    = indexDictList[splitIndex]['Train']
# trainGroups   = indexDictList[splitIndex]['TrainGroups']
# testGroups    = indexDictList[splitIndex]['TestGroups'] 
# testIndex     = indexDictList[splitIndex]['Test']

# # Data prep -------------------------------------------------------------------
# # Y ===========================================================================
# Y = np.load(pathData+'Y.npy')
# YStd = Y[trainIndex].std()
# YMean = Y[trainIndex].mean()
# YNorm = ((Y - YMean) / YStd)

# # "Model" with point estimates ------------------------------------------------
# out_part = pd.DataFrame({'y':YNorm[testIndex]})
# out_part['model_type'] = 'naive_mean'
# out_part['yhat'] = np.mean(YNorm[testIndex])

# out_part1 = pd.DataFrame({'y':YNorm[testIndex]})
# out_part1['model_type'] = 'naive_median'
# out_part1['yhat'] = np.median(YNorm[testIndex])

# baseline_part = pd.concat([out_part, out_part1])
# baseline_part['YStd']  = YStd
# baseline_part['YMean'] = YMean

# # -----------------------------------------------------------------------------
# #                           Require genome present                            #
# # -----------------------------------------------------------------------------

# indexDictList = json.load(open(pathData+'indexDictList_PCA.txt')) # <--- Changed to PCA

# trainIndex    = indexDictList[splitIndex]['Train']
# trainGroups   = indexDictList[splitIndex]['TrainGroups']
# testGroups    = indexDictList[splitIndex]['TestGroups'] 
# testIndex     = indexDictList[splitIndex]['Test']

# # Data prep -------------------------------------------------------------------
# # Y ===========================================================================
# Y = np.load(pathData+'Y.npy')
# YStd = Y[trainIndex].std()
# YMean = Y[trainIndex].mean()
# YNorm = ((Y - YMean) / YStd)

# # "Model" with point estimates ------------------------------------------------
# out_part = pd.DataFrame({'y':YNorm[testIndex]})
# out_part['model_type'] = 'Gpca_naive_mean'
# out_part['yhat'] = np.mean(YNorm[testIndex])

# out_part1 = pd.DataFrame({'y':YNorm[testIndex]})
# out_part1['model_type'] = 'Gpca_naive_median'
# out_part1['yhat'] = np.median(YNorm[testIndex])

# baseline_part1 = pd.concat([out_part, out_part1])
# baseline_part1['YStd']  = YStd
# baseline_part1['YMean'] = YMean

# baseline = pd.concat([baseline_part, baseline_part1])
# baseline['error'] = baseline['y'] - baseline['yhat']

# %%
# baseline

# %%

# %%
tidy_results[3]


# %%
# all_model_res.json contains the y/yhat info for training AND testing entries
def _retrieve_tidy_data(path = '../data/atlas/models/Gpca_finalize_model/all_model_res.json'):
    with open(path) as f:
        data = json.load(f)  

    def _tidy_results(repId): #repId = 'rep1'
        temp = data[repId]

        out_part = pd.concat([
            pd.DataFrame(
            {'split' : ['train' for train in range(len(temp['y_train']))],
             'y'     : temp['y_train'],
             'yhat'  : temp['yhat_train']
            }),
        pd.DataFrame(
            {'split' : ['test' for test in range(len(temp['y_test']))],
             'y'     : temp['y_test'],
             'yhat'  : temp['yhat_test']
            })
        ])
        out_part['replicate'] = repId
        return(out_part)

    # make sure that replication 0 is first
    keys_list = list(data.keys())
    keys_list.sort()
    # get as list and concatenate
    tidy_data = pd.concat([_tidy_results(repId) for repId in keys_list])
    tidy_data['error'] = tidy_data['y']-tidy_data['yhat']
    return(tidy_data)


# %%

# %%

# %%
def np_q10(x): return(np.percentile(x, 10, axis=0))
def np_q20(x): return(np.percentile(x, 20, axis=0))
def np_q30(x): return(np.percentile(x, 30, axis=0))
def np_q40(x): return(np.percentile(x, 40, axis=0))
def np_q50(x): return(np.percentile(x, 50, axis=0))
def np_q60(x): return(np.percentile(x, 60, axis=0))
def np_q70(x): return(np.percentile(x, 70, axis=0))
def np_q80(x): return(np.percentile(x, 80, axis=0))
def np_q90(x): return(np.percentile(x, 90, axis=0))


# %%

# %%

# %%
summary_list = []

for i in range(len(tidy_results)):
    tidy_data = tidy_results[i]
    temp = tidy_data.groupby(['replicate', 'split']).agg(
    mean_error = ('error', np.mean),
    q10 = ('error', np_q10),
    q20 = ('error', np_q20),
    q30 = ('error', np_q30),
    q40 = ('error', np_q40),
    q50 = ('error', np_q50),
    q60 = ('error', np_q60),
    q70 = ('error', np_q70),
    q80 = ('error', np_q80),
    q90 = ('error', np_q90)
    ).reset_index()
    
    temp['model_type'] = list(tidy_data['model_type'].drop_duplicates())[0]
    
    summary_list.append(temp)
    
summary = pd.concat(summary_list)

# %%
px.line(summary.sort_values(['model_type', 'split'], ascending=[True, False]), 
        x = 'split', y = 'mean_error', 
        color = 'replicate',
        facet_col = 'model_type')

# %%

# %%

# %%

# %%

# %%

# %%
summary_list = []

for i in range(len(tidy_results)):
    tidy_data = tidy_results[i]
    temp = tidy_data.groupby([#'replicate', 
                              'split']).agg(
    mean_error = ('error', np.mean),
    q10 = ('error', np_q10),
    q20 = ('error', np_q20),
    q30 = ('error', np_q30),
    q40 = ('error', np_q40),
    q50 = ('error', np_q50),
    q60 = ('error', np_q60),
    q70 = ('error', np_q70),
    q80 = ('error', np_q80),
    q90 = ('error', np_q90)
    ).reset_index()
    
    temp['model_type'] = list(tidy_data['model_type'].drop_duplicates())[0]
    
    summary_list.append(temp)
    
summary = pd.concat(summary_list)

# %%
summary

# %%
fig = go.Figure()


list_model_types = list(summary.model_type.drop_duplicates())

for i in range(len(list_model_types)):
#     model_type_i = "Gpca"
    model_type_i = list_model_types[i]
    
    mask = (summary.model_type == model_type_i) 
    temp = summary.loc[mask,]

    temp = pd.DataFrame(
        {'split' : ['train', 'test'],
         'left'  : [i      , i+0.5 ],
         'center': [i+0.25 , i+0.75],
         'right' : [i+0.5  , i+1]}).merge(temp)


    
    train_fill  = 'rgba(26,150,165,0.1)'
    train_color = 'rgba(26,150,165,1)'
    test_fill   = 'rgba(150,50,65,0.1)'
    test_color  = 'rgba(150,50,65,1)'

    # layer percentiles
    qPairsLower = ['q10', 'q20', 'q30', 'q40']
    qPairsUpper = ['q90', 'q80', 'q70', 'q60']
    for j in range(len(qPairsLower)):
        fig.add_shape(type="rect",
            x0=float(temp.loc[temp.split == 'train', 'left']), 
            x1=float(temp.loc[temp.split == 'train', 'right']), 
            y0=float(temp.loc[temp.split == 'train', qPairsLower[j]]), 
            y1=float(temp.loc[temp.split == 'train', qPairsUpper[j]]),
            line=dict(width=0), 
            fillcolor=train_fill,
        )    

        fig.add_shape(type="rect",
            x0=float(temp.loc[temp.split == 'test', 'left']), 
            x1=float(temp.loc[temp.split == 'test', 'right']), 
            y0=float(temp.loc[temp.split == 'test', qPairsLower[j]]), 
            y1=float(temp.loc[temp.split == 'test', qPairsUpper[j]]),
            line=dict(width=0), 
            fillcolor=test_fill,
        )


    # add in point estimates
    for j in range(2):
        fig.add_shape(type="line",
                x0=float(temp.loc[temp.split == 'train', 'left']), 
                x1=float(temp.loc[temp.split == 'train', 'right']), 
                y0=float(temp.loc[temp.split == 'train', ['mean_error', 'q50'][j] ]), 
                y1=float(temp.loc[temp.split == 'train', ['mean_error', 'q50'][j] ]),
                line=dict(width=1, color=train_color, dash=["dash", None][j])
            )
        fig.add_shape(type="line",
            x0=float(temp.loc[temp.split == 'test', 'left']), 
            x1=float(temp.loc[temp.split == 'test', 'right']), 
            y0=float(temp.loc[temp.split == 'test', ['mean_error', 'q50'][j] ]), 
            y1=float(temp.loc[temp.split == 'test', ['mean_error', 'q50'][j] ]),
            line=dict(width=1, color=test_color, dash=["dash", None][j] )
        )
    
    fig.add_annotation(x=i+0.5, y=3, xref="x", yref="y", text= model_type_i, showarrow=False, align="center", 
                       font=dict(# size=16, 
                           color="#000000"), bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#ffffff")   
    fig.add_annotation(x=i+0.25, y=2, xref="x", yref="y", text= 'Train', 
                       showarrow=False, align="center", 
                   font=dict(# size=16, 
                       color="#000000"), bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#ffffff")   
    fig.add_annotation(x=i+0.75, y=2, xref="x", yref="y", text= 'Test', 
                       showarrow=False, align="center", 
               font=dict(# size=16, 
                   color="#000000"), bordercolor="#c7c7c7", borderwidth=2, borderpad=4, bgcolor="#ffffff")   
    

    
fig.add_shape(type="line",
        x0=0,
        x1=len(list_model_types),
        y0=float(summary.loc[
            ((summary.model_type == model_type_i) & (summary.split == 'test')), 'mean_error']), 
        y1=float(summary.loc[((summary.model_type == model_type_i) & (summary.split == 'test')), 'mean_error']),
        line=dict(width=1, color='rgba(0,0,0, 1)', dash="dashdot" )
    )
    
fig.update_layout(
    template='plotly_white', 
    title="",
    xaxis_range=[0,9])

fig.update_shapes(dict(xref='x', yref='y'))

fig.show()

# %%
# what about MAE?



# %%

# %%
# which weather should we use?

temp = pd.concat(tidy_results).reset_index().drop(columns = 'index')


mask = (temp.split == 'test') & (temp.model_type.isin(['Wmlp', 'Wc1d', 'Wc2d']))
sns.kdeplot(data=temp.loc[mask], 
            x="error", 
            hue="model_type"
           )

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
