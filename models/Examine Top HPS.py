# -*- coding: utf-8 -*-
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
import shutil 
import os
import re
import json

import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

#▁▂▃▄▅▆▇█


# %%
project_path = './0_hp_search_syr_G/'
project_path = './0_hp_search_syr_S/'
project_path = './0_hp_search_syr_W/'
project_path = './0_hp_search_syr_full/'
project_path = './0_hp_search_syr_cat/'

history_files = [e for e in os.listdir(project_path+'eval/') if re.match('hps_rank_\d_fold_\d_history.json', e)]
hps_files = [e for e in os.listdir(project_path+'eval/') if re.match('hps_rank_\d.json', e)]


# %%
# os.listdir()

# %%
def _hps_to_df(json_file):
    with open(project_path+'eval/'+json_file) as f:
        dat = json.load(f)
    dat = pd.DataFrame(dat, index = [0])
    
    # Get rid of hps that are not actually used (hps in layers that are not added to the model)
    col_names = list(dat)
    # convolution type is interfereing with the code
    # everything should be convertable to a float except for any bool/choice hps.
    # Test if the value can be a float and if not then we modify the key/column 
    # name to include the value and code it as 1. In effect we're one hot encoding 
    # catagorical hps. 
    for col_name in col_names:
        try:
            float(dat[col_name])
        except:
            # update name, value
            new_col_name = col_name+'-'+str(dat[col_name][0])
            dat[col_name] = 1
            dat= dat.rename(columns = {col_name:new_col_name})

    # refresh names in case there have been changes
    col_names = list(dat)
    num_layers = dat[[e for e in col_names if re.match('.+_num_layers', e)][0]]

    for col_name in col_names:
        # overwrite hps in non-used layers with na
        if re.match('.+_\d$', col_name):
            layer_num = col_name.split('_')[-1]
            layer_num = int(layer_num)
            # if outside of max layer
            if not (layer_num <= (int(num_layers) + 1)):
                dat.loc[0, col_name] = np.nan

    rank = re.findall('\d.json', json_file)[-1].strip('.json')
    rank = int(rank)

    dat = dat.T.reset_index().rename(columns = {'index':'hyperparameter', 0:rank})

    return(dat)



def merge_df_list(df_list):
    for i in range(len(df_list)):
        if i == 0:
            df = df_list[i]
        else:
            df = df.merge(df_list[i], how = 'outer')
    return(df)



# %% code_folding=[53]
# Retrieve hyperparameters for vis ------------------------------------------------

hps_files = [_hps_to_df(e) for e in hps_files ]

hps = merge_df_list(df_list = hps_files)

hps.index = hps.hyperparameter


# Process for vis -----------------------------------------------------------------
## Make a long format version of the HP set and scale within each parameter set.
hps_scaled = hps
hps_scaled = hps_scaled.drop(columns = 'hyperparameter')


hps_scaled = hps_scaled.T

hps_text = hps_scaled
# hps_scaled


for col in list(hps_scaled):
    # col = 'learning_rate'
    cmax = np.max(hps_scaled.loc[:, col])
    cmin = np.min(hps_scaled.loc[:, col])

    # if there's a single value set to 100%.
    if cmax == cmin:
        hps_scaled.loc[:, col] = ((hps_scaled.loc[:, col]) / (cmin))
    else:
        hps_scaled.loc[:, col] = ((hps_scaled.loc[:, col] - cmin) / (cmax - cmin))
    
hps_scaled = hps_scaled.reset_index().rename(columns = {'index':'Trial'})
hps_scaled = hps_scaled.melt(id_vars = ['Trial'], value_vars= [e for e in list(hps_scaled) if e != 'Trial'])
hps_scaled= hps_scaled.rename(columns = {'value':'percent'})


# Process for vis -----------------------------------------------------------------
## Convert raw values to long format

hps_raw = hps.drop(columns = 'hyperparameter')
hps_raw = hps_raw.T
hps_raw = hps_raw.reset_index().rename(columns = {'index':'Trial'})
hps_raw = hps_raw.melt(id_vars = ['Trial'], value_vars= [e for e in list(hps_raw) if e != 'Trial'])

hps_plt = hps_raw.merge(hps_scaled, how = 'outer')

hps_plt= hps_plt.sort_values('hyperparameter')


# Vis -----------------------------------------------------------------------------
fig = go.Figure()

fig.add_trace(go.Heatmap(
    x = [str(e) for e in hps_plt['hyperparameter']],
    y = hps_plt['Trial'],
    z = hps_plt['percent']
))

for i in hps_plt.index:
    fig.add_annotation(
        x = hps_plt.loc[i, 'hyperparameter'],
        y = hps_plt.loc[i, 'Trial'],
        text = [int(round(e)) if e>1 else round(e, 4) for e in [hps_plt.loc[i, 'value']]][0],
        showarrow=False
    #     yshift=0
    )

fig.write_html("hps.html")

fig.show()


# %%
fig.write_image("hps.svg", width=6000, height=1000, scale=2)


# %%
def _history_to_df(json_file):
    with open(project_path+'eval/'+json_file) as f:
        dat = json.load(f)
    dat = pd.DataFrame(dat)

    rank = re.findall('rank_\d', json_file)[0].strip('rank_')
    fold = re.findall('fold_\d', json_file)[0].strip('fold_')
    
    dat['rank'] = int(rank)
    dat['fold'] = int(fold)
    dat = dat.reset_index().rename(columns = {'index':'epoch'})
    return(dat)


# %%
# Retrieve hyperparameters for vis ------------------------------------------------
history = [_history_to_df(e) for e in history_files ]
history = merge_df_list(df_list = history)

# %%
history

# %%

# %%
fig = make_subplots(rows=4, cols=2, shared_yaxes = True)

color_list = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']


for rank in list(set(history['rank'])):
    for fold in list(set(history.loc[(history['rank'] == rank), 'fold'])):
        mask = ((history['rank'] == rank) & (history['fold'] == fold))
        
        fig.add_trace(
            go.Scatter(x = history.loc[mask, 'epoch'], 
                       y = history.loc[mask, 'loss'],
                       line = dict(color=color_list[fold]),
                       name='fold '+str(fold)),
            row=rank+1, col=1
        )

        fig.add_trace(
            go.Scatter(x = history.loc[mask, 'epoch'], 
                       y = history.loc[mask, 'val_loss'],
                       opacity=0.35,
                       line = dict(color=color_list[fold]),
                       name='fold '+str(fold)),
            row=rank+1, col=2
        )

fig.update_layout(height=600, width=800, title_text="Training and Validation Losses")
fig.show()


# %%
temp = history.groupby(['rank', 'epoch']).agg(
    mean = ('val_loss', 'mean'),
    std =  ('val_loss', 'std')).reset_index()

temp['mean_sd'] = temp['mean']+temp['std']

# %%
temp['roll'] = temp.mean_sd.rolling(5).mean()

# %%
px.line(temp, x = 'epoch', 
#         y = 'mean_sd', 
        y = 'roll', 
        color = 'rank',
        facet_col="rank")

# %%
nbins = 10

temp['bin'] = pd.cut(temp['epoch'], nbins)
temp = temp.loc[:, ['rank', 'bin', 'mean_sd']
        ].groupby(['rank', 'bin']
        ).agg(mean_mean_end = ('mean_sd', 'mean')
        ).reset_index(
        ).pivot(index = 'rank', columns = 'bin', values = 'mean_mean_end')
temp

# %%
temp2 = temp

# %%

# %%
for i in list(temp2):
    print(i)
    temp2.loc[:, i] = (temp2.loc[:, i] == min(temp2.loc[:, i]))
    
temp2

# %%
# Which entry has the most "votes"
temp2['votes'] = [sum(temp2.loc[i, ]) for i in range(temp2.shape[0])]

# %%
temp2

# %%
mask = (temp2['votes'] == max(temp2['votes']))
best_rank = list(temp.loc[mask, ].index)

# Assert there's no tie
assert len(best_rank) == 1
best_rank = best_rank[0]
best_rank


# %% [markdown]
# # After we've selected an hps set:

# %%
selected_rank = best_rank

history = history.loc[history['rank'] == selected_rank, ]
history = history.sort_values(['epoch'])

# %%
history['fold'] = history['fold'].astype(str)



# %%
fig = px.scatter(history, x = 'epoch', y = 'val_loss', 
           color = 'fold'#, trendline="lowess"
                ).update_layout(template='plotly_white')#, facet_row = 'fold')
fig.show()

# %%
px.line(history, x = 'epoch', y = 'val_loss', 
           color = 'fold'#, facet_row = 'fold'
       ).update_layout(template='plotly_white')


# %%

# %%

# %%
# fig = px.scatter(history, x = 'epoch', y = 'val_loss', 
#            color = 'fold', trendline="lowess")#, facet_col = 'fold')

# fig.data = [t for t in fig.data if t.mode == "lines"]
# fig.show()

# %%



# %%
# n_checkpoints = 10
# bin_size = 20

# mask = ((history['rank'] == 2) & (history['fold'] == "0"))

# temp = history.loc[mask, ]
# temp = temp.copy()

# temp['around_100x'] = ''

# # n_checkpoints = 10
# # bin_size = 10

# bin_end =  max(temp['epoch']) #999
# for bin_end in list(np.linspace(0+bin_size, bin_end, n_checkpoints)):
#     bin_start = bin_end - bin_size
#     mask2 = ((bin_start < temp['epoch']) & (temp['epoch'] <= bin_end))
#     temp.loc[mask2, 'around_100x'] = str(bin_end)
    
# temp = temp.loc[temp['around_100x'] != '', ]
# temp = temp.groupby('around_100x').agg(checkpoint_val_loss = ('val_loss', np.mean),
#                                       epoch = ('epoch', max)).reset_index()

# %%
# temp

# %%
# fig = px.scatter(temp, x = 'epoch', y = 'checkpoint_val_loss')#, 
# #            color = 'fold', trendline="lowess")
# fig.show()

# %%
# history.loc[mask, ].merge(temp, how = 'outer') 

# %%

# %%
n_roll = 20

# # FIXME --v  this is a hacky way to get just the epochs for the saved model
# templist = [[j-(9-k) for k in range(10)] for j in [(100*(i+1)-1) for i in range(10)]]

# agglist = []
# for e in templist:
#     agglist.extend(e)
# # Fixme --^


color_list = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']

fig = go.Figure()



for fold in list(history.sort_values('fold').loc[:, 'fold'].drop_duplicates()): 
           #[str(j) for j in range(10)]:
    i = int(fold)
    mask = ((history['fold'] == fold) #& 
#            (history['epoch'].isin(agglist)) # <-- FIXME
           
           ) #"0"))

    df = history.loc[mask, ].copy()
    df.loc[:, 'roll_max'] = df['val_loss'].rolling(n_roll).max()
    df.loc[:, 'roll_min'] = df['val_loss'].rolling(n_roll).min()
    df.loc[:, 'roll_mean'] = df['val_loss'].rolling(n_roll).mean()

    df.loc[:, 'roll_q25'] = df['val_loss'].rolling(n_roll).quantile(.25)
    df.loc[:, 'roll_q75'] = df['val_loss'].rolling(n_roll).quantile(.75)
    
    df.loc[:, 'cmin_q25'] = df['roll_q25'].cummin()   
    
#     df.loc[:, 'cmin_q25'] = df.loc[:, 'cmin_q25'] - min(df['roll_q25']) # FIXME
    df.loc[:, 'separation'] = df['roll_mean']#-df['roll_q25']

    df
    if i == 0:
        accumulator = df
    else:
        accumulator = accumulator.merge(df, how = 'outer')
    
#     df = df.loc[df['epoch'].isin([999-(k*100) for k in range(10)]),]# <---------------------------------- FIXME 

    # fig.add_trace(go.Scatter(x = df['epoch'], y = df['val_loss'], line = dict(color=color_list[1]) ))
    # fig.add_trace(go.Scatter(x = df['epoch'], y = df['roll_max'], line = dict(color=color_list[2]) ))
    
    fig.add_trace(go.Scatter(x = df['epoch'], y = df['roll_q25'], 
                             line = dict(color=color_list[i]), name=i , legendgroup=i))
    fig.add_trace(go.Scatter(x = df['epoch'], y = df['roll_q75'], 
                             line = dict(color=color_list[i]), name=i , legendgroup=i,
                             fill='tonexty' ))
    
    fig.add_trace(go.Scatter(x = df['epoch'], y = df['cmin_q25'], 
                             line = dict(color=color_list[i]), name=i , legendgroup=i))
    
    fig.add_trace(go.Scatter(x = df['epoch'], y = df['separation'], 
                             line = dict(color='#6a3d9a'), #color_list[i]), 
                             name=i , legendgroup=i))

fig.update_layout(template='plotly_white')
fig    

# %%

# %% [markdown]
#

# %% [markdown]
# These are the stopping epochs we might select by looking at the cumulative min of q25. 
# We're looking to minimize 
# * Q25's distance from the cumulative min
# * number of epochs (1% better val loss isn't necessarily work 1000 more epochs)
#
# Several of these look sort of like this: █▁▆▅▄▄▄▅▄▄▄
# ☑
#
# |          |    |Window: 20  |            |Close?   |
# |:---------|---:|:-----------|:-----------|:-:|
# |Settings  |Rep |Every obs.  |every 100   |   |
# |          |    |            |            |   |
# |DNN Full  |0   |492         |499         | ☑ |
# |          |1   |556         |999         | ☐ |
# |          |2   |79          |99          | ☑ |
# |          |3   |25          |99          | ☑ |
# |          |4   |615         |599         | ☑ |
# |          |5   |115         |99          | ☑ |
# |          |6   |25?, 364    |399         | ☑ |
# |          |7   |43?, 206    |199         | ☑ |
# |          |8   |44          |799         | ☐ |
# |          |9   |725-999~=388|999         | ☑ |
#
# Decent agreement between looking at every 100 and every obs for window=20.

# %% [markdown]
# |          |    |Window: 20  |            |   |
# |:---------|---:|:-----------|:-----------|:-:|
# |Settings  |Rep |Every obs.  |every 100   |   |
# |          |    |            |            |   |
# |DNN G     |0   |43          |299         | ☐
# |          |1   |188         |99          | ☐ |
# |          |2   |110         |199         | ☑ |
# |          |3   |124         |199         | ☑ |
# |          |4   |262         |199         | ☑ |
# |          |5   |65          |399         | ☐ |
# |          |6   |102         |599         | ☐ |
# |          |7   |58          |299         | ☐ |
# |          |8   |170         |199         | ☑ |
# |          |9   |108         |399         | ☐ |
#
# Not good agreement

# %% [markdown]
# |          |    |Window: 20  |            |   |
# |:---------|---:|:-----------|:-----------|:-:|
# |Settings  |Rep |Every obs.  |every 100   |   |
# |          |    |            |            |   |
# |DNN S     |0   |442         |899 ~=599   | ☐
# |          |1   |309         |499 ~=299   | ☑ |
# |          |2   |393         |699         | ☐ |
# |          |3   |568         |499         | ☑ |
# |          |4   |507         |599         | ☑ |
# |          |5   |93          |99          | ☑ |
# |          |6   |303         |299         | ☑ |
# |          |7   |70          |99          | ☑ |
# |          |8   |905         |899         | ☑ |
# |          |9   |514         |399         | ☐ |
#
# Decent agreement between the two methods

# %% [markdown]
# |          |    |Window: 20  |            |   |
# |:---------|---:|:-----------|:-----------|:-:|
# |Settings  |Rep |Every obs.  |every 100   |   |
# |          |    |            |            |   |
# |DNN W     |0   |300, 350    |499         | ☐ | 
# |Rank 2    |1   |52, 133     |100         | ☑ | MEH
# |          |2   |198         |199         | ☑ | MEH
# |          |3   |519         |399         | ☐ |
# |          |4   |200, 564    |199         | ☑ | MEH
# |          |5   |60          |199 ~=799   | ☐ | MEH
# |          |6   |43          |399 ~=299   | ☐ | MEH
# |          |7   |22, 235     |499 ~=299   | ☐ |
# |          |8   |57, 245     |99          | ☑ |
# |          |9   |19, 126, 236|399         | ☐ | MEH
#
# Not good agreement

# %%
# px.scatter_3d(accumulator, x = 'epoch', y = 'cmin_q25', z = 'separation', color = 'fold')


# %% [markdown]
#

# %%
import scipy

# Genetics ----
pStop = [
]


# Soil ----
# pStop = [
#     899,
#     499,
#     699,
#     499,
#     599,
#     99,
#     299,
#     99,
#     899,
#     399
# ]




# Weather ----
# pStop = [
#     499,
#     100,
#     199,
#     399,
#     199,
#     199,
#     399,
#     499,
#     99,
#     399
# ]
[np.median(pStop),np.mean(pStop), scipy.stats.mode(pStop)]

# %%
# what if we sum all the val losses?

# accumulator.loc[:, ['epoch', 'fold', 'roll_mean']].pivot(index = 'epoch', columns = 'fold', values = 'roll_mean')
out = accumulator.loc[:, ['epoch', 'fold', 'roll_mean']\
                     ].groupby('epoch'
                     ).agg(sd = ('roll_mean', np.std),
                           mean = ('roll_mean', np.mean),
                           total = ('roll_mean', sum)
                           )

out.loc[out['total']==0, 'total'] = np.nan

out = out.reset_index()
out['mpsd'] = out['mean']+out['sd']
out['mmsd'] = out['mean']-out['sd']


# fig = go.Figure()
fig = make_subplots(rows=1, cols=2, shared_yaxes = False)
    
fig.add_trace(go.Scatter(x = out['epoch'], y = out['mpsd'], 
                         line = dict(color='#56B4E9'), name="Sd" , legendgroup=0))
fig.add_trace(go.Scatter(x = out['epoch'], y = out['mmsd'], 
                         line = dict(color='#56B4E9'), name="Sd" , legendgroup=0,
                         fill='tonexty' ))
    
fig.add_trace(go.Scatter(x = out['epoch'], 
                         y = out['mean'], 
                         line = dict(color='#E69F00'), name=i , legendgroup=i))

fig.add_trace(go.Scatter(x = out['epoch'], y = out['total'], 
                         line = dict(color='#D55E00'), #color_list[i]), 
                         name=i , legendgroup=i),
                         row=1, col=2)



# Add in selected epoch numbers by looking for the minimum of different values
mins = out.apply(np.nanmin, 0)

def _get_col_min(col_name = 'mmsd'):
    res = out.loc[out[col_name]==mins[col_name], ['epoch', col_name]
                 ].reset_index(
                 ).drop(columns = 'index'
                 ).rename(columns = {col_name:'y'})
    res['col_name'] = col_name
    return(res)
mins_df = pd.concat([_get_col_min(col_name = c_name) for c_name in ['mean', 'total', 'mpsd', 'mmsd']])
mins_df = mins_df.reset_index().drop(columns = 'index')
mins_df
mins_df['order'] = 1

floor_df = mins_df.copy()
floor_df['y'] = floor_df['y']-0
floor_df['order'] = 2

Nones_df = mins_df.copy()
Nones_df['y'] = None
Nones_df['order'] = 3

annotations_df = pd.concat([mins_df, floor_df, Nones_df]).reset_index().sort_values(['index', 'order'])

mask = annotations_df['col_name'] != 'total'

# Finally plot them out
fig.add_trace(
    go.Scatter(x=annotations_df.loc[mask, 'epoch'], y=annotations_df.loc[mask, 'y'], 
               line = dict(color='black')))

fig.add_trace(
    go.Scatter(x=annotations_df.loc[~mask, 'epoch'], y=annotations_df.loc[~mask, 'y'],
              line = dict(color='black')), row=1, col=2)




fig.update_layout(template='plotly_white')

  

# %% code_folding=[]
mpsd_mask = ((annotations_df['order'] == 1) & (annotations_df['col_name'] == 'mpsd'))
total_mask = ((annotations_df['order'] == 1) & (annotations_df['col_name'] == 'total'))
print("Best Target Epoch is \n"+str(
         int(annotations_df.loc[mpsd_mask, 'epoch']-(n_roll/2)))+" by mean+sd\n"+str(
         int(annotations_df.loc[total_mask, 'epoch']-(n_roll/2)))+" by total")

# %% [markdown]
# G - 10, 12
#
# S - 161, 199
#
# W - 225, 629
#
# Cat -- 362, 364
#
# Full - 796, 711
#

# %%
((10 - 12),
(161 - 199),
(225 - 629),
(362 - 364),
(796 - 711))

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

# %%
