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

# %%

# %%
# Gpca_finalize_hp


result_dir = '1_S_finalize_hp'
dir_contains = os.listdir('../data/atlas/models/'+result_dir+'/')
dir_results = [entry for entry in dir_contains if re.match('^hps_.*_res.json$', entry)]

# %%
i = 0
current_model = dir_results[i]
print(current_model)
with open('../data/atlas/models/'+result_dir+'/'+current_model) as f:
    data = json.load(f)#f.read()

# %%
cv_metric_dict = data

# %%
cv_metric_dict

# %%
# todo get the name of the model too.
fold_id = 'k0'
rep_id  = 'rep0'

temp = pd.DataFrame(cv_metric_dict[fold_id][rep_id])

# temp['fold'] = fold_id
temp['rep'] = rep_id

temp


def _get_a_fold_reps(cv_metric_dict, fold_id):
    rep_keys = list(cv_metric_dict[fold_id].keys())
    df_list_one_fold = list(map(lambda x: pd.DataFrame(cv_metric_dict[fold_id][x]).assign(rep = x), rep_keys))
    df_one_fold = pd.concat(df_list_one_fold, axis=0)
    df_one_fold['fold'] = fold_id
    return(df_one_fold)

    
fold_ids = list(cv_metric_dict.keys())
df_list = [_get_a_fold_reps(cv_metric_dict = cv_metric_dict, fold_id=fold_id) for fold_id in fold_ids]
df = pd.concat(df_list, axis = 0)
df = df.reset_index().rename(columns = {'index':'epoch'})
#TODO add in the experiment name.
df


# %% [markdown]
# ## Examination of a single entry

# %%
temp = df.loc[df.fold == 'k0', ]

fig = px.line(temp, x='epoch', y= 'loss', color = 'rep', line_group = 'rep')
fig.update_layout(template = 'plotly_white',
                      width = 1000,
                      height = 750,
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))


# fig.update_yaxes(range = [0, 1])
fig.show()

# %%
# convert to long format for plotting
temp = df
response = 'loss'
col_sep = "_"

value_vars = [entry for entry in list(temp) if re.match('.*'+response+'.*', entry)]


out = temp.melt(id_vars=['fold', 'rep', 'epoch'], 
                value_vars= ['loss', 'val_loss'] )

# here's how to undo the above melt
# out.pivot(index=['fold', 'rep', 'epoch'], columns="variable", values="value").reset_index()
out

# %% [markdown]
# ### Overview of fitting 
#
# Shows performance on each fold and each replicate.
# * performance per replicate seems very consistent
# * performance per fold is highly variable on validation set.

# %%
fig = px.line(out, x='epoch', y= 'value', color = 'variable', 
                 line_group = 'rep',  
                 #facet_row="rep",
                 facet_col="fold")

fig.update_layout(title = 'Loss for 10 replicates')
fig.show()



# %%
df.head()

# %%
import plotly.express as px
fig = px.line_3d(df#.loc[((df.rep == 'rep0'))]
#                  , x="loss", y="val_loss", z="epoch", color='fold')
                 , x="mae", y="val_mae", z="epoch", color='fold')

fig.update_layout(
#     scene = dict(
#         xaxis = dict(nticks=4, range=[0,2],),
#         yaxis = dict(nticks=4, range=[0,2],),
#         zaxis = dict(nticks=4, range=[0,30],),),
#     width=700,
#     margin=dict(r=20, l=10, b=10, t=10),
    template='plotly_white')

fig.show()




# %%

# %%
# what is epoch with the best average performance?

# temp = out.pivot(index=['fold', 'variable', 'epoch'], columns="rep", values="value").reset_index()
# temp
temp = out.groupby(['variable', 'fold', #'rep', 
             'epoch']).agg(value = ('value', np.nanmean)).reset_index()

fig = px.line(temp, x='epoch', y= 'value', color = 'variable', facet_col="fold")
fig.update_layout(title = 'Average loss')
fig.show()


# %%
# what's the best we could do? 

df_min_loss = df.groupby(['rep', 'fold']).agg(loss = ('loss', np.nanmin)).reset_index()
df_min_loss = df_min_loss.merge(df, how = 'inner')

df_min_loss

# %%
# lets suppose that for each we used the best 
fig = px.scatter(df_min_loss, 'loss', 'val_loss', color = 'fold'
#                  , marginal_y = 'rug'
                )
fig.update_layout(template='plotly_white')
fig.update_xaxes(range=[0, 2])
fig.update_yaxes(range=[0, 2])
fig.show()


# %%

# %%

# %% [markdown]
# ## Expanding to multiple experiments

# %%

# %%

# %%

# %% code_folding=[34]
def _get_a_fold_reps(cv_metric_dict, fold_id):
    rep_keys = list(cv_metric_dict[fold_id].keys())
    df_list_one_fold = list(map(lambda x: pd.DataFrame(cv_metric_dict[fold_id][x]).assign(rep = x), rep_keys))
    df_one_fold = pd.concat(df_list_one_fold, axis=0)
    df_one_fold['fold'] = fold_id
    return(df_one_fold)


def wrap_trial_to_df(trial_file):
    with open('../data/atlas/models/'+result_dir+'/'+trial_file) as f:
        data = json.load(f)#f.read()

    cv_metric_dict = data

    fold_ids = list(cv_metric_dict.keys())
    df_list = [_get_a_fold_reps(cv_metric_dict = cv_metric_dict, fold_id=fold_id) for fold_id in fold_ids]
    df = pd.concat(df_list, axis = 0)
    df = df.reset_index().rename(columns = {'index':'epoch'})
    df['trial_file'] = trial_file

    # reorder columns so the metadata is at left. Not hardcoding to make this a little safer.
    context_cols = ['trial_file', 'fold', 'rep', 'epoch']
    reordered_cols = [entry for entry in list(df) if entry in context_cols]
    reordered_cols.extend([entry for entry in list(df) if entry not in context_cols])

    df = df.loc[:, reordered_cols]
    return(df)


def anonymous_animals(
    in_df,# = trials_df,
    col_to_anonymize,# = 'trial_file',
    seed# = 65464
):
    anonymous_animals = [
        "Alligator", "Anteater", "Armadillo", "Auroch", "Axolotl", 
        "Badger", "Bat", "Bear", "Beaver", "Buffalo", "Camel", 
        "Capybara", "Chameleon", "Cheetah", "Chinchilla", 
        "Chipmunk", "Chupacabra", "Cormorant", "Coyote", "Crow", 
        "Dingo", "Dinosaur", "Dog", "Dolphin", "Duck", "Elephant", 
        "Ferret", "Fox", "Frog", "Giraffe", "Gopher", "Grizzly", 
        "Hedgehog", "Hippo", "Hyena", "Ibex", "Ifrit", "Iguana", 
        "Jackal", "Kangaroo", "Koala", "Kraken", "Lemur", "Leopard", 
        "Liger", "Lion", "Llama", "Loris", "Manatee", "Mink", 
        "Monkey", "Moose", "Narwhal", "Nyan Cat", "Orangutan", 
        "Otter", "Panda", "Penguin", "Platypus", "Pumpkin", 
        "Python", "Quagga", "Rabbit", "Raccoon", "Rhino", "Sheep", 
        "Shrew", "Skunk", "Squirrel", "Tiger", "Turtle", "Walrus", 
        "Wolf", "Wolverine", "Wombat"]
    rng = np.random.RandomState(seed)
    
    values_to_anonymize = list(in_df[col_to_anonymize].drop_duplicates())
    new_names = rng.choice(anonymous_animals, len(values_to_anonymize))
    name_lookup = pd.DataFrame(zip(values_to_anonymize, list(new_names)), columns = [col_to_anonymize, 'anonymous_animals'])

    out_df = name_lookup.merge(in_df, how = 'outer')
    return(out_df)


# %%
result_dir = '1_Gpca_finalize_hp'
# result_dir = 'S_finalize_hp'


dir_contains = os.listdir('../data/atlas/models/'+result_dir+'/')
dir_results = [entry for entry in dir_contains if re.match('^hps_.*_res.json$', entry)]

trials_df = [wrap_trial_to_df(trial_file) for trial_file in dir_results]
trials_df = pd.concat(trials_df)

# %%

# %% code_folding=[]
trials_df = anonymous_animals(
    in_df = trials_df,
    col_to_anonymize = 'trial_file',
    seed = 65464
)

# %%
trials_df

# %%
# MAE, Val. MAE consilience over epochs

import plotly.express as px
fig = px.scatter_3d(trials_df, x="mae", y="val_mae", z="epoch", color='anonymous_animals')

fig.update_layout(
#     scene = dict(
#         xaxis = dict(nticks=4, range=[0,2],),
#         yaxis = dict(nticks=4, range=[0,2],),
#         zaxis = dict(nticks=4, range=[0,30],),),
#     width=700,
#     margin=dict(r=20, l=10, b=10, t=10),
    template='plotly_white')

fig.show()




# %%
# Loss over epochs

temp = trials_df
temp['plot_cat'] = temp['anonymous_animals']+'_'+temp['fold']+'_'+temp['rep']

temp = temp.sort_values(['plot_cat', 'epoch'])

fig = px.line_3d(temp, x="plot_cat", y="epoch", z="loss", color='anonymous_animals'                   )
fig.update_layout(template='plotly_white')
fig.show()

# %%
# Validation loss over epochs

fig = px.line_3d(temp, x="plot_cat", y="epoch", z="val_loss", color='anonymous_animals', 
                 line_group = 'anonymous_animals') 
#                  line_group = 'plot_cat') # <- this makes the plot unweildy
fig.update_layout(template='plotly_white')
fig.show()

# %%

# %%

# %% code_folding=[4]
# let's say we go with the last entry on each.
find_these = temp.groupby(['trial_file', 'rep', 'fold']).agg(epoch = ('epoch', np.nanmax)).reset_index()

find_these = find_these.merge(temp, how = 'inner')
find_these

# %% [markdown]
# ### Promising figures

# %%
# dispersion
# mean val_loss
find_these_descriptives = find_these.groupby(['trial_file', 'anonymous_animals']).agg(
    val_loss_mean = ('val_loss', np.nanmean),
    val_loss_sd = ('val_loss', np.nanstd),
    val_mae_mean = ('val_mae', np.nanmean),
    val_mae_sd = ('val_mae', np.nanstd)
    ).reset_index()

find_these_descriptives = find_these_descriptives.sort_values('anonymous_animals')

# %%
# Look for low loss, low loss sd

fig = px.scatter(find_these_descriptives, x="val_loss_mean", y="val_loss_sd", color='anonymous_animals')
fig.update_layout(template='plotly_white', title = 'Validation Loss: Standard Deviation vs Mean')
fig.show()

# %%
fig = px.scatter(find_these_descriptives, x="val_mae_mean", y="val_mae_sd", color='anonymous_animals')
fig.update_layout(template='plotly_white', title = 'Validation MAE: Standard Deviation vs Mean')
fig.show()

# %%
find_these_descriptives['mean_plus_sd'] = find_these_descriptives['val_mae_mean'] + find_these_descriptives['val_mae_sd'] 

fig = px.scatter(find_these_descriptives, x="val_mae_mean", y="mean_plus_sd", color='anonymous_animals')
fig.update_layout(template='plotly_white', title = 'Validation MAE: Possible Selection Metric')
fig.show()

# %% [markdown]
# ### Tested but not useful figures

# %%
# distribution of val_loss in terminal models
fig = px.violin(find_these, x = 'anonymous_animals', y = 'val_loss', color = 'fold' )
fig.update_layout(template='plotly_white')
fig.show()

# %%
# mean loss 
plt_me = find_these.groupby(['anonymous_animals'#, 'rep'
                    , 'fold']
                  ).agg(val_loss = ('val_loss', np.mean)
                       ).reset_index(
).pivot(index = 'anonymous_animals', columns = 'fold', values = 'val_loss')

px.imshow(plt_me, color_continuous_scale='RdBu_r', origin='lower')

# %%

# %% [markdown]
# ### Visualizing the "best" Model
#
# After selecting one from a set, we want to retrieve the hps for that entry.

# %%
mask = find_these_descriptives.mean_plus_sd == np.min(find_these_descriptives.mean_plus_sd)
best_trial = find_these_descriptives.loc[mask, 'trial_file']
best_trial = list(best_trial)[0]
best_trial

# %%
best_trial_hps = best_trial.split('_')[:-1]
best_trial_hps = "_".join(best_trial_hps)+".json"

# %%
best_trial_hps

# %%
with open('../data/atlas/models/'+result_dir+'/'+best_trial_hps) as f:
    data = json.load(f)  
   
pd.DataFrame(zip(data['hyperparameters'],data['value']), columns = ['hyperparameter', 'Value'])

# %% [markdown]
# ## Apply to all Model Types

# %%
# result_dir = 'S_finalize_hp'


# dir_contains = os.listdir('../data/atlas/models/'+result_dir+'/')
# dir_results = [entry for entry in dir_contains if re.match('^hps_.*_res.json$', entry)]

# trials_df = [wrap_trial_to_df(trial_file) for trial_file in dir_results]
# trials_df = pd.concat(trials_df)



# trials_df = anonymous_animals(
#     in_df = trials_df,
#     col_to_anonymize = 'trial_file',
#     seed = 65464
# )

# temp = trials_df
# temp['plot_cat'] = temp['anonymous_animals']+'_'+temp['fold']+'_'+temp['rep']

# temp = temp.sort_values(['plot_cat', 'epoch'])


# # let's say we go with the last entry on each.
# find_these = temp.groupby(['trial_file', 'rep', 'fold']).agg(epoch = ('epoch', np.nanmax)).reset_index()

# find_these = find_these.merge(temp, how = 'inner')



# # dispersion
# # mean val_loss
# find_these_descriptives = find_these.groupby(['trial_file', 'anonymous_animals']).agg(
#     val_loss_mean = ('val_loss', np.nanmean),
#     val_loss_sd = ('val_loss', np.nanstd),
#     val_mae_mean = ('val_mae', np.nanmean),
#     val_mae_sd = ('val_mae', np.nanstd)
#     ).reset_index()

# find_these_descriptives = find_these_descriptives.sort_values('anonymous_animals')


# find_these_descriptives['mean_plus_sd'] = find_these_descriptives['val_mae_mean'] + find_these_descriptives['val_mae_sd'] 



# ## Plotting ==========================================================

# fig = px.scatter(find_these_descriptives, x="val_mae_mean", y="mean_plus_sd", color='anonymous_animals')
# fig.update_layout(template='plotly_white', title = 'Validation MAE: Possible Selection Metric')
# # fig.show()

# # graph object based version to show location 

# find_these_descriptives = find_these_descriptives.reset_index().drop(columns = 'index')

# fig1 = go.Figure()

# temp = find_these_descriptives.sort_values('val_mae_mean')
# fig1.add_trace(go.Line(
#         x = temp.loc[:, 'val_mae_mean'], 
#         y = temp.loc[:, 'mean_plus_sd'],
#         marker_color ='black',
#         marker_size  = 3))

# fig1.add_trace(go.Scatter(
#     x = temp.loc[temp.mean_plus_sd == np.min(temp.mean_plus_sd), 'val_mae_mean'], 
#     y = temp.loc[temp.mean_plus_sd == np.min(temp.mean_plus_sd), 'mean_plus_sd'],
#     marker_color = 'red'
# ))


# fig1.update_layout(template='plotly_white')
# # fig1.show()

# ## Find best index ===================================================
# mask = find_these_descriptives.mean_plus_sd == np.min(find_these_descriptives.mean_plus_sd)
# best_trial = find_these_descriptives.loc[mask, 'trial_file']
# best_trial = list(best_trial)[0]


# best_trial_hps = best_trial.split('_')[:-1]
# best_trial_hps = "_".join(best_trial_hps)+".json"


# ## Get the hyperparameters =========================================== 
# with open('../data/atlas/models/'+result_dir+'/'+best_trial_hps) as f:
#     data = json.load(f)  

# best_trial_hps_df = pd.DataFrame(zip(data['hyperparameters'],data['value']), columns = ['hyperparameter', 'Value'])

# nlayers = int(best_trial_hps_df.loc[best_trial_hps_df.hyperparameter == 'num_layers', 'Value'])
# layer_params = [entry for entry in list(best_trial_hps_df.hyperparameter) if re.match('.*_\d', entry)]
# discard_params = [entry for entry in layer_params if int(entry[-1]) >= nlayers]
# mask = [True if entry not in discard_params else False for entry in list(best_trial_hps_df.hyperparameter)]
# best_trial_hps_df = best_trial_hps_df.loc[mask, ]

# %%

# %%

def wrap_find_best_hps_in_dir(result_dir = 'Gpca_finalize_hp'):

    dir_contains = os.listdir('../data/atlas/models/'+result_dir+'/')
    dir_results = [entry for entry in dir_contains if re.match('^hps_.*_res.json$', entry)]

    trials_df = [wrap_trial_to_df(trial_file) for trial_file in dir_results]
    trials_df = pd.concat(trials_df)



    trials_df = anonymous_animals(
        in_df = trials_df,
        col_to_anonymize = 'trial_file',
        seed = 65464
    )

    temp = trials_df
    temp['plot_cat'] = temp['anonymous_animals']+'_'+temp['fold']+'_'+temp['rep']

    temp = temp.sort_values(['plot_cat', 'epoch'])


    # let's say we go with the last entry on each.
    find_these = temp.groupby(['trial_file', 'rep', 'fold']).agg(epoch = ('epoch', np.nanmax)).reset_index()

    find_these = find_these.merge(temp, how = 'inner')



    # dispersion
    # mean val_loss
    find_these_descriptives = find_these.groupby(['trial_file', 'anonymous_animals']).agg(
        val_loss_mean = ('val_loss', np.nanmean),
        val_loss_sd = ('val_loss', np.nanstd),
        val_mae_mean = ('val_mae', np.nanmean),
        val_mae_sd = ('val_mae', np.nanstd)
        ).reset_index()

    find_these_descriptives = find_these_descriptives.sort_values('anonymous_animals')


    find_these_descriptives['mean_plus_sd'] = find_these_descriptives['val_mae_mean'] + find_these_descriptives['val_mae_sd'] 



    ## Plotting ==========================================================

    fig = px.scatter(find_these_descriptives, x="val_mae_mean", y="mean_plus_sd", color='anonymous_animals')
    fig.update_layout(template='plotly_white', title = 'Validation MAE: Possible Selection Metric')
    # fig.show()

    # graph object based version to show location 

    find_these_descriptives = find_these_descriptives.reset_index().drop(columns = 'index')

    fig1 = go.Figure()

    temp = find_these_descriptives.sort_values('val_mae_mean')
    fig1.add_trace(go.Line(
            x = temp.loc[:, 'val_mae_mean'], 
            y = temp.loc[:, 'mean_plus_sd'],
            marker_color ='black',
            marker_size  = 3))

    fig1.add_trace(go.Scatter(
        x = temp.loc[temp.mean_plus_sd == np.min(temp.mean_plus_sd), 'val_mae_mean'], 
        y = temp.loc[temp.mean_plus_sd == np.min(temp.mean_plus_sd), 'mean_plus_sd'],
        marker_color = 'red'
    ))


    fig1.update_layout(template='plotly_white')
    # fig1.show()

    ## Find best index ===================================================
    mask = find_these_descriptives.mean_plus_sd == np.min(find_these_descriptives.mean_plus_sd)
    best_trial = find_these_descriptives.loc[mask, 'trial_file']
    best_trial = list(best_trial)[0]


    best_trial_hps = best_trial.split('_')[:-1]
    best_trial_hps = "_".join(best_trial_hps)+".json"


    ## Get the hyperparameters =========================================== 
    with open('../data/atlas/models/'+result_dir+'/'+best_trial_hps) as f:
        data = json.load(f)  

    best_trial_hps_df = pd.DataFrame(zip(data['hyperparameters'],data['value']), columns = ['hyperparameter', 'Value'])

#     nlayers = int(best_trial_hps_df.loc[best_trial_hps_df.hyperparameter == 'num_layers', 'Value'])
#     layer_params = [entry for entry in list(best_trial_hps_df.hyperparameter) if re.match('.*_\d', entry)]
#     discard_params = [entry for entry in layer_params if int(entry[-1]) >= nlayers]
#     mask = [True if entry not in discard_params else False for entry in list(best_trial_hps_df.hyperparameter)]
#     best_trial_hps_df = best_trial_hps_df.loc[mask, ]


    return({'best_trial_hps':best_trial_hps,
            'best_trial_hps_df':best_trial_hps_df,
            'fig':fig,
            'fig1':fig1
           })

# %%
result_dir = '1_Gpca_finalize_hp' # Very odd behavior here. function is using the global, not local scope for this variable. 
# Defining it here is a quick and dirty work around.
res = wrap_find_best_hps_in_dir(result_dir = result_dir)
print(res['best_trial_hps'])
print(res['best_trial_hps_df'])
res['fig1']

# %%
result_dir = 'S_finalize_hp'
res = wrap_find_best_hps_in_dir(result_dir = result_dir)
print(res['best_trial_hps'])
print(res['best_trial_hps_df'])
res['fig1']

# %%
result_dir = 'Wmlp_finalize_hp'
res = wrap_find_best_hps_in_dir(result_dir = result_dir)
print(res['best_trial_hps'])
print(res['best_trial_hps_df'])
res['fig']

# %%
result_dir = 'Wc1d_finalize_hp'
res = wrap_find_best_hps_in_dir(result_dir = result_dir)
print(res['best_trial_hps'])
print(res['best_trial_hps_df'])
res['fig']

# %%
result_dir = 'Wc2d_finalize_hp'
res = wrap_find_best_hps_in_dir(result_dir = result_dir)
print(res['best_trial_hps'])
print(res['best_trial_hps_df'])
res['fig']

# %%

# %%
# Analysis conditioned on dropout:

# def wrap_find_best_hps_in_dir(result_dir = 'Gpca_finalize_hp'):
result_dir = '1_Gpca_finalize_hp'
result_dir = '1_S_finalize_hp'

dir_contains = os.listdir('../data/atlas/models/'+result_dir+'/')
dir_results = [entry for entry in dir_contains if re.match('^hps_.*_res.json$', entry)]

trials_df = [wrap_trial_to_df(trial_file) for trial_file in dir_results]
trials_df = pd.concat(trials_df)

trials_df = anonymous_animals(
    in_df = trials_df,
    col_to_anonymize = 'trial_file',
    seed = 65464
)

temp = trials_df
temp['plot_cat'] = temp['anonymous_animals']+'_'+temp['fold']+'_'+temp['rep']

temp = temp.sort_values(['plot_cat', 'epoch'])


# let's say we go with the last entry on each.
find_these = temp.groupby(['trial_file', 'rep', 'fold']).agg(epoch = ('epoch', np.nanmax)).reset_index()

find_these = find_these.merge(temp, how = 'inner')



# dispersion
# mean val_loss
find_these_descriptives = find_these.groupby(['trial_file', 'anonymous_animals']).agg(
    val_loss_mean = ('val_loss', np.nanmean),
    val_loss_sd = ('val_loss', np.nanstd),
    val_mae_mean = ('val_mae', np.nanmean),
    val_mae_sd = ('val_mae', np.nanstd)
    ).reset_index()

find_these_descriptives = find_these_descriptives.sort_values('anonymous_animals')


find_these_descriptives['mean_plus_sd'] = find_these_descriptives['val_mae_mean'] + find_these_descriptives['val_mae_sd'] 




## 

dir_path = '../data/atlas/models/'+result_dir+'/'
# get all json
dir_json = [entry for entry in os.listdir(dir_path) if re.match("hps_\w+.json", entry)]
# select only the non result json
dir_json_res = [entry for entry in dir_json if re.match(".*_res.json", entry)]
dir_json_hps = [entry for entry in dir_json if entry not in dir_json_res]


trial_list = []
for entry in dir_json_hps:
    with open('../data/atlas/models/'+result_dir+'/'+entry) as f:
        data = json.load(f)  
    trial_hps_df = pd.DataFrame(zip(data['hyperparameters'],data['value']), columns = ['hyperparameter', 'Value'])
    trial_hps_df['trial'] = entry
    trial_list.append(trial_hps_df)
    
    
trial_hps_df = pd.concat(trial_list)
trial_hps_df['is_dropout'] = [True if re.match(".*dropout.*", e) else False for e in list(trial_hps_df['hyperparameter'])]
temp = trial_hps_df.loc[trial_hps_df.is_dropout].groupby(['trial']).agg(max_drop = ('Value', np.max))
temp

# %%
temp = temp.reset_index().rename(columns = {'trial':'trial_file'})
temp['trial_file'] = temp['trial_file'].str.strip(".json")+'_res.json'
find_these_descriptives = temp.merge(find_these_descriptives, how = "outer")

# %%
find_these_descriptives

# %%
## Plotting ==========================================================
fig = px.scatter(find_these_descriptives, x="val_mae_mean", y="mean_plus_sd", 
#                  color='anonymous_animals'
                 color='max_drop'
                )
fig.update_layout(template='plotly_white', title = 'Validation MAE: Possible Selection Metric')
fig.show()




# graph object based version to show location 

# find_these_descriptives = find_these_descriptives.reset_index().drop(columns = 'index')

# fig1 = go.Figure()

# temp = find_these_descriptives.sort_values('val_mae_mean')
# fig1.add_trace(go.Line(
#         x = temp.loc[:, 'val_mae_mean'], 
#         y = temp.loc[:, 'mean_plus_sd'],
#         marker_color ='black',
#         marker_size  = 3))

# fig1.add_trace(go.Scatter(
#     x = temp.loc[temp.mean_plus_sd == np.min(temp.mean_plus_sd), 'val_mae_mean'], 
#     y = temp.loc[temp.mean_plus_sd == np.min(temp.mean_plus_sd), 'mean_plus_sd'],
#     marker_color = 'red'
# ))


# fig1.update_layout(template='plotly_white')
# # fig1.show()

# ## Find best index ===================================================
# mask = find_these_descriptives.mean_plus_sd == np.min(find_these_descriptives.mean_plus_sd)
# best_trial = find_these_descriptives.loc[mask, 'trial_file']
# best_trial = list(best_trial)[0]


# best_trial_hps = best_trial.split('_')[:-1]
# best_trial_hps = "_".join(best_trial_hps)+".json"


# ## Get the hyperparameters =========================================== 
# with open('../data/atlas/models/'+result_dir+'/'+best_trial_hps) as f:
#     data = json.load(f)  

# best_trial_hps_df = pd.DataFrame(zip(data['hyperparameters'],data['value']), columns = ['hyperparameter', 'Value'])

    


#     return({'best_trial_hps':best_trial_hps,
#             'best_trial_hps_df':best_trial_hps_df,
#             'fig':fig,
#             'fig1':fig1
#            })

# %%

# %%

# %%

# %%

# %%
import matplotlib.pyplot as plt
# make plot
fig, ax = plt.subplots()
# show image
shw = ax.imshow(temp)
# make bar
bar = plt.colorbar(shw)
  
# show plot with labels
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.show()

# %%

# %%

# %%

# %%

# %%
