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
result_dir = '5_concat_v0_finalize_hp'

dir_contains = os.listdir('../data/atlas/models/'+result_dir+'/')
dir_results = [entry for entry in dir_contains if re.match('^hps_.*_res.json$', entry)]

trials_df = [wrap_trial_to_df(trial_file) for trial_file in dir_results]
trials_df = pd.concat(trials_df)

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

# import plotly.express as px
# fig = px.scatter_3d(trials_df, x="mae", y="val_mae", z="epoch", color='anonymous_animals')

# fig.update_layout(
# #     scene = dict(
# #         xaxis = dict(nticks=4, range=[0,2],),
# #         yaxis = dict(nticks=4, range=[0,2],),
# #         zaxis = dict(nticks=4, range=[0,30],),),
# #     width=700,
# #     margin=dict(r=20, l=10, b=10, t=10),
#     template='plotly_white')

# fig.show()




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
result_dir = '5_concat_v0_finalize_hp'
res = wrap_find_best_hps_in_dir(result_dir = result_dir)
print(res['best_trial_hps'])
print(res['best_trial_hps_df'])
res['fig']

# %%
res['fig1']

