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

# %% [markdown]
# ## Expanding to multiple experiments

# %%
os.listdir("../data/atlas/models/")


# %%

# %%

# %% code_folding=[34]
def _get_a_fold_reps(cv_metric_dict, fold_id):
    rep_keys = list(cv_metric_dict[fold_id].keys())
    df_list_one_fold = list(map(lambda x: pd.DataFrame(cv_metric_dict[fold_id][x]).assign(rep = x), rep_keys))
    df_one_fold = pd.concat(df_list_one_fold, axis=0)
    df_one_fold['fold'] = fold_id
    return(df_one_fold)


def wrap_trial_to_df(result_dir, trial_file):
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

# %%
def mk_3d_loss(result_dir = '2_Gpca_finalize_epoch'):
    dir_contains = os.listdir('../data/atlas/models/'+result_dir+'/')
    dir_results = [entry for entry in dir_contains if re.match('^hps_.*_res.json$', entry)]

    trials_df = [wrap_trial_to_df(result_dir, trial_file) for trial_file in dir_results]
    trials_df = pd.concat(trials_df)

    trials_df = anonymous_animals(
        in_df = trials_df,
        col_to_anonymize = 'fold',
        seed = 65464
    )


    # Loss over epochs

    temp = trials_df
    temp['plot_cat'] = temp['anonymous_animals']+'_'+temp['fold']+'_'+temp['rep']

    temp = temp.sort_values(['plot_cat', 'epoch'])

    fig = px.line_3d(temp, x="plot_cat", y="epoch", z="loss", color='anonymous_animals')
    fig.update_layout(template='plotly_white')
    # fig.show()
    fig_loss = fig


    # Validation loss over epochs

    fig = px.line_3d(temp, x="plot_cat", y="epoch", z="val_loss", color='anonymous_animals', 
                     line_group = 'anonymous_animals') 
    #                  line_group = 'plot_cat') # <- this makes the plot unweildy
    fig.update_layout(template='plotly_white')
    # fig.show()
    fig_valoss = fig

    return({"loss":fig_loss,
           "val":fig_valoss})

# %%

# %%

# %%
res = mk_3d_loss(result_dir = '2_Gpca_finalize_epoch')
res["loss"]

# %%
res["val"]

# %%
res = mk_3d_loss(result_dir = 'syr_2_Gpca_finalize_epoch')
res["loss"]

# %%
res["val"]

# %%
res = mk_3d_loss(result_dir = 'sy_2_Gpca_finalize_epoch')
res["loss"]

# %%
res["val"]

# %%






# %%

# %%
res = mk_3d_loss(result_dir = '2_S_finalize_epoch')
res["loss"]

# %%
res["val"]

# %%
res = mk_3d_loss(result_dir = 'syr_2_S_finalize_epoch')
res["loss"]

# %%
res["val"]

# %%
res = mk_3d_loss(result_dir = 'sy_2_S_finalize_epoch')
res["loss"]

# %%
res["val"]

# %%
asdf

# %%
res = mk_3d_loss(result_dir = '2_Wc1d_finalize_epoch')
res["loss"]

# %%
res["val"]

# %%
res = mk_3d_loss(result_dir = 'syr_2_Wc1d_finalize_epoch')
res["loss"]

# %%
res["val"]

# %%
res = mk_3d_loss(result_dir = 'sy_2_Wc1d_finalize_epoch')
res["loss"]

# %%
res["val"]

# %%
#TODO add in sy and syr versions of Wc1d when they are done.

# %%
asdf


# %%
def _get_trial(result_dir):
    dir_contains = os.listdir('../data/atlas/models/'+result_dir+'/')
    dir_results = [entry for entry in dir_contains if re.match('^hps_.*_res.json$', entry)]

    trials_df = [wrap_trial_to_df(result_dir, trial_file) for trial_file in dir_results]
    trials_df = pd.concat(trials_df)

    trials_df = anonymous_animals(
        in_df = trials_df,
        col_to_anonymize = 'fold',
        seed = 65464
    )
    # Loss over epochs
    temp = trials_df
    temp['plot_cat'] = temp['anonymous_animals']+'_'+temp['fold']+'_'+temp['rep']

    temp = temp.sort_values(['plot_cat', 'epoch'])
    
    temp["result_dir"] = result_dir
    return(temp)
trial_list = [_get_trial(result_dir = result_dir) for result_dir in ["2_Gpca_finalize_epoch", 
                                                                     "syr_2_Gpca_finalize_epoch", 
                                                                     "sy_2_Gpca_finalize_epoch"]]

# %%
trials = pd.concat(trial_list)
trials


# %%
# mask = (trials.result_dir == "2_Gpca_finalize_epoch") & (trials.fold == "k0") #& (trials.rep == "rep0")
# temp = trials.loc[mask, ]
# temp

# %%

# %%
gif_df = trials#.loc[trials.epoch.isin([5*i for i in range(100)])]
gif_df["epoch_shift"] = gif_df["epoch"]
mask = (gif_df.result_dir == "sy_2_Gpca_finalize_epoch")
gif_df.loc[mask, "epoch_shift"] = gif_df.loc[mask, "epoch"] +5
mask = (gif_df.result_dir == "syr_2_Gpca_finalize_epoch")
gif_df.loc[mask, "epoch_shift"] = gif_df.loc[mask, "epoch"] +10

# gif_df

plt_gif = px.scatter(
    gif_df.loc[gif_df.epoch.isin([10*i for i in range(50)])], 
    x = 'epoch_shift', y = 'val_loss', color = 'result_dir',
    opacity=0.3,
#            size="pop", color="continent", hover_name="country",
#            log_x=True, size_max=55,           
    animation_group="plot_cat", 
    animation_frame="epoch", 
    range_x=[-5,510], range_y=[0.5, 3])

plt_gif.write_html("./G_val_anim.html")
plt_gif

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%
fig = px.line(temp, x = 'epoch', y = 'val_loss', color = 'rep', 
        facet_row="fold", facet_col="result_dir")
fig
fig.write_html("./G_compare_fold_epochs.html")

# %%
trial_list = [_get_trial(result_dir = result_dir) for result_dir in ['2_S_finalize_epoch', 
                                                                     'syr_2_S_finalize_epoch',
                                                                     'sy_2_S_finalize_epoch']]
trials = pd.concat(trial_list)
temp = trials
fig = px.line(temp, x = 'epoch', y = 'val_loss', color = 'rep', 
        facet_row="fold", facet_col="result_dir")
fig
fig.write_html("./S_compare_fold_epochs.html")

# %%
gif_df = trials#.loc[trials.epoch.isin([5*i for i in range(100)])]
gif_df["epoch_shift"] = gif_df["epoch"]
mask = (gif_df.result_dir == "sy_2_S_finalize_epoch")
gif_df.loc[mask, "epoch_shift"] = gif_df.loc[mask, "epoch"] +5
mask = (gif_df.result_dir == "syr_2_S_finalize_epoch")
gif_df.loc[mask, "epoch_shift"] = gif_df.loc[mask, "epoch"] +10

# gif_df

plt_gif = px.scatter(
    gif_df.loc[gif_df.epoch.isin([10*i for i in range(50)])], 
    x = 'epoch_shift', y = 'val_loss', color = 'result_dir',
    opacity=0.3,   
    animation_group="plot_cat", 
    animation_frame="epoch", 
    range_x=[-5,510], range_y=[0.5, 3])

plt_gif.write_html("./S_val_anim.html")
plt_gif


# %%
# temp = trials.loc[mask, ].sort_values('epoch')

# import plotly.graph_objects as go

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=temp['epoch'], y=temp['loss']#, fill='tonexty',
# #                     mode='none' # override default markers+lines
#                     ))
# fig.add_trace(go.Scatter(x=temp['epoch'], y=temp['val_loss'], fill='tonexty',
# #                     mode= 'none'
#                         ))

# fig.show()

# %%

# %%

# %%

# %%
# in depth comparsison of one triple
def mk_3d_loss(result_dir = '2_Gpca_finalize_epoch'):
    dir_contains = os.listdir('../data/atlas/models/'+result_dir+'/')
    dir_results = [entry for entry in dir_contains if re.match('^hps_.*_res.json$', entry)]

    trials_df = [wrap_trial_to_df(result_dir, trial_file) for trial_file in dir_results]
    trials_df = pd.concat(trials_df)

    trials_df = anonymous_animals(
        in_df = trials_df,
        col_to_anonymize = 'fold',
        seed = 65464
    )


    # Loss over epochs

    temp = trials_df
    temp['plot_cat'] = temp['anonymous_animals']+'_'+temp['fold']+'_'+temp['rep']

    temp = temp.sort_values(['plot_cat', 'epoch'])

    fig = px.line_3d(temp, x="plot_cat", y="epoch", z="loss", color='anonymous_animals')
    fig.update_layout(template='plotly_white')
    # fig.show()
    fig_loss = fig


    # Validation loss over epochs

    fig = px.line_3d(temp, x="plot_cat", y="epoch", z="val_loss", color='anonymous_animals', 
                     line_group = 'anonymous_animals') 
    #                  line_group = 'plot_cat') # <- this makes the plot unweildy
    fig.update_layout(template='plotly_white')
    # fig.show()
    fig_valoss = fig

    return({"loss":fig_loss,
           "val":fig_valoss})

# %%

# %%



# %%

# "2_Gpca_finalize_epoch", 
# "2_S_finalize_epoch", 
# "2_Wc1d_finalize_epoch", 
# "syr_2_Gpca_finalize_epoch", 
# "syr_2_S_finalize_epoch", 
# "sy_2_Gpca_finalize_epoch", 
# "sy_2_S_finalize_epoch"

# result_dir = '6_concat_v0_finalize_epoch'
result_dir = '2_Gpca_finalize_epoch'




dir_contains = os.listdir('../data/atlas/models/'+result_dir+'/')
dir_results = [entry for entry in dir_contains if re.match('^hps_.*_res.json$', entry)]

trials_df = [wrap_trial_to_df(trial_file) for trial_file in dir_results]
trials_df = pd.concat(trials_df)

# %%

# %% code_folding=[]
trials_df = anonymous_animals(
    in_df = trials_df,
    col_to_anonymize = 'fold',
    seed = 65464
)

# %%
trials_df

# %% code_folding=[]
# get lowess fits for each fold/rep combination.
# oddly there's no rollmean in numpy
combo_df = trials_df.loc[:,['fold', 'rep']].drop_duplicates().reset_index().drop(columns = 'index')

def rollmean_list(in_list, #= temp['loss'], 
                  window = 5
                 ):
    # the last (or first) (window-1) observations don't have the right number of obs
    length = len(in_list)
    rolling_mean = [np.mean(in_list[i:(i+window)]) for i in range(length)]
    # Slice off part of list without the right number of obs
    rolling_mean = rolling_mean[0:(length-(window-1))] 
    # Fill removed portion with NAs   
    rolling_mean.extend([np.nan for i in range(window-1)])
    return(rolling_mean)

def _apply_rollmean_by_combo(i, window = 50 #window
                            ):
    mask = (trials_df.fold == combo_df.loc[i, 'fold']
       ) & (trials_df.rep == combo_df.loc[i, 'rep'])
    temp = trials_df.loc[mask, ]
    temp = temp.sort_values('epoch')

    temp['roll_loss'] = rollmean_list(temp['loss'], window = window)
    temp['roll_val_loss'] = rollmean_list(temp['val_loss'], window = window)
    temp['roll_mae'] = rollmean_list(temp['mae'], window = window)
    temp['roll_val_mae'] = rollmean_list(temp['val_mae'], window = window)

    return(temp)

# window = 50
# rolled_result_list = [_apply_rollmean_by_combo(i, window = window) for i in combo_df.index]

# rolled_trials_df = pd.concat(rolled_result_list)

# fig = px.scatter(rolled_trials_df, 
#            x='epoch', y='roll_val_loss', 
#            trendline="lowess", 
#            trendline_color_override="red",
#            facet_col= 'fold', 
#            title="Rolling Average Validation performance (Each fold independently scaled)"
#           )
# fig.update_yaxes(matches=None)

# %%
# fig = px.scatter(rolled_trials_df, 
#            x='epoch', y='roll_val_mae', 
#            trendline="lowess", 
#            trendline_color_override="red",
#            facet_col= 'fold', 
#            title="Rolling Average Validation performance (Each fold independently scaled)"
#           )
# fig.update_yaxes(matches=None)

# %%

# %%

# %%



# %%

# %%

# %%
# mask = (trials_df.fold == 'k0') & (trials_df.rep == 'rep0')
# temp = trials_df.loc[mask,].sort_values('epoch')

# import plotly.graph_objects as go

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=temp['epoch'], y=temp['loss'], fill='tozeroy',
#                     mode='none' # override default markers+lines
#                     ))
# fig.add_trace(go.Scatter(x=temp['epoch'], y=temp['val_loss'], fill='tonexty',
#                     mode= 'none'))

# fig.show()

# %%

from plotly.subplots import make_subplots

fig = make_subplots(1, 3)
for i in range(1, 4):
    mask = (trials_df.fold == 'k0') & (trials_df.rep == 'rep'+str(i-1))
    print('rep'+str(i))
    
    temp = trials_df.loc[mask,].sort_values('epoch')
    
    fig.add_trace(go.Scatter(
        x=temp['epoch'], 
        y=temp[ 'loss'],
        fill=None,
        mode='lines', line_color='indigo'), 1, i)
    fig.add_trace(go.Scatter(
        x=temp[ 'epoch'], 
        y=temp[ 'val_loss'],
        fill='tonexty', # fill area between trace0 and trace1
        mode='lines', line_color='red'),    1, i)



# fig.update_xaxes(matches='x')
fig.show()



# %%




# %%

# %%
# Loss over epochs

temp = trials_df
temp['plot_cat'] = temp['anonymous_animals']+'_'+temp['fold']+'_'+temp['rep']

temp = temp.sort_values(['plot_cat', 'epoch'])

fig = px.line_3d(temp, x="plot_cat", y="epoch", z="loss", color='anonymous_animals')
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

# %%

# %%

# %%

# %%

# %%
