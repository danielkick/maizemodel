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
import os, re, time
import numpy as np
import pandas as pd
import json
# for plotting =======================
import math # for ceil. Used in color assignment for plotting
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# %% [markdown]
# # Demo of functionality

# %% [markdown]
# # Functions

# %% [markdown]
# ##  Processing logs in `/kt/`

# %%
# used by `kt_logs_to_dfs()` ------------------------------------------------------------------------

# This function is designed take a trial in /kt/ and return the trial id, 
# hyperparamters used, specified hyperparamter ranges, and results.
# it will need to be called on all n=40 trials in /kt/
def parse_kt_logs(trial_path):
    
    with open(trial_path) as f:    
        data = json.load(f)

    # Data processing --------------------------------------------------------------
    assert data['status'] == 'COMPLETED'

    ## Get the hyperparameter info =================================================
    trial_hps = pd.DataFrame(data['hyperparameters']['values'], index = [0])
    trial_hps = trial_hps.T.reset_index()
    trial_hps = trial_hps.rename(columns = {'index':'hyperparameters',
                                                  0:'value'})
    trial_hps['trial_id'] = data['trial_id']


    ## Get the response info =======================================================
    out_dict = {'metric':[],
                 'value':[],
                  'step':[]}

    for key_index in data['metrics']['metrics'].keys():
        # expected_dict_format = {
        #     'direction': 'min', 
        #     'observations': [
        #         {'value': [1.0340402126312256], 
        #          'step': 0}]
        # }

        expected_dict_format = data['metrics']['metrics'][key_index]

        # make sure that the observation matches the expected format
        assert len(expected_dict_format['observations']) == 1
        assert type(expected_dict_format['observations'][0]['value']) == list
        assert len(expected_dict_format['observations'][0]['value']) == 1
        assert type(expected_dict_format['observations'][0]['step']) == int


        metrics_drill_down = data['metrics']['metrics'][key_index]['observations'][0]

        out_dict['metric'].append(key_index)
        out_dict['value'].append(metrics_drill_down['value'][0])
        out_dict['step'].append(metrics_drill_down['step'])

        trial_response = pd.DataFrame(out_dict)
        trial_response['trial_id'] = data['trial_id']
        trial_response[['data', 'metric']] = trial_response['metric'].str.split('_', expand=True)


    trial_response = pd.pivot_table(trial_response, index = ['trial_id', 'step', 'data'], columns = 'metric', values = 'value')
    trial_response = trial_response.reset_index()


    ## find maximum values of hyps =================================================

    # this is set up assuming non char choices (e.g. no mlp vs c1d)
    hyperparameters = []
    hyperparameters_min = []
    hyperparameters_max = []

    for i in range(len(data['hyperparameters']['space'])):
        temp = data['hyperparameters']['space'][i]['config']
        new_name = temp['name'] # <-- default name (overwritten for catagorical variables)
    #         hyperparameters.append(temp['name'])
        if (('min_value' not in temp.keys()) & ('max_value' not in temp.keys())):
            values_are_numeric = [True if ((type(entry) == float) | (type(entry) == int)
                                          ) else False for entry in temp['values']]

            if False not in values_are_numeric:


                hyperparameters_min.append(min(temp['values']))
                hyperparameters_max.append(max(temp['values']))
            else:
                # Make a new variable that contains the levels for a catagorical. 
                # Then we'll be able to convert cat <--> num
                new_name = temp['name']+'_'+','.join(temp['values']) # <-- overwrite with non-default name
                # for character vectors
                hyperparameters_min.append(0)
                hyperparameters_max.append((len(temp['values'])-1))

        else:    
            if 'min_value' in temp.keys():
                hyperparameters_min.append(temp['min_value'])
            if 'max_value' in temp.keys():
                hyperparameters_max.append(temp['max_value'])

        # Add the new name after we've had time to create it if need be.
        hyperparameters.append(new_name)   
        
    hps_minmax = pd.DataFrame({
        'hyperparameters': hyperparameters,
    'hyperparameters_min': hyperparameters_min,
    'hyperparameters_max': hyperparameters_max}) 
   
    hps_minmax['trial_id'] = data['trial_id']

    
    """
    For all the character containing values, reference hp_minmax 
    (which has the factor levels and a factor containing name)
    and make these values and hp names consistent. 

    check trial hps to make sure there are no character entries.
    if there are look at trial ranges for the new hps name and the right interger to assign there. 
    find rows in trial_hps that can't be converted to a numeric
    """
    value_is_chr = [True if ((type(entry) != float) & (type(entry) != int)
                             ) else False for entry in trial_hps['value']]

    for hyperparameter in list(trial_hps.loc[value_is_chr, 'hyperparameters']):
        matches_in_minmax = [entry for entry in hps_minmax['hyperparameters'] if re.match(hyperparameter, entry)] 
        assert len(matches_in_minmax) == 1
        # extract the factor list
        chr_variable_levels = matches_in_minmax[0].split('_')[-1].split(',')
        chr_variable_levels

        # get the value to encode
        mask = trial_hps.hyperparameters == hyperparameter
        string_to_encode = trial_hps.loc[mask, 'value']
        string_to_encode = list(string_to_encode)[0]
        # look up conversion
        string_as_number = [i for i in range(len(chr_variable_levels)) if chr_variable_levels[i] == string_to_encode]
        string_as_number = string_as_number[0]
        # overwrite with index
        trial_hps.loc[mask, 'value'] = string_as_number
        # overwrite with the level containing name
        trial_hps.loc[mask, 'hyperparameters'] = matches_in_minmax[0]


    return({
        'trial_id':data['trial_id'],
        'trial_hps':trial_hps,
        'trial_hp_ranges':hps_minmax,
        'trial_response':trial_response})


# This function takes a list of dictionaries containting dfs and the key for a single df. 
# It defines a dataframe and then joins (outer) each df matching the key into an accumulator.
# The full df is returned.
def apply_df_accumulator_pattern(
    list_with_df_dicts,# =  result,
    df_key             # = 'trial_hps'
):
    df_accumulator = pd.DataFrame()
    for df_dict in list_with_df_dicts:
        if df_accumulator.shape[0] == 0:
            df_accumulator = df_dict[df_key]
        else:
            df_accumulator = df_accumulator.merge(df_dict[df_key], how = 'outer')
    return(df_accumulator)


# High level functions ------------------------------------------------------------------------------

# Wrap the proecessing of kt neatly. Produces a df with 3 dataframes containing 
# the tested hps, their ranges, and the results of the trials.
def kt_logs_to_dfs(kt_path = '../data/atlas/models/Gpca_k0/kt/'):
    # we need to walk through the kt log dirs. 
    # To do this, let's have code that will generate the paths that need be stepped through
    # and a funciton that parses the json at that path

    # find all trials in the kt directory
    contents = os.listdir(kt_path)
    trials = [entry for entry in contents if re.match('^trial_.*', entry)]


    # go throught all the trial directories, process each to get the output df. 
    trial_json_paths = map(lambda trial : kt_path+trial+'/trial.json', trials)
    result = map(parse_kt_logs, list(trial_json_paths))
    result = list(result)


    # Restructure the dictionary output for all the `/trial_.+`s
    # combine each type of retrieved data (except `trial_id`)
    kt_tested_hps = apply_df_accumulator_pattern(
        list_with_df_dicts = result,
        df_key = 'trial_hps')

    kt_hp_ranges = apply_df_accumulator_pattern(
        list_with_df_dicts = result,
        df_key = 'trial_hp_ranges')

    kt_trial_results = apply_df_accumulator_pattern(
        list_with_df_dicts = result,
        df_key = 'trial_response')
        
    return({
        'kt_tested_hps': kt_tested_hps,
        'kt_hp_ranges': kt_hp_ranges,
        'kt_trial_results': kt_trial_results
    })


# given a `kt_trial_results` from `kt_logs_to_dfs()` return a loss metric 
# named as a hyperparameter to make for easy plotting with the rest of them.
def kt_loss_as_hp(
    kt_trial_results,
    considered_loss_metric# = 'loss'
):
    # Make sure the requested loss is actually in the df.
    acceptable_metrics = [entry for entry in list(kt_trial_results) 
                          if entry not in ['trial_id', 'step', 'data']]
    if considered_loss_metric not in acceptable_metrics:
        raise Exception('Requested metric is not allowed. Use one of the following:\n', acceptable_metrics)

    ## Prepare the validation data
    loss_as_hp = kt_trial_results.loc[kt_trial_results.data == 'val', ]
    loss_as_hp = loss_as_hp.loc[:, ['trial_id', considered_loss_metric]]
    loss_as_hp = loss_as_hp.rename(columns = {considered_loss_metric:'value'})
    loss_as_hp['hyperparameters'] = considered_loss_metric
    loss_as_hp['hyperparameters_min'] = np.nanmin(loss_as_hp.loc[:, 'value'])
    loss_as_hp['hyperparameters_max'] = np.nanmax(loss_as_hp.loc[:, 'value'])

    loss_as_hp['hyperparameters'] = 'response_'+loss_as_hp['hyperparameters']
    return(loss_as_hp)


# %% [markdown]
# ## Processing logs in `/logs/`

# %% code_folding=[4]
# log_path = '../data/atlas/models/Gpca_k0/logs/training_log.csv'

def get_training_log(log_path):
    # inputs
    # kt_path = '../data/atlas/models/Gpca_k0/kt/'

    def get_train_order(kt_path):
        # find all trials
        contents = os.listdir(kt_path)
        trials = [entry for entry in contents if re.match('^trial_.*', entry)]

        def make_formatted_ctime(path):
            x = os.path.getctime(path)
            x = time.ctime(x)
            x = time.strptime(x)
            x = time.strftime("%Y-%m-%d %H:%M:%S", x)
            return (x)

        temp = pd.DataFrame({'trial':trials,
                     'ctime':[ 
                         make_formatted_ctime(kt_path+trial+'/')
                         for trial in trials] })
        temp = temp.sort_values('ctime')
        # reorder and create a trial number column for merging
        temp = temp.reset_index()
        temp = temp.drop(columns = 'index')
        temp = temp.reset_index()
        temp = temp.rename(columns = {'index':'n_trial'})
        trial_name_order = temp

        return(trial_name_order)

    train_log = pd.read_csv(log_path)

    # train_log 
    n_epochs = train_log.epoch.max()+1
    n_trials = int(train_log.shape[0]/n_epochs)
#     train_log.loc[:, 'n_trial'] = np.repeat([i for i in range(n_trials)], n_epochs)   
    # add in trial numbers even if trials have unequal epoch numbers (early stopping, failed run)

    trial_starts = train_log.reset_index()
    trial_starts = trial_starts.loc[trial_starts.epoch == 0, ['index', 'epoch']]
    trial_starts = trial_starts.rename(columns={'index': 'trial_start'})
    trial_starts = trial_starts.reset_index().drop(columns=['index', 'epoch'])
    trial_starts = trial_starts.reset_index().rename(columns={'index': 'trial'})
    trial_starts['trial_end'] = np.nan

    end_of_trials = list(trial_starts.loc[1:, 'trial_start'])
    # use the index right before the next trial
    end_of_trials = [entry-1 for entry in end_of_trials]

    trial_starts.loc[:, 'trial_end'] = end_of_trials+[train_log.shape[0]]

    train_log['n_trial'] = int
    for i in trial_starts.index:
        # i = 0
        train_log.loc[trial_starts.loc[i, 'trial_start'
                                       ]:trial_starts.loc[i, 'trial_end'], 'n_trial'
                      ] = trial_starts.loc[i, 'trial']    
    # train_log

    ## Get the order of trials run to match training data with the right names. 
    kt_path = log_path.split('/')
    kt_path = kt_path[:(len(kt_path)-2)]+['kt']
    kt_path = '/'.join(kt_path)+'/'
    trial_name_order = get_train_order(kt_path = kt_path) #'../data/atlas/models/Gpca_k0/kt/')
    
    trial_log = train_log.merge(trial_name_order, how = 'outer')

    return(trial_log)


# %%

# %% [markdown]
# ## Preparation for plotting

# %%
def plot_loss_curves(training_log
):
    nrow = 8
    ncol = 5

    fig = make_subplots(rows=nrow,
                        cols=ncol#, #column_widths=[0.7, 0.3], 
                        # row_heights = [0.3, 0.7]
                       )


    trial_list = list(training_log.loc[:, 'trial'].drop_duplicates())
    # trial = 'trial_fb10ad3173be290bb53c33f45332670a'

    subplot_col = 1
    subplot_row = 1
    for i in range(len(trial_list)):
        trial = trial_list[i]

        mask = training_log['trial'] == trial

        fig.add_trace(go.Scatter(
            x= training_log.loc[mask, 'epoch'],
            y= training_log.loc[mask, 'loss'], 
            mode='lines',
            line_color= '#7f7f7f',  # middle gray

            name= i
        ),
        row=subplot_row, col=subplot_col)

        fig.add_trace(go.Scatter(
            x= training_log.loc[mask, 'epoch'],
            y= training_log.loc[mask, 'val_loss'], 
            mode='lines',
            line_color= '#17becf',   # blue-teal
            name= i
        ),
        row=subplot_row, col=subplot_col)

        if subplot_col == ncol:
            subplot_col = 1
            subplot_row += 1
        else:
            subplot_col += 1


    fig.layout.update(template = 'plotly_white',
                      width = 1000,
                      height = 750,
                      showlegend=False
                     )

    return(fig)

# plot_loss_curves(training_log)

# %% [markdown]
# ### note there's some unexpected mismatching in the trial ids:

# %%
# training_log
def mk_trial_to_number_df(training_log):
    ## Prepare consistent numbering ================================================
    trial_to_number = training_log.loc[:, ['trial', 'n_trial']].drop_duplicates().rename(columns = {'trial':'trial_id'})

    # get rid of prefix
    trial_to_number['trial_id'] = trial_to_number['trial_id'].str.strip('trial_')
    # add in number to use instead of the full name.
    trial_to_number['n_trial'] = pd.DataFrame(trial_to_number.loc[:, 'n_trial']).astype(str)
    return(trial_to_number)


# %%
def mk_hp_info_df(
    kt_hp_ranges,
    kt_tested_hps,
    loss_as_hp,
    trial_to_number
):
    # merge all the output from kt_logs_to_dfs
    hp_info = kt_hp_ranges.merge(kt_tested_hps, how = 'outer').merge(loss_as_hp, how = 'outer')

    # get rounded values for plotting
    hp_info.loc[:, 'rounded_value'] = np.round(hp_info.loc[:, 'value'], 4)

    # add in percentages to plot with
    hp_info.loc[:, 'percent_value'] = 100*(
        (hp_info.loc[:, 'value'] - hp_info.loc[:, 'hyperparameters_min']
        )/(hp_info.loc[:, 'hyperparameters_max'] - hp_info.loc[:, 'hyperparameters_min']))


    hp_info.loc[:, 'percent_value'] = np.round(hp_info.loc[:, 'percent_value'], 3)
    hp_info = hp_info.loc[hp_info.hyperparameters.notna()]


    hp_info = hp_info.merge(trial_to_number, how = 'outer')
    hp_info = hp_info.sort_values(['n_trial', 'hyperparameters'])

    hp_info = hp_info.loc[hp_info.hyperparameters.notna(), ]
    
    
    # For Lineplot generation ------------------------------------------------------
    # In the line plot missing values are preventing a reasonable x axis
    hp_list = list(set(list(hp_info['hyperparameters'])))

    for trial in list(hp_info.loc[hp_info.trial_id.notna(), 'trial_id'].drop_duplicates()):
        trial_hp_list = list(hp_info.loc[hp_info.trial_id == trial, 'hyperparameters'])
        # print([entry for entry in hp_list if entry not in trial_hp_list])
        filler_df = pd.DataFrame({
            'hyperparameters':[entry for entry in hp_list if entry not in trial_hp_list],
            'trial_id': trial,
            'n_trial': list(hp_info.loc[hp_info.trial_id == trial, 'n_trial'].drop_duplicates())[0],
            'rounded_value': np.nan,
            'percent_value': 0.0 })
        filler_df = filler_df['hyperparameters'].astype('object') # prevents blanks from becoming nan floats and not merging
        hp_info = hp_info.merge(filler_df, how = 'outer')
    
    return(hp_info)



# %%
def mk_trial_lookup(hp_info, training_log):
    hp_trials = pd.DataFrame(hp_info.trial_id).rename(columns = {'trial_id':'hp_info'}).drop_duplicates().reset_index().drop(columns = ['index'])

    log_trials = pd.DataFrame(training_log.trial).rename(columns = {'trial':'training_log'}).drop_duplicates()
    log_trials['expected_trial'] = [entry.strip('trial_') for entry in list(log_trials['training_log'])]
    # for trial in hp_info, look for a regex match. 
    # if there is one add in that match. then join and check if there are missing values. 
    hp_trials['expected_trial'] = ''
    for i in hp_trials.index:
        entry_to_match = hp_trials.loc[i, 'hp_info']
        expected_trial_matches = [entry for entry in list(log_trials['expected_trial']) if re.match('[a-zA-Z_]*'+entry+'[a-zA-Z_]*', entry_to_match)]
        if len(expected_trial_matches) < 1:
            print("Warning!", entry_to_match, "had no match!")
        elif len(expected_trial_matches) > 1:
            print("Warning!", entry_to_match, "had multiple matches!\nNone will be used.\nMatches:",expected_trial_matches)
        else:
            hp_trials.loc[i, 'expected_trial'] = expected_trial_matches

    trial_matching_df = hp_trials.merge(log_trials, how = 'outer')
    return(trial_matching_df)

# mk_trial_lookup(hp_info, training_log)


# %%
def mk_violin_df(
    kt_trial_results
):
    # For Violin generation --------------------------------------------------------    
    ## Add desired labeling into df, prep df =======================================
    # remove duplicate validation fold
    violin_df = kt_trial_results.loc[kt_trial_results.data !=  'val', ].copy()

    # let us use trial number instead of mess of trial name.
    violin_df = violin_df.merge(trial_to_number, how = 'outer')
    violin_df = violin_df.sort_values('n_trial')
    violin_df = violin_df.loc[violin_df.data.notna(), ]
    return(violin_df)



# %%
def plot_faux_tensorboard(
    hp_info,
    violin_df,
    considered_loss_metric # = 'loss',
):
    fig = make_subplots(rows=2, cols=1, #column_widths=[0.7, 0.3], 
                        row_heights = [0.3, 0.7])

    # get all the factors ordered the way we want
    trials = hp_info.loc[hp_info.n_trial.notna(), 'n_trial']
    trials = pd.DataFrame(trials)
    trials = trials.astype(float).astype(int)
    trials = trials.sort_values(by = 'n_trial')
    trials = trials.drop_duplicates()
    trials = trials.astype(str)
    trials = list(trials['n_trial'])

    use_colors = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'   # blue-teal
    ]

    # repeat to cover all the trials
    accumulate_colors = []
    for i in range(math.ceil(len(trials)/len(use_colors))):
        accumulate_colors += use_colors
    use_colors = accumulate_colors


    color_idx = 0

    for trial in trials:
        temp =  hp_info.loc[hp_info['n_trial'] == trial, ]
        temp = temp.sort_values('hyperparameters')

        x_data    = temp.loc[:, 'hyperparameters']
        y_data    = temp.loc[:, 'percent_value']
        text_data = temp.loc[:, 'rounded_value']

    #     x_data = hp_info.loc[hp_info['n_trial'] == trial, 'hyperparameters']
    #     y_data = hp_info.loc[hp_info['n_trial'] == trial, 'percent_value']

        fig.add_trace(go.Scatter(
            x= x_data,
            y= y_data, 
            text = text_data,
            line_color=use_colors[color_idx],
            mode='lines+markers',
            name= trial
        ),
        row=1, col=1)


        x_data = violin_df['n_trial'][violin_df['n_trial'] == trial]
        y_data = violin_df[considered_loss_metric][violin_df['n_trial'] == trial]

        fig.add_trace(go.Violin(
            x= x_data,
            y= y_data, 
            name=trial,

            points='all',
            pointpos = 0.5,

            side='negative',
            line_color='Black', #use_colors[color_idx],
            fillcolor=use_colors[color_idx],
            # opacity = 0.3,

            box_visible=True,
            meanline_visible=True), 
        row = 2, col = 1)

        color_idx +=1

    fig.layout.update(template = 'plotly_white',
                      width = 1000,
                      height = 750)
    fig.update_yaxes(title_text="% Max Value", row=1, col=1)
    return(fig)


# %% [markdown]
# ## Scaling Processing to folders.

# %%

def mk_plots_for_model(
    model_path = '../data/atlas/models/Gpca_raw_k0',
    write_to_folder = '../reports/211215_checkin'
):
    # Setup variables for output ==================================================
    model_name = model_path.split('/')[-1]
    kt_path = '../data/atlas/models/'+model_name+'/kt/'
    csv_log_path = '../data/atlas/models/'+model_name+'/logs/training_log.csv'
    
    
    # Make sure that the needed paths exist =======================================
    if (os.path.exists(kt_path) & os.path.exists(csv_log_path)) == False:
        print('model_name='+str(model_name) )
        print('Warning: needed paths were not found.\n/kt dir found: '+str(os.path.exists(kt_path)
                                                 )+'\nlog csv found: '+str(os.path.exists(csv_log_path)))

        data_output = {
            'model': model_name,
            'dfs': {},
            'plotting_dfs': {}}

    else:
        # Extract data from /kt/ ======================================================
        results = kt_logs_to_dfs(kt_path=kt_path)
        kt_tested_hps = results['kt_tested_hps']
        kt_hp_ranges = results['kt_hp_ranges']
        kt_trial_results = results['kt_trial_results']
        # get loss metric in a df to make plotting with the hps easy.
        loss_as_hp = kt_loss_as_hp(kt_trial_results, considered_loss_metric='loss')


        # Extract data from /logs/ ====================================================
        training_log = get_training_log(csv_log_path)

        # Make training curves ========================================================
        fig = plot_loss_curves(training_log)
        fig.write_html(write_to_folder+'/'+model_name+"_loss.html")

        # Make plotting dfs ===========================================================
        trial_to_number = mk_trial_to_number_df(training_log)

        hp_info = mk_hp_info_df(
            kt_hp_ranges,
            kt_tested_hps,
            loss_as_hp,
            trial_to_number)

        violin_df = mk_violin_df(kt_trial_results)

        # Make faux tensorboards ======================================================

        fig = plot_faux_tensorboard(
            hp_info = hp_info,
            violin_df = violin_df,
            considered_loss_metric='loss'
        )
        fig.write_html(write_to_folder+'/'+model_name+"_tb.html")

        # Additional outputs go here ==================================================


        data_output = {
            'model': model_name,
            
            'dfs': {
                'kt_tested_hps': kt_tested_hps,
                 'kt_hp_ranges': kt_hp_ranges,
             'kt_trial_results': kt_trial_results,
                   'loss_as_hp': loss_as_hp,
                 'training_log': training_log},
            
            'plotting_dfs': {
              'trial_to_number': trial_to_number,
                      'hp_info': hp_info,
                    'violin_df': violin_df}
        }

    return(data_output)


def mk_data_for_model(
    model_path = '../data/atlas/models/Gpca_raw_k0'
):
    # Setup variables for output ==================================================
    model_name = model_path.split('/')[-1]
    kt_path = '../data/atlas/models/'+model_name+'/kt/'
    csv_log_path = '../data/atlas/models/'+model_name+'/logs/training_log.csv'
    
    
    # Make sure that the needed paths exist =======================================
    if (os.path.exists(kt_path) & os.path.exists(csv_log_path)) == False:
        print('model_name='+str(model_name))
        print('Warning: needed paths were not found.\n/kt dir found: '+str(os.path.exists(kt_path)
                                                 )+'\nlog csv found: '+str(os.path.exists(csv_log_path)))

        data_output = {
            'model': model_name,
            'dfs': {},
            'plotting_dfs': {}}

    else:
        # Extract data from /kt/ ======================================================
        results = kt_logs_to_dfs(kt_path=kt_path)
        kt_tested_hps = results['kt_tested_hps']
        kt_hp_ranges = results['kt_hp_ranges']
        kt_trial_results = results['kt_trial_results']
        # get loss metric in a df to make plotting with the hps easy.
        loss_as_hp = kt_loss_as_hp(kt_trial_results, considered_loss_metric='loss')


        # Extract data from /logs/ ====================================================
        training_log = get_training_log(csv_log_path)

        # Make plotting dfs ===========================================================
        trial_to_number = mk_trial_to_number_df(training_log)

        hp_info = mk_hp_info_df(
            kt_hp_ranges,
            kt_tested_hps,
            loss_as_hp,
            trial_to_number)

        violin_df = mk_violin_df(kt_trial_results)

        # Additional outputs go here ==================================================
        data_output = {
            'model': model_name,
            
            'dfs': {
                'kt_tested_hps': kt_tested_hps,
                 'kt_hp_ranges': kt_hp_ranges,
             'kt_trial_results': kt_trial_results,
                   'loss_as_hp': loss_as_hp,
                 'training_log': training_log},
            
            'plotting_dfs': {
              'trial_to_number': trial_to_number,
                      'hp_info': hp_info,
                    'violin_df': violin_df}
        }

    return(data_output)


# %% [markdown]
# ## Processing multiple types for comparisons:

# %%
# result str
# model
# dfs
#   - kt_tested_hps
#   - kt_hp_ranges
#   - kt_trial_results
#   - loss_as_hp
#   - training_log
# plotting_dfs
#   - trial_to_number
#   - hp_info
#   - violin_df

# violin df is what we expect to be most useful here. 
def extract_df_for_model(
    result_list, # = result_list, # list of folders to use
    model_name = 'Gpca_k',     # Grouping name
    key1 = 'plotting_dfs',
    key2 = 'violin_df'
):
    accumulator = None
    # find the indices that match the desired model name
    matching_result_idx = [i if re.match(model_name, result_list[i]['model']) else None for i in range(len(result_list))]
    # at those indices get the df specified by the user
    first = True
    for i in matching_result_idx:
        if key1 in list(result_list[i].keys()):
            if key2 in list(result_list[i][key1].keys()):
                temp = result_list[i][key1][key2]
                temp['model'] = result_list[i]['model']
                if first == True:
                    accumulator = temp
                    first = False
                else:
                    accumulator = accumulator.merge(temp, how = 'outer')

    return(accumulator)


# %% [markdown]
# ## Scale to consider multiple model types

# %% code_folding=[7, 55]
# This is a replacement of the previous way of making a list of dictionaries containing results. 
# This version is useful in so far as it only looks at folders containing results so there are 
# no list entries that need to be skipped (expected, at least).
# Additionally by matching upfront matching by the model name in `extract_df_for_model` should
# have no effect.

# if there were no results found with the string/path combination None will be returned.
def make_model_type_result_list(
    match_str = 'Gpca_k',
    search_path = '../data/atlas/models/'
    ):
    def get_models_with_results(match_str = 'Gpca_k',
                            search_path = '../data/atlas/models/'):
        # reduce to matches
        possible_matches = os.listdir(search_path)
        possible_matches = [possible_match for possible_match in possible_matches if re.match(match_str+'.*', possible_match)]

        if len(possible_matches) != 0:

            possible_match = possible_matches[0]
            def check_possible_match(possible_match, search_path
                                    ):
                # check that matches have expected files
                kt_exists = os.path.exists(search_path+possible_match+'/kt/')
                if kt_exists:
                    kt_contains = os.listdir(search_path+possible_match+'/kt/')
                    kt_trial_exists = True in [True if re.match('trial_.+', entry) else False for entry in kt_contains]
                else:
                    kt_trial_exists = False
                log_exists = os.path.exists(search_path+possible_match+'/logs/training_log.csv')

                # Return path if it's okay
                if kt_exists & kt_trial_exists:
                    return(possible_match)

            # Winnow down to only those that have the expected files/directories
            matches = [check_possible_match(possible_match = entry, 
                                            search_path = search_path) for entry in possible_matches ]
            matches = [entry for entry in matches if entry != None]
            return(matches)
    

    models_with_res = get_models_with_results(
        match_str = match_str,
        search_path = search_path)
    
    # if there were no matches the output above is none.
    if models_with_res != None:
        paths_with_res = [search_path+model for model in models_with_res]

        res_list = map(mk_data_for_model, paths_with_res )
        res_list = list(res_list)
        return(res_list)
    
def get_best_models_from_vdf(vdf):
    # add in the model-wise minimum of trial-wise max(mae)
    # this is harder to do than in R because .assign isn't working with grouped df
    trial_max = vdf.groupby(['model', 'trial_id']).agg(max_mae = ('mae', np.nanmax)).reset_index()
    model_min = trial_max.groupby('model').agg(min_mae = ('max_mae', np.nanmin)).reset_index()
    model_min_max_mae = trial_max.merge(model_min, how = 'outer')

    # now we can reduce this down to the best results for each model set.
    mask = model_min_max_mae.max_mae == model_min_max_mae.min_mae
    model_min_max_mae = model_min_max_mae.loc[mask, ]

    # finally we can left join to get only those best models to visualize.
    best_models = model_min_max_mae.loc[:, ['model', 'trial_id']].merge(vdf, how = 'left')
    
    return(best_models)


# %% code_folding=[1, 18]
# wrapper to make it easy to get all the model outputs:
def wrap_violin_prep(match_str = 'Gpca_k'):
    res_list = make_model_type_result_list(
        match_str = match_str,
        search_path = '../data/atlas/models/')

    vdf = extract_df_for_model(
            result_list = res_list,
            model_name = match_str,
            key1 = 'plotting_dfs',
            key2 = 'violin_df'
        )   

    best_vdf = get_best_models_from_vdf(vdf)
    return(best_vdf)

def wrap_violin_prep_top_n(match_str = 'Gpca_k', top_n = 1):

    res_list = make_model_type_result_list(
            match_str = match_str,
            search_path = '../data/atlas/models/')

    vdf = extract_df_for_model(
            result_list = res_list,
            model_name = match_str,
            key1 = 'plotting_dfs',
            key2 = 'violin_df'
        ) 

    # add in the model-wise minimum of trial-wise max(mae)
    # this is harder to do than in R because .assign isn't working with grouped df
    trial_max = vdf.groupby(['model', 'trial_id']).agg(max_mae = ('mae', np.nanmax)).reset_index()
    trial_max = trial_max.sort_values('max_mae').reset_index().drop(columns = 'index')

    trial_max = trial_max.loc[0:(top_n-1), ]

    # # finally we can left join to get only those best models to visualize.
    best_models = trial_max.loc[:, ['model', 'trial_id']].merge(vdf, how = 'left')
    return(best_models)


# %%

# %% [markdown]
# ## Convenience functions to make processing easier

# %%
"""
Now we can pull these in two ways. 

1. If we just want one, we can use this:
#make_model_type_result_list(
#        match_str = 'Gpca_k0',
#        search_path = '../data/atlas/models/')
and then filter the results. 

2. If we want many we can write another wrapper
and filter the outputs. 
"""

# wrapper to make it easy to get all the model outputs:
def wrap_get_hp_info(match_str = 'Gpca_k'):
    res_list = make_model_type_result_list(
        match_str = match_str,
        search_path = '../data/atlas/models/')

    hpdf = extract_df_for_model(
            result_list = res_list,
            model_name = match_str,
            key1 = 'plotting_dfs',
            key2 = 'hp_info'
        )   
    return(hpdf)

# desired_models = desired_models.merge(Gpca_hps, how = 'left')
# desired_models


# %%
def wrap_hp_plot_by_model(plot_df):
    fig = go.Figure()
    model_groups = list(plot_df.model.drop_duplicates())

    plotly_qual_hex = px.colors.qualitative.Plotly
    uniq_model_groups = list(plot_df.model_group.drop_duplicates())
    models = list(plot_df.model.drop_duplicates())

    for model in models:
        temp = plot_df.loc[plot_df['model'] == model, ]
        temp = temp.sort_values('hyperparameters')

        x_data    = temp.loc[:, 'hyperparameters']
        y_data    = temp.loc[:, 'percent_value']
        text_data = temp.loc[:, 'rounded_value']

        model_group = list(temp.loc[:, 'model_group'])[0]
        color_idx = [i for i in range(len(uniq_model_groups)) if model_group == uniq_model_groups[i]][0]


        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            text = text_data,
    #         line_color=plotly_qual_hex[color_idx],
            mode='lines+markers',
            name= model
        ))


    fig.update_layout(
        template = 'plotly_white',
                      width = 1000,
                      height = 750,
        legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))


    fig.update_yaxes(title_text="% HP Range")
    # fig.show()
    return(fig)


# %% [markdown]
# ## Write out selected hps for future use.

# %%
def write_trial_hps(df_in,# = Gpca_hps,
                    save_path = '../data/atlas/models/Gpca_finalize_hp/'):
    for trial in list(df_in.trial_id.drop_duplicates()): 
        temp = df_in.loc[df_in.trial_id == trial, ['hyperparameters', 'value']]
        temp = temp.to_dict('list') # 'list' has the column names as keys, and the entries as values. 
        
        # Convert any character hps back to characters.
        # there should only be a comma in a hyperparameter if it is a character entry that's been coereced to an int.
        hps_to_convert = [entry for entry in temp['hyperparameters'] if re.match('.*,.*', entry)]

        for hp in hps_to_convert:
            hp_official_name = '_'.join(hp.split('_')[:-1])
            hp_options = hp.split('_')[-1].split(',')

            hp_idx = [i for i in range(len(temp['hyperparameters'])) if temp['hyperparameters'][i] == hp]
            hp_idx = hp_idx[0]
            hp_option_idx = int(temp['value'][hp_idx])

            hp_replacement_value = hp_options[hp_option_idx]

            temp['value'][hp_idx] = hp_replacement_value
            temp['hyperparameters'][hp_idx] = hp_official_name        
        
        
        
        with open(save_path+'hps_'+trial+'.json', 'w') as f:
            json.dump(temp, f)

# %%

# %% [markdown]
# ##  Demo internal functionality

# %% [markdown] code_folding=[]
# ### Processing a single folder:

# %% code_folding=[]
# Full join to create a tidy df with the kt results
# kt_data = kt_hp_ranges.merge(kt_tested_hps, how = 'outer'
#                      ).merge(kt_trial_results, how = 'outer')

# %%
# Make dataframes from the kt folders
demo_folder = '4_hp_search_concat_v0_k9'
demo_output = 'concat_v0_hps'

results = kt_logs_to_dfs(kt_path = '../data/atlas/models/'+demo_folder+'/kt/')
kt_tested_hps    = results['kt_tested_hps']
kt_hp_ranges     = results['kt_hp_ranges'] 
kt_trial_results = results['kt_trial_results'] 

# get loss metric in a df to make plotting with the hps easy.
loss_as_hp = kt_loss_as_hp(kt_trial_results, considered_loss_metric = 'loss')
loss_as_hp.head()

# %%
csv_log_path = '../data/atlas/models/'+demo_folder+'/logs/training_log.csv'
training_log = get_training_log(csv_log_path)
training_log.head()

# %%
# training_log
# make a nice plotting df


# %%
trial_to_number = mk_trial_to_number_df(training_log)

hp_info = mk_hp_info_df(
    kt_hp_ranges,
    kt_tested_hps,
    loss_as_hp,
    trial_to_number)

hp_info.head()

# %%
violin_df =mk_violin_df(kt_trial_results)
violin_df.head()

# %% code_folding=[2]
fig = plot_faux_tensorboard(
    hp_info,
    violin_df,
    considered_loss_metric = 'loss'
)
# fig.write_html("../reports/211215_checkin/faux_tb.html")
fig

# %% [markdown]
# ### Scale to folders:

# %%
# Goal:
# Given a directory, pull all the trial that have completed 
# extract all the information we can and return
# 1. Plot showing hp-validation perfomrance, cv performance
# 2. nicely organized dfs with
#   a. training history
#   b. hyps

# %%
model_path = '../data/atlas/models/'+demo_folder+'/'
write_to_folder = '../reports/'+demo_output

model_name = model_path.split('/')[-1]
kt_path = '../data/atlas/models/'+model_name+'/kt/'
csv_log_path = '../data/atlas/models/'+model_name+'/logs/training_log.csv'


# %%
demo_data = mk_data_for_model(
    model_path = '../data/atlas/models/'+demo_folder
)

# %%
demo_plts = mk_plots_for_model(
    model_path = '../data/atlas/models/'+demo_folder,
    write_to_folder = '../reports/'+demo_output
)

# %% [markdown]
# ## Apply to models where possible
#
# 1. Automatically find the folders to be used then process them. 

# %%
# TODO mk_data_for_model is failing! Needs refactoring.

# %%
model_names = ['4_hp_search_concat_v0_k', 
#                'Gpca_raw_k', 'Gnum_k', 
#                'S_k', 'Wmlp_k', 'Wc1d_k', 'Wc1d_k'
              ]
model_ks = [str(i) for i in range(10)]

result_list = []

import tqdm
for model_name in model_names:
    for model_k in tqdm.tqdm(model_ks):
        model_path = '../data/atlas/models/'+model_name+model_k
#         result = mk_plots_for_model(
#             model_path = model_path,
#             write_to_folder = '../reports/211215_checkin'
#         )
        result = mk_data_for_model(model_path = model_path)
        
        result_list.extend([result])

# %%

# %%
# list(result_list[0].keys())

# %% [markdown]
# ## On a single model demonstrate possible cross fold hp selection criteria

# %%
M = extract_df_for_model(
    result_list = result_list,
    model_name = '4_hp_search_concat_v0',
    key1 = 'plotting_dfs',
    key2 = 'violin_df'
)
M.head()

# %%
# we ultimately don't want plots but statistics. Lets go there first. 
M['k'] = [entry.split('_')[-1] for entry in list(M['model'])]

# M[['train_mae', 'train_mse']] = np.nan
# separate the validation from the training results:
training_fold = M.loc[M.data == M.k, ['trial_id', 'mae', 'mse']
                     ].rename(columns ={'mae': 'train_mae', 'mse': 'train_mse',})
M = M.loc[M.data != M.k, ]

M = M.merge(training_fold, on = 'trial_id')
M.head()
# M.assign(train_mae = lambda x: np.nanmedian(x['trial_id'].groupby('trial_id')))

# %%
M_summary_stats = M.groupby(['model', 'k', 'trial_id']).agg(
    val_mae_mean = ('mae', np.nanmean),
    val_mae_max = ('mae', np.nanmax),
    val_mae_min = ('mae', np.nanmin),
    val_mae_std  = ('mae', np.nanstd),
    val_mse_mean = ('mse', np.nanmean),
    val_mse_std  = ('mse', np.nanstd),
    train_mae_mean = ('train_mae', np.nanmean),
    train_mse_mean = ('train_mse', np.nanmean)
).reset_index()

M_summary_stats = M_summary_stats.assign(val_cvish = M_summary_stats.val_mae_mean+(M_summary_stats.val_mae_std / M_summary_stats.val_mae_mean) )

M_summary_stats.head()


# %%
## Plot from M

# %%
fig = make_subplots(rows=1, cols=1, subplot_titles=(""))

fig.add_trace(go.Scatter(
    x = M['train_mae'], 
    y=  M['mae'], 
    mode = 'markers',
    name = 'CV Loss'))
# unity
fig.add_trace(go.Scatter(
    x = M['train_mae'], 
    y=  M['train_mae'], 
    mode = 'markers',
    name = 'Training Loss'))

fig.update_xaxes(range = [0,2], title_text="MAE - Training fold")
fig.update_yaxes(range = [0,2], title_text="MAE - Other folds")
fig.layout.update(template = 'plotly_white',
#                   width = 1000,
#                   height = 750
                 )

fig.write_html("../reports/concat_v0_hps/validation_performance.html")
fig.show()


# %%
# Plot from summary stats

# %%
fig = make_subplots(rows=1, cols=1, subplot_titles=(["Performance ~ Consistency"]))

fig.add_trace(go.Scatter(
    x = M_summary_stats['val_mae_std'], 
    y=  M_summary_stats['val_mae_mean'], 
    mode = 'markers',
    name = 'CV'))
# unity

fig.update_xaxes(range = [0,2], title_text="MAE SD")
fig.update_yaxes(range = [0,2], title_text="MAE Mean")
fig.layout.update(template = 'plotly_white',
#                   width = 1000,
#                   height = 750
                 )

fig.show()
fig.write_html("../reports/concat_v0_hps/validation_mean_by_sd.html")

# %% code_folding=[18, 26, 29, 34, 42, 52, 62]
fig = make_subplots(rows=1, cols=1, subplot_titles=(""))


# Data Ranges -----------------------
# for i in range(M_summary_stats.shape[0]):
#     fig.add_shape(type="line",
#         xref="x", yref="y",
#         x0=M_summary_stats.loc[i, 'train_mae_mean'], 
#         y0=M_summary_stats.loc[i, 'val_mae_min'], 
#         x1=M_summary_stats.loc[i, 'train_mae_mean'], 
#         y1=M_summary_stats.loc[i, 'val_mae_max'],
#         line=dict(
#             color="#3d3d3d",
#             width=1            
#         ), 
#         opacity = 0.5)
    
# Data ------------------------------
fig.add_trace(go.Scatter(
    x = M['train_mae'], 
    y=  M['mae'], 
    mode = 'markers',
    line_color = '#d4d4d4',
    name = 'CV Loss'))
                  
# Possible Selections metrics:
min_max_mae_sort = M_summary_stats.sort_values('val_mae_max', ascending = True
                                              ).reset_index(
                                              ).drop(columns = ['index'])
cvish_sort       = M_summary_stats.sort_values('val_cvish', ascending = True
                                              ).reset_index(
                                              ).drop(columns = ['index'])
# Lines -----------------------------
##  Min( Max( Mae ) ) ===============
fig.add_trace(go.Scatter(
    x = min_max_mae_sort.sort_values('train_mae_mean').loc[:, 'train_mae_mean'], 
    y=  min_max_mae_sort.sort_values('train_mae_mean').loc[:, 'val_mae_max'],  
    line_color = '#9803fc',
    opacity = 0.5,
    mode = 'lines',
    name = 'Max MAE'))
##  Min( mean + ( sd/mean ) ) =======
fig.add_trace(go.Scatter(
    x = cvish_sort.sort_values('train_mae_mean').loc[:, 'train_mae_mean'], 
    y=  cvish_sort.sort_values('train_mae_mean').loc[:, 'val_cvish'], 
    line_color= '#0300c4',
    opacity = 0.5,
    mode = 'lines',
    name = 'Mean+Coef.Var'))

# Best Points -----------------------
##  Min( Max( Mae ) ) ===============
fig.add_trace(go.Scatter(
    x = [min_max_mae_sort.loc[0, 'train_mae_mean'], min_max_mae_sort.loc[0, 'train_mae_mean']], 
    y=  [min_max_mae_sort.loc[0, 'val_mae_max'], min_max_mae_sort.loc[0, 'val_mae_max']], 
#     line_color = '#9803fc',
    marker=dict(size=[8, 4],color=['#9803fc', '#ffffff']),
    mode = 'markers',
    name = 'Best Max MAE'))


##  Min( mean + ( sd/mean ) ) =======
fig.add_trace(go.Scatter(
    x = [cvish_sort.loc[0, 'train_mae_mean'], cvish_sort.loc[0, 'train_mae_mean']], 
    y=  [cvish_sort.loc[0, 'val_cvish'], cvish_sort.loc[0, 'val_cvish']],
#     line_color= '#0300c4',
    marker=dict(size=[8, 4],color=['#0300c4', '#ffffff']),
    mode = 'markers',
    name = 'Best Mean+Coef.Var'))


fig.update_xaxes(range = [0,2], title_text="MAE - Training fold")
fig.update_yaxes(range = [0,2], title_text="MAE - Other folds")
fig.layout.update(template = 'plotly_white',
#                   width = 1000,
#                   height = 750
                 )

fig.show()

fig.write_html("../reports/concat_v0_hps/validation_selection_criteria.html")

# %% [markdown]
# # Visualize performance across folds and across models

# %%
model_names = ['4_hp_search_concat_v0_k'
#     'Gpca_k', 'Gpca_raw_k', 'Gnum_k', 
#                'S_k', 'Wmlp_k', 'Wc1d_k', 'Wc1d_k'
              ]

match_str = '4_hp_search_concat_v0_k'

res_list = make_model_type_result_list(
    match_str = match_str,
    search_path = '../data/atlas/models/')

vdf = extract_df_for_model(
        result_list = res_list,
        model_name = match_str,
        key1 = 'plotting_dfs',
        key2 = 'violin_df'
    )   

get_best_models_from_vdf(vdf)

# %%

# %%

# %%

# %%

# %%

# %%



# %%

# %% code_folding=[]

# %%

# %%

# %% code_folding=[1]
# violin_df_list = map(wrap_violin_prep, ['Gpca_k', 'Gpca_raw_k', 'Gnum_k', 'S_k', 'Wmlp_k', 'Wc1d_k', 'Wc1d_k'])
# violin_df_list = list(violin_df_list)

# %%
# Gpca_vdf = wrap_violin_prep(match_str = 'Gpca_k')
# Gnum_vdf = wrap_violin_prep(match_str = 'Gnum_k')
# Gpca_raw_vdf = wrap_violin_prep(match_str = 'Gpca_raw_k')
# S_vdf = wrap_violin_prep(match_str = 'S_k')
# Wmlp_vdf = wrap_violin_prep(match_str = 'Wmlp_k')
# Wc1d_vdf = wrap_violin_prep(match_str = 'Wc1d_k')
# Wc2d_vdf = wrap_violin_prep(match_str = 'Wc2d_k')

# go across all models with results, merge into 

violin_df_list = [wrap_violin_prep(match_str = '4_hp_search_concat_v0_k')
    
]

# %%
# apply accumulator pattern
first = True
for i in range(len(violin_df_list)):
    if first:
        accumulator = violin_df_list[i]
        first = False
    else:
        accumulator = accumulator.merge(violin_df_list[i], how = 'outer')
        
df = accumulator
df.head()

# %%
# prepare for plotting ====
model_types = pd.DataFrame(df.loc[:, 'model'].drop_duplicates())
# drop the _k\d from the end
model_types['model_group'] = ['_'.join(entry.split('_')[:-1]) for entry in list(model_types['model'])]
df = model_types.merge(df, how = 'outer')
df.head()


# %%

# %%

# %%

# %%

# %%

# %%

# %% code_folding=[9]
fig = go.Figure()
models = list(df.model.drop_duplicates())

plotly_qual_hex = px.colors.qualitative.Plotly
uniq_model_groups = list(df.model_group.drop_duplicates())

for model in models:
    # find right color for this group
    model_group = list(df.loc[df.model == model, 'model_group'])[0]
    color_idx = [i for i in range(len(uniq_model_groups)) if model_group == uniq_model_groups[i]][0]
    # plot
    fig.add_trace(go.Violin(
        x=df.loc[df.model == model, 'model'],
        y=df.loc[df.model == model, 'mae'],

        name = model,
        points='all',
        pointpos = 0.5,
        side='negative',
        box_visible = True,
        meanline_visible = True,
        
        line_color='Black', #use_colors[color_idx],
        fillcolor=plotly_qual_hex[color_idx]
        ))


fig.update_yaxes(title_text="Mae", 
                 #type="log", 
                 range = [0,1.25])

fig.update_layout(
    template = 'plotly_white',
                  width = 1000,
                  height = 750,
    legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))

fig.write_html("../reports/concat_v0_hps/best_cross_trials.html")
fig.show()



# %% [markdown] heading_collapsed=true
# ## Big picture, within one set of trials, what's a pca look like?  

# %% hidden=true

# %% [markdown]
# ## Given a model (e.g. the best in the set) get the hyperparameters for it.

# %%

# %%
# #previously we've found some models we're interested in.
# desired_models = df.loc[:, #df.model_group == 'Gpca', 
#                         ['model', 'model_group', 'trial_id']].drop_duplicates()
# desired_models

# %%

# %%
# replaced, get the top 2 per trial.
# Because `wrap_violin_prep_top_n` returns the top `n` per search string, we need to step through 
# each k separately or we'll get the top n even if they're all from one abnormal fold.
search_strs = []
for entry in ['4_hp_search_concat_v0_k']:
    search_strs.extend([entry+str(k) for k in range(10)])
    
# top_2_each = pd.concat([wrap_violin_prep_top_n(match_str = search_str, top_n = 2) for search_str in search_strs])
# Done verbosely because the above failed when a file did not copy from atlas. 
top_2_each = []
for search_str in search_strs:
    if not search_str in []:    
        print(search_str)
        if search_str in os.listdir("../data/atlas/models/"):      
            top_2_each.append(wrap_violin_prep_top_n(match_str = search_str, top_n = 2))
        else:
            print(' ^ Not found.')

top_2_each = pd.concat(top_2_each)





model_groups = top_2_each.loc[:, ['model']].drop_duplicates()
model_groups['model_group'] = [entry.split('_')[0] for entry in list(model_groups['model'])]
top_2_each = top_2_each.merge(model_groups)

top_2_each = top_2_each.loc[:, ['model', 'model_group', 'trial_id']].drop_duplicates()
desired_models = top_2_each

# %%

# %%

# %%

# %%
Gpca_hps = wrap_get_hp_info(match_str = '4_hp_search_concat_v0_k')
Gpca_hps = desired_models.merge(Gpca_hps, how = 'inner')

# %%
Gpca_hps.head()

# %%
wrap_hp_plot_by_model(plot_df = Gpca_hps)

# %% [markdown]
# ## Now write out those hps to be used elsewhere

# %%
# Gpca_hps.to_csv('../data/atlas/models/Gpca_finalize_hp/hpsSelectedForGpca.csv')
Gpca_hps.head()

# %%
write_trial_hps(
    df_in = Gpca_hps, 
    save_path = '../data/atlas/models/5_concat_v0_finalize_hp/')
