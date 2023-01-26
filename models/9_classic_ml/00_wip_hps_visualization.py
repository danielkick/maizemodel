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
import os, re, json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# %%

# %% code_folding=[149, 152, 155]
class hps_search_experiment:
    """
    This is a class to work with all the results for hps on a 
    single data modality (e.g. soil) and algorithm (e.g. knn).    
    # for each file we'll 
    # 1. Load the file
    # 2. Decide if it's a 
    #    - Trial result
    #    - The best hps in the set <------------- There is not one of these for each trial type. (search ended early)
    #    - Trial search space string
    # 3. Parse each 
    #    - Trial -> hps, losses
    #    - Best hps -> save as flag
    #    - Space -> wide df with min max or levels of each hps.
    
    """
    def __init__(self, path="", trial_type=''):
        self.path = path
        self.trial_type = trial_type
        self.hps_best_trial = None
        self.hps_search_space = None
        self.hps_trials = None
                
    def _load_json(self, path, file):
        with open(path+file,'r') as f:
            dat = json.load(f)
        return(dat)        
    
    def _process_trial(self, current_file):
        trial_res = self._load_json(self.path, file = current_file)
        df_res = pd.DataFrame(
            zip(trial_res['cv_folds'], 
                trial_res['cv_losses']),
                columns = ['cv_folds', 'cv_losses'])
        # add all individual values to the df as a col
        for key in [key for key in trial_res.keys() if key not in ['cv_folds', 'cv_losses']]:
            df_res[key] = trial_res[key]

        df_res['trial_file'] = current_file
        return(df_res)    

    def _process_best_hps(self, current_file):
        hps_best = pd.DataFrame(self._load_json(path, file = current_file), index = [0])
        hps_best['best'] = True
        return(hps_best)

        # current_file = 'hps_rnr_hp_search_space.json'
    #     def _process_space_str(self, current_file):    
    #         # parse string into useful df. 
    #         hp_search_space_string = _load_json(path, file = current_file)
    #         assert type(hp_search_space_string) == str, print('search space is not a string.')
    #         # all catagorical values will need to become a lookup index.

    #         # clean up string
    #         # Get rid of the text before the hps dict starts
    #         stringStart =  re.match(r'.*\{', hp_search_space_string, re.MULTILINE)
    #         stringStart = stringStart.group(0)
    #         hp_search_space_string = hp_search_space_string.replace(stringStart, '')

    #         # retrieve the contents of the hps dictionary by dropping everything after the last }, get rid of newlines
    #         hp_search_space_string = re.match(r'[^\}]+', hp_search_space_string, re.MULTILINE).group(0)
    #         hp_search_space_string = hp_search_space_string.replace('\n', '')

    #         # find each of the dictionary entries. Then interate over each retruning a  list with the hp type and values. 
    #         res = re.findall(r"[^\,]+hp\.[^\)]+", hp_search_space_string)

    #         res_df_list = []
    #         for entry in res:
    #             # for each match get the dict key, 
    #             hp_dict_key = re.match(r'[^:]+', entry).group().strip("'").strip('"') # strip single and double quotes in case there's non standarad use of them in some files.

    #             # the hp type, 
    #             hp_type = re.search(r'hp\.[^\(]*', entry).group().strip('hp.')

    #             # the hp name,
    #             hp_name = re.search(r"hp\.[\w]+\('[\w]+'", entry).group().split('(')[-1].strip("'").strip('"')

    #             # and all values after the hp name
    #                 # if there are two values and they're not in a list they should be called min, max
    #                 # if there's a list, then the they should take the form "name_i"
    #             # get the content of the hp function
    #             hp_content = re.search(r"hp\.[\w]+\('[\w]+.+", entry).group()
    #             # use the name of the hp to get a list entry containing the variables
    #             hp_content = hp_content.split(hp_name)
    #             hp_vars = [e.replace("'", "").replace('"', "") for e in hp_content][-1]


    #             # This assumes that if a single variable is named all of them will be. 
    #             # TODO make this more robust. What if one of the arguments is position based and the rest are named?
    #             has_varname = [True if re.findall('=', e) else False for e in hp_vars]
    #             if True in has_varname:
    #                 clean_hp_var_names = []
    #                 clean_hp_vars = []

    #                 for current_var in hp_vars:
    #                     split_var = current_var.split('=')
    #                     clean_hp_var_names.extend([hp_name+'_'+split_var[0]])
    #                     clean_hp_vars.extend([split_var[1]])

    #                 hps_space = pd.DataFrame(
    #                                 dict(zip(
    #                                     clean_hp_var_names,
    #                                     clean_hp_vars)),
    #                                 index = [0]
    #                             )
    #             # Contains list
    #             elif re.match('.+\[.+', hp_vars):
    #                 hp_vars = hp_vars.strip(',').replace(' ', '').replace('[', '').replace(']', '').split(',')
    #                 hps_space = pd.DataFrame(
    #                     dict(zip(
    #                         [hp_name+'_'+str(i) for i in range(len(hp_vars))], # current_weight_metric_0 .. _i
    #                         hp_vars)),
    #                     index = [0]
    #                 )
    #             else:
    #                 # non list 
    #                 # Drop any leading ',' and drop whitespace
    #                 hp_vars = hp_vars.strip(',').replace(' ', '').split(',')
    #                 assert len(hp_vars) == 2, "Expected 2 hyperparameter variables (min/max)." 
    #                 hps_space = pd.DataFrame(
    #                     dict(zip(
    #                         [hp_name+'_min', hp_name+'_max'], 
    #                         hp_vars)),
    #                     index = [0]
    #                 )

    #             # add in other extracted values
    #             hps_space[hp_name+'_key'] = hp_dict_key
    #             hps_space[hp_name+'_type'] = hp_type

    #             res_df_list.extend([hps_space])   

    #         hps_space = pd.concat(res_df_list, axis = 1)
    #         return(hps_space)
    
    def process_hps_files(self):
        trial_list = []
        best_list = []
        space_list = []
        path = self.path

        dir_contents = os.listdir(path)
        # restrict to specificed model type
        dir_contents = [e for e in dir_contents if re.findall(trial_type, e)]


        # current_file = dir_contents[0] # Fixme!
        for current_file in dir_contents:
            # Find current state -- what type of file are we processing?
            if re.match('^hps_.+_\d+.json$', current_file):
                file_type = 'trial' 

            if re.match('.+best\.json$', current_file):
                file_type = 'best'

            if re.match('.+hp_search_space\.json$', current_file):
                file_type = '' #'space' # todo - ideally we want to recreate the search space from file not from the code.

            # Apply rules for current state
            if file_type == 'trial':
                df_res = self._process_trial(current_file = current_file)
                trial_list.extend([df_res])

            if file_type == 'best':
                hps_best = self._process_best_hps(current_file = current_file)
                best_list.extend([hps_best])

            if file_type == 'space':
                trial_space = self._process_space_str(current_file = current_file)
                space_list.extend([trial_space])
            # clear current state so that if there's an error the script fails fast.
            file_type = ''

        if len(best_list) > 1:
            Warning('Only one best hps set should exist per search, '+str(len(best_list))+' found.')
#         hps_best_trial = best_list[0]
            hps_best_trial = pd.concat(best_list, axis = 0)
    
            self.hps_best_trial = hps_best_trial

        if len(space_list) >= 1:
            Warning('Only one search space should exist per search.\nChecking for equivalence.')
            print(space_list)
            print(space_list[0] == space_list[1])
            hps_search_space = space_list[0]     
            
            self.hps_search_space = hps_search_space
            
        assert len(trial_list) != 0, 'No trials found!'
        hps_trials = pd.concat(trial_list, axis = 0)
        self.hps_trials = hps_trials        


# %%

# %% code_folding=[19, 30]
# create a tidy df to work with. 
# find all the experiment containing directories ('All', 'G', 'S', 'WOneHot'),
#    find all the ML methods within it ('rf', 'knn', 'rnr', 'vrl')
#        Create a nice df to work with
# then merge all of them into a single df

exp_res_list = []

exp_containing_dirs = [e for e in os.listdir() if re.match('hps_search_intermediates_.+', e)]
input_data_types = [e.split('_')[-1] for e in exp_containing_dirs]
# input_data_type = 'G'

for input_data_type in input_data_types:
    path = './hps_search_intermediates_'+input_data_type+'/'
    # figure out what trial_types are in a give directory:
    # ['rf', 'knn', 'rnr', 'vrl']
    trial_types = list(set([re.match('hps_[^_]+', e).group(0).strip('hps_') for e in os.listdir(path)]))

    # trial_type = trial_types[0]
    for trial_type in trial_types:
        temp = hps_search_experiment(
            path = path, 
            trial_type = trial_type)
        
        print(path, trial_type)
        temp.process_hps_files()

        # frustratingly, I must have the search space written out at the end of the search not the beginning. 
        # Since we can't gurantee that we'll have it, we must lookup the values from the generative script.
        # temp.hps_best_trial
        # temp.hps_search_space
        exp_res = temp.hps_trials
        exp_res['method'] = trial_type
        exp_res['input_data'] = input_data_type

        exp_res_list.extend([exp_res])


# %%
accumulator = pd.DataFrame()

for exp in exp_res_list:
    if accumulator.shape[0] == 0:
        accumulator = exp
    else:
        accumulator = accumulator.merge(exp, 'outer')



# %%
accumulator

# %%
df = accumulator

# fix method name (due to strip)
df.loc[df['method'] == 'vrl', 'method'] = 'svrl'

# What's the best hps perameters for each
# method x input_data ?


# what's the best average loss for each 
# data x 

# %%
df.loc[:, ['method', 'input_data', 'trial_file']
      ].drop_duplicates(
      ).groupby(['input_data', 'method']
      ).count().T#.reset_index()


# %%
import numpy as np
df_summary = df.groupby(['method', 'input_data', 'trial_file']).agg(ave_cv_loss = ('cv_losses', np.mean)).reset_index()
df = df.merge(df_summary, how = 'outer')

# %%
is_min = df_summary.groupby(['method', 'input_data']).agg(ave_cv_loss = ('ave_cv_loss', np.min)).reset_index()
is_min['is_min'] = True

is_min = is_min.merge(df, how = 'outer')

is_min = is_min.loc[is_min['is_min'] == True].drop(columns = ['cv_folds', 'cv_losses']).drop_duplicates()
is_min

# %%

# %%
mask = is_min['method'] == 'knn'
knn_min = is_min.loc[mask]
knn_min.dropna(axis=1).drop(columns = ['trial_file', 'ave_cv_loss', 'is_min']).drop_duplicates()

# %%

# %%

# %%
mask = is_min['method'] == 'rf'
rf_min = is_min.loc[mask]
rf_min.dropna(axis=1).drop(columns = ['trial_file']).drop_duplicates()


# if there are mulitple values (for numerics) get the range and use the mid point.
best_few = rf_min.groupby(
    ['method', 'input_data']
               ).agg(
    min_max_depth = ('current_max_depth', np.min),
    max_max_depth = ('current_max_depth', np.max),
    min_min_samples_leaf = ('current_min_samples_leaf', np.min),
    max_min_samples_leaf = ('current_min_samples_leaf', np.max)
    ).reset_index()

best_few['midpoint_max_depth'] = ((best_few['max_max_depth'] + best_few['min_max_depth'])/2)
best_few['midpoint_min_samples_leaf'] = round(((best_few['max_min_samples_leaf'] + best_few['min_min_samples_leaf'])/2))

best_few['midpoint_max_depth'] = round(best_few['midpoint_max_depth'])
best_few['midpoint_min_samples_leaf'] = round(best_few['midpoint_min_samples_leaf'])


best_few.loc[:, ['method', 'input_data', 'midpoint_max_depth', 'midpoint_min_samples_leaf']]

# %%

# %%

# %%

# %%
mask = is_min['method'] == 'rnr'
rnr_min = is_min.loc[mask]
rnr_min = rnr_min.dropna(axis=1).drop(columns = ['trial_file']).drop_duplicates()

# if there are mulitple values (for numerics) get the range and use the mid point.
best_few = rnr_min.groupby(
    ['method', 'input_data', 'current_weight_metric']
               ).agg(
    min_radius = ('current_radius', np.min),
    max_radius = ('current_radius', np.max)
    ).reset_index()

best_few['midpoint_radius'] = ((best_few['max_radius'] + best_few['min_radius'])/2)
best_few = rnr_min.merge(best_few)
best_few['error'] = best_few['current_radius'] - best_few['midpoint_radius'] 

closest_to_best = best_few.groupby(
    ['method', 'input_data', 'current_weight_metric']
    ).agg(error = ('error', np.min)
    ).reset_index()

closest_to_best = closest_to_best.merge(best_few, how = 'inner')
closest_to_best

closest_to_best.drop(columns = [e for e in ['is_min', 'min_radius', 'max_radius'] if e in list(closest_to_best)]                    )




# %%

# %%

# %%

# %%
mask = is_min['method'] == 'svrl'
svrl_min = is_min.loc[mask]
svrl_min.dropna(axis=1).drop(columns = ['trial_file']).drop_duplicates()

# %%

# %%
# df_summary

# px.scatter_3d(
#     x = df_summary['method'],
#     y = df_summary['input_data'],
#     z = df_summary['ave_cv_loss'],
#     color = df_summary['ave_cv_loss']
#                    )

# %%
# fig = px.scatter(
#     x = df_summary['method'],
#     y = df_summary['ave_cv_loss'],
#     color = df_summary['ave_cv_loss'],
#     facet_col=df_summary['input_data'],
#                    )
# fig.write_html('ML_Ave_Loss_First_Look.html')

# %%
