# -*- coding: utf-8 -*-
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

# %%

# %%
import numpy as np # for np.nan
import pandas as pd
pd.set_option('display.max_columns', None)
import os   # for write_log, delete_logs
import glob # for delete_all_logs
from datetime import date, timedelta

import json # for saving a dict to txt with json.dumps

import matplotlib as mpl
import matplotlib.pyplot as plt

remakeDaymet = False
# remakeDaymet = True
findBestWeatherModels = False#True 
    # takes ~11 minutes for full daymet set (with yearsite) 
    # only 0:00:46.512530 without.
# findBestWeatherModels = True

# %% [markdown]
# # Setup

# %% [markdown]
# ## Custom Functions

# %% code_folding=[]
def write_log(pathString= '../data/external/logfile.txt', message = 'hello'):
    # fileName = pathString.split(sep = '/')[-1]
    # mk log file if it doesn't exist
#     print(os.path.isfile(pathString))
    if os.path.isfile(pathString) == False:
        with open(pathString, 'wb') as textFile: 
            # wb allows for unicode 
            # only needed because there's a non-standard character in one of the column names
            # https://www.kite.com/python/answers/how-to-write-unicode-text-to-a-text-file-in-python
            textFile.write(''.encode("utf8"))
    else:    
        with open(pathString, "ab") as textFile:
            textFile.write(message)
            textFile.write('\n'.encode("utf8"))


# %% code_folding=[0]
def delete_logs(pathString = '../data/external/'):
    logs = glob.glob(pathString+'log*.txt') 
    for log in logs:
        os.remove(log)


# %% code_folding=[]
def update_col_names(df, nameDict):
    changes = []

    keysNotInDf = [key for key in nameDict.keys() if key not in list(df)]
    for key in keysNotInDf:
        for colName in [colName for colName in list(df) if colName in nameDict[key]]:
            df= df.rename(columns={colName: key})
            changes = changes + [(colName+' -> '+ key)]

    issues = [colName for colName in list(df) if colName not in nameDict.keys()]

    return([df, changes, issues])


# %% code_folding=[0]

# %% code_folding=[]
def add_SiteYear(df):
    df['SiteYear'] = df['ExperimentCode'].astype(str) + '_' + df['Year'].astype(str)
    return(df)


# %% code_folding=[]
# metadata, 'Metadata', colGroupings
def check_col_types(data, grouping, colGroupings
):
#     data = metadata
#     grouping = 'Metadata'
#     colGroupings = pd.read_csv('../data/external/UpdateColGroupings.csv')

    considerCols = colGroupings.loc[colGroupings[grouping].notna(), (grouping, grouping+'Type')]
    considerCols= considerCols.reset_index().drop(columns = 'index')
    changeMessages = []
    issueMessages  = []

    for i in range(considerCols.shape[0]):
        considerCol = considerCols.loc[i, grouping]
        considerColType = considerCols.loc[i, grouping+'Type']
        # print(i, considerCol, considerColType)
        
        if considerCol in list(data):
            # handle nans, datetime
            if not pd.isna(considerColType):
                try:
                    if considerColType == 'datetime':
                        data[considerCol]= pd.to_datetime(data[considerCol])
                    else:
                        data[considerCol]= data[considerCol].astype(considerColType)
                    changeMessage = (considerCol+' -> type '+considerColType+'')
                    changeMessages = changeMessages + [changeMessage]

                except (ValueError, TypeError):
                    data[considerCol]= data[considerCol].astype('string')
                    issueMessage = ('Returned as string. '+considerCol+' could not be converted to '+considerColType)
                    issueMessages = issueMessages + [issueMessage]

            else: # i.e. if type == na
                data[considerCol]= data[considerCol].astype('string')
                issueMessage = (considerCol+' has no type specified (na). Returned as string.')
                issueMessages = issueMessages + [issueMessage]

    return(data, changeMessages, issueMessages)


# 'metadata2018', colGroupings
def check_col_types_all_groupings(
    dfNameStr, 
    colGroupings
):
    reTypedColumns = {
        'Metadata': [], 
        'Soil': [],
        'Phenotype': [],
        'Genotype': [],
        'Management': [],
        'Weather': []
    }

    for grouping in reTypedColumns.keys(): #['Metadata', 'Soil', 'Phenotype', 'Genotype', 'Management', 'Weather']:
    #     print(grouping)

        outputList = check_col_types(data = eval(dfNameStr), grouping = grouping, colGroupings = colGroupings)
    #     updatedDf = outputList[0]

        reTypedColumns[grouping] = outputList[0]

        if outputList[1] != []:
            write_log(pathString= '../data/external/'+'log_'+dfNameStr+'_changes.txt', 
                      message = (grouping+' ========').encode())
            for change in outputList[1]:
                write_log(pathString= '../data/external/'+'log_'+dfNameStr+'_changes.txt', 
                          message = change.encode())

        if outputList[2] != []:
            write_log(pathString= '../data/external/'+'log_'+dfNameStr+'_issues.txt', 
                      message = (grouping+' ========').encode())
            for issue in outputList[2]:
                write_log(pathString= '../data/external/'+'log_'+dfNameStr+'_issues.txt', 
                          message = issue.encode())

    return(reTypedColumns)


def check_col_types_all_groupings_across_tables(
    metadataDfNameStr, # = 'metadata2018',
    soilDfNameStr, # = 'soil2018',
    weatherDfNameStr, # = 'weather2018',
    managementDfNameStr, # = 'agro2018',
    phenotypeDfNameStr, # = 'pheno2018',
    genotypeDfNameStr, # = '',
    colGroupings # = colGroupings
):
    metadataDict = []
    soilDict = []
    phenoDict = []
    genoDict = []
    agroDict = []
    weatherDict = []

    if metadataDfNameStr != '':
        metadataDict = check_col_types_all_groupings(dfNameStr = metadataDfNameStr, 
                                                     colGroupings = colGroupings)
    if soilDfNameStr != '':
        soilDict = check_col_types_all_groupings(dfNameStr = soilDfNameStr, 
                                                 colGroupings = colGroupings)
    if phenotypeDfNameStr != '':
        phenoDict = check_col_types_all_groupings(dfNameStr = phenotypeDfNameStr, 
                                                  colGroupings = colGroupings)
    if genotypeDfNameStr != '':
        genoDict = check_col_types_all_groupings(dfNameStr = genotypeDfNameStr, 
                                                 colGroupings = colGroupings)
    if managementDfNameStr != '':
        agroDict = check_col_types_all_groupings(dfNameStr = managementDfNameStr , 
                                                 colGroupings = colGroupings)
    if weatherDfNameStr != '':
        weatherDict = check_col_types_all_groupings(dfNameStr = weatherDfNameStr, 
                                                    colGroupings = colGroupings)

    outputList = [entry for entry in [metadataDict,
                                      soilDict,
                                      phenoDict,
                                      genoDict, 
                                      agroDict, 
                                      weatherDict] if entry != {}]

    return(outputList)

def combine_dfDicts(
    key, # = 'Metadata', 
    dfDictList, # = [], #[metadataDict, soilDict, phenoDict, agroDict, weatherDict]
    logfileName, # = 'combineMetadata',
    colGroupings #= colGroupings
):

    desiredCols = list(colGroupings[key].dropna())
    accumulator = pd.DataFrame()
    
    for dfDict in dfDictList:
        if dfDict != []:
            if accumulator.shape == (0,0):
                accumulator = dfDict[key].loc[:, [col for col in list(dfDict[key]) if col in desiredCols]].drop_duplicates()
            else:
                temp = dfDict[key].loc[:, [col for col in list(dfDict[key]) if col in desiredCols]].drop_duplicates()
                
                # Only proceed if if there are more columns shared than just identifiers.
                if [entry for entry in list(temp) if entry not in list(colGroupings['Keys'].dropna())] != []:

                    # find mismatched columns
                    downgradeToStr = [col for col in list(temp) if col in list(accumulator)]    
#                     downgradeToStr = [col for col in downgradeToStr if type(temp[col][0]) != type(accumulator[col][0])]
                    # This is not done as 
                    # downgradeToStr = [col for col in downgradeToStr if temp[col].dtype != accumulator[col].dtype]
                    # because doing so causes TypeError: Cannot interpret 'StringDtype' as a data type
                    # see also https://stackoverflow.com/questions/65079545/compare-dataframe-columns-typeerror-cannot-interpret-stringdtype-as-a-data-t

                    for col in downgradeToStr:
                        accumulator[col] = accumulator[col].astype(str)
                        temp[col] = temp[col].astype(str)

                        logMessage = "Set to string:"+col
                        write_log(pathString= '../data/external/'+'log_'+logfileName+'_changes.txt', 
                                  message = logMessage.encode())

                    # merge the now compatable dfs.
                    accumulator= accumulator.merge(temp, how = 'outer').drop_duplicates()
    return(accumulator)



# combine_dfDicts(key, dfDictList, logfileName, colGroupings) -> accumulator
#                   ^        ^
#      e.g. 'metadata'       |_______________________________
#                                                           \  
# check_col_types_all_groupings_across_tables(              |
# metadataDfNameStr, ... weatherDfNameStr, colGroupings) -> [metadataDict ... weatherDict] if entry != {}]
#                                 |                          ^           
#                                 v                          |    
# check_col_types_all_groupings(dfNameStr, colGroupings) -> {'Metadata': [], 'Soil': [], 'Phenotype': [], 'Genotype': [], 'Management': [], 'Weather': []}
#                                 |                                      ^
#                                 v                                      |
#               check_col_types(data, grouping, colGroupings) -> return(data, changeMessages, issueMessages)


# %%

# %% [markdown]
# ## Remove output files

# %%
delete_logs(pathString = '../data/external/')

# # rm database files
for filePath in glob.glob('../data/interim/*.db'):
    os.remove(filePath)

# %% [markdown]
# # Prepare and write minimally altered G2F database

# %% [markdown]
# ## Load Data

# %%

# %%
metadata2019 = pd.read_csv('../data/raw/GenomesToFields_data_2019/z._2019_supplemental_info/g2f_2019_field_metadata.csv', 
                           encoding = "ISO-8859-1", low_memory=False)

metadata2018 = pd.read_csv('../data/raw/GenomesToFields_G2F_Data_2018/e._2018_supplemental_info/g2f_2018_field_metadata.csv',  
                           encoding = "ISO-8859-1", low_memory=False)

metadata2017 = pd.read_csv('../data/raw/G2F_Planting_Season_2017_v1/z._2017_supplemental_info/g2f_2017_field_metadata.csv'
                           , encoding = "ISO-8859-1", low_memory=False)

metadata2016 = pd.read_csv('../data/raw/G2F_Planting_Season_2016_v2/z._2016_supplemental_info/g2f_2016_field_metadata.csv'
                           , encoding = "ISO-8859-1", low_memory=False)

print('Note! Many management factors are recorded in 2015!')
metadata2015 = pd.read_csv('../data/raw/G2F_Planting_Season_2015_v2/z._2015_supplemental_info/g2f_2015_field_metadata.csv'
                           , encoding = "ISO-8859-1", low_memory=False)


metadata2014 = pd.read_csv('../data/raw/G2F_Planting_Season_2014_v4/z._2014_supplemental_info/g2f_2014_field_characteristics.csv'
                           , encoding = "ISO-8859-1", low_memory=False)


soil2019 = pd.read_csv('../data/raw/GenomesToFields_data_2019/c._2019_soil_data/g2f_2019_soil_data.csv'
                           , encoding = "ISO-8859-1", low_memory=False)

soil2018 = pd.read_csv('../data/raw/GenomesToFields_G2F_Data_2018/c._2018_soil_data/g2f_2018_soil_data.csv'
                       , encoding = "ISO-8859-1", low_memory=False)


soil2017 = pd.read_csv('../data/raw/G2F_Planting_Season_2017_v1/c._2017_soil_data/g2f_2017_soil_data_clean.csv'
                       , encoding = "ISO-8859-1", low_memory=False)


soil2016 = pd.read_csv('../data/raw/G2F_Planting_Season_2016_v2/c._2016_soil_data/g2f_2016_soil_data_clean.csv'
                       , encoding = "ISO-8859-1", low_memory=False)


soil2015 = pd.read_csv('../data/raw/G2F_Planting_Season_2015_v2/d._2015_soil_data/g2f_2015_soil_data.csv'
                       , encoding = "ISO-8859-1", low_memory=False)


# no soil 2014, some info in metadata

pheno2019 = pd.read_csv('../data/raw/GenomesToFields_data_2019/a._2019_phenotypic_data/g2f_2019_phenotypic_clean_data.csv'
                           , encoding = "ISO-8859-1", low_memory=False)

pheno2018 = pd.read_csv('../data/raw/GenomesToFields_G2F_Data_2018/a._2018_hybrid_phenotypic_data/g2f_2018_hybrid_data_clean.csv'
                        , encoding = "ISO-8859-1", low_memory=False)


pheno2017 = pd.read_csv('../data/raw/G2F_Planting_Season_2017_v1/a._2017_hybrid_phenotypic_data/g2f_2017_hybrid_data_clean.csv'
                        , encoding = "ISO-8859-1", low_memory=False)


pheno2016 = pd.read_csv('../data/raw/G2F_Planting_Season_2016_v2/a._2016_hybrid_phenotypic_data/g2f_2016_hybrid_data_clean.csv'
                        , encoding = "ISO-8859-1", low_memory=False)


pheno2015 = pd.read_csv('../data/raw/G2F_Planting_Season_2015_v2/a._2015_hybrid_phenotypic_data/g2f_2015_hybrid_data_clean.csv'
                        , encoding = "ISO-8859-1", low_memory=False)


pheno2014 = pd.read_csv('../data/raw/G2F_Planting_Season_2014_v4/a._2014_hybrid_phenotypic_data/g2f_2014_hybrid_data_clean.csv'
                        , encoding = "ISO-8859-1", low_memory=False)


# Placeholder for GENETICS data ----
import re
newDecoderNames={
         'Sample Names': 'Sample',
'Inbred Genotype Names': 'InbredGenotype',
               'Family': 'Family'
}

geno2018= pd.read_table(
    '../data/raw/GenomesToFields_G2F_Data_2018/d._2018_genotypic_data/G2F_PHG_minreads1_Mo44_PHW65_MoG_assemblies_14112019_filtered_plusParents_sampleDecoder.txt',
    delimiter= ' ', 
    header=None, 
    names= ['Sample Names', 'Inbred Genotype Names'])#.sort_values('Inbred Genotype Names')

# Add to enable join to phenotype data
def clip_2018_family(i):
    # for i in range(geno2018.shape[0]):
    if(re.search('_', geno2018['Inbred Genotype Names'][i]) == None):
        newName = str(geno2018['Inbred Genotype Names'].str.split("_")[i][0])
    else:
        newName = str(geno2018['Inbred Genotype Names'].str.split("_")[i][0:1][0]
                     )+'_'+str(geno2018['Inbred Genotype Names'].str.split("_")[i][1:2][0])
    return(newName)
    
geno2018['Family'] = [clip_2018_family(i) for i in range(len(geno2018[['Inbred Genotype Names']]))]

# Data is in AGP format saved as a h5
geno2017= pd.read_excel('../data/raw/G2F_Planting_Season_2017_v1/d._2017_genotypic_data/g2f_2017_gbs_hybrid_codes.xlsx')
geno2017['Year']= '2017'





agro2019 = pd.read_csv('../data/raw/GenomesToFields_data_2019/z._2019_supplemental_info/g2f_2019_agronomic_information.csv'
                           , encoding = "ISO-8859-1", low_memory=False)


agro2018 = pd.read_csv('../data/raw/GenomesToFields_G2F_Data_2018/e._2018_supplemental_info/g2f_2018_agronomic information.csv'
                       , encoding = "ISO-8859-1", low_memory=False)


agro2017 = pd.read_csv('../data/raw/G2F_Planting_Season_2017_v1/z._2017_supplemental_info/g2f_2017_agronomic information.csv'
                       , encoding = "ISO-8859-1", low_memory=False)


agro2016 = pd.read_csv('../data/raw/G2F_Planting_Season_2016_v2/z._2016_supplemental_info/g2f_2016_agronomic_information.csv'
                       , encoding = "ISO-8859-1", low_memory=False)


# There is data to be had but it's not formatted in a machine friendly way.
# I've reformatted it to be easy to read in.
agro2015 = pd.read_csv('../data/raw/Manual/g2f_2015_agronomic information.csv'
                       , encoding = "ISO-8859-1", low_memory=False)


# no agro 2014, some info in metadata
weather2019 = pd.read_csv('../data/raw/GenomesToFields_data_2019/b._2019_weather_data/2019_weather_cleaned.csv'
                           , encoding = "ISO-8859-1", low_memory=False)


weather2018 = pd.read_csv('../data/raw/GenomesToFields_G2F_Data_2018/b._2018_weather_data/g2f_2018_weather_clean.csv'
                          , encoding = "ISO-8859-1", low_memory=False)


weather2017 = pd.read_csv('../data/raw/G2F_Planting_Season_2017_v1/b._2017_weather_data/g2f_2017_weather_data.csv'
                          , encoding = "ISO-8859-1", low_memory=False)


weather2016 = pd.read_csv('../data/raw/G2F_Planting_Season_2016_v2/b._2016_weather_data/g2f_2016_weather_data.csv'
                          , encoding = "ISO-8859-1", low_memory=False)


weather2015 = pd.read_csv('../data/raw/G2F_Planting_Season_2015_v2/b._2015_weather_data/g2f_2015_weather.csv'
                          , encoding = "ISO-8859-1", low_memory=False)

weather2014 = pd.read_csv('../data/raw/G2F_Planting_Season_2014_v4/b._2014_weather_data/g2f_2014_weather.csv'
                          , encoding = "ISO-8859-1", low_memory=False)


# %% [markdown]
# # Clean Data

# %% [markdown] code_folding=[4]
# ## Functions

# %% code_folding=[4]
def rename_columns(
    dataIn,
    colNamePath='../data/external/UpdateColNames.csv',
    logfileName=''
):
    # Make col name update dict ===================================================
    temp = pd.read_csv(colNamePath, low_memory=False) # <--- set because Rainfall is coded with numerics and string based na
    temp = temp.loc[:, (['Standardized']+[entry for entry in list(temp) if entry.startswith('Alias')])]

    # Turn slice of df into one to many dict
    renamingDict = {}
    for i in temp.index:
        tempSlice = list(temp.loc[i, :].dropna())
        renamingDict[tempSlice[0]] = tempSlice[1:]

    # Update Column Names =========================================================
    outputList = update_col_names(
        df=dataIn,
        nameDict= renamingDict)

    updatedDf = outputList[0]
    dfChanges = outputList[1]
    dfIssues = outputList[2]

    # Write logs ==================================================================
    for change in dfChanges:
        write_log(pathString='../data/external/log_'+logfileName+'_changes.txt',
                  message=str(change).encode("utf8"))

    for issue in dfIssues:
        write_log(pathString='../data/external/log_'+logfileName+'_issues.txt',
                  message=('Not defined: "'+str(issue)+'"').encode("utf8"))

    return(updatedDf)


# %% code_folding=[]
def sort_cols_in_dfs(inputData = [metadata2015,
soil2015,
pheno2015,
agro2015,
weather2015],

colGroupings = pd.read_csv('../data/external/UpdateColGroupings.csv')
):

    # inputData = [metadata2015,
    # soil2015,
    # pheno2015,
    # agro2015,
    # weather2015]

    # colGroupings = pd.read_csv('../data/external/UpdateColGroupings.csv')

    keyCols= list((colGroupings.Keys).dropna())
    groupings= [col for col in list(colGroupings) if ((col.endswith('Type') == False) & (col != 'Keys'))]

    groupingAccumulators = {} #groupings.copy()

    # i = 0
    # j = 0

    for j in range(len(groupings)):
        #keepTheseCols = keyCols+list(colGroupings[groupings[j]].dropna())
        keepTheseCols = list(colGroupings[groupings[j]].dropna())
        accumulator = pd.DataFrame()
        for i in range(len(inputData)):
            data = inputData[i]
            data = data.loc[:, [entry for entry in list(data) if entry in keepTheseCols]].drop_duplicates()

            # this will hopefully prevent dropping data that is text in datetime or numeric cols.
            # And then we don't have to worry about merging by like types here!
            data = data.astype(str)

            if ((accumulator.shape[0] == 0) | (accumulator.shape[1] == 0) ):
                accumulator = data
            elif ((data.shape[0] > 0) & (data.shape[1] > 0)) :
#                 print(list(accumulator))
#                 print(list(data))
#                 print('\n')
                accumulator = accumulator.merge(data, how = 'outer')

            currentGrouping = groupings[j] # string about to be replace
            #     groupingAccumulators[j] = {currentGrouping : accumulator}
            #     groupingAccumulators = groupingAccumulators + {currentGrouping : accumulator}

            groupingAccumulators.update({currentGrouping : accumulator})

    return(groupingAccumulators)



# colGroupings = pd.read_csv('../data/external/UpdateColGroupings.csv')
# out2018 = sort_cols_in_dfs(inputData=[
#     metadata2018,
#     soil2018,
#     pheno2018,
#     agro2018,
#     weather2018],
#     colGroupings=colGroupings)


# %%
def rename_col_entries(
        df, #=agro2018,
        colIn, # ='Application',
        newToOldDict): #=newApplicationNames

    temp = pd.DataFrame(df[colIn]).copy()
    for key in list(newToOldDict):
        temp.loc[temp[colIn].isin(newToOldDict[key])] = key

    # Check for names that were not accounted for.
    allNames = list(temp[colIn].unique())
    newNames = (list(newToOldDict.keys()))
    missingNames = [name for name in allNames if ((name in newNames) == False)]

    if len(missingNames) > 0:
        print('The following keys are missing from the input dictionary!')
        print('')
        print(missingNames)
        print('')
        for name in missingNames:
            print(name)

    return(temp)



# %% code_folding=[]
def convertable_to_float(value = '1pt'):
    try:
        float(value)
        return(True)
    except:
        return(False)


# %%
def convertable_to_datetime(value = '2000'):
    try:
        pd.to_datetime(value)
        return(True)
    except:
        return(False)

def convertable_to_datetime_not_NaT(value = '2000'):
    try:
        if pd.isnull(pd.to_datetime(value)):
            return(False)
        else:
            return(True)
    except:
        return(False)



# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %%

# %% [markdown]
# ## Data Objects

# %%
[entry for entry in [] if len(entry.split(sep = ' ')) == 1]

# %% code_folding=[3, 75, 88, 123, 135, 296, 466, 487, 491]
# For all datasets except geno ----

# ExperimentCode should be in this list
expectedExperimentCodes = ["ARH1",	"ARH2",	"DEH1",	"GAH1",	"GAH2",	"IAH1",	"IAH2",	"IAH3",	"IAH4",	"ILH1",	"INH1",	"KSH1",	"MIH1",	"MNH1",	"MOH1",	"NCH1",	"NEH2",	"NYH1",	"NYH2",	"NYH3",	"OHH1",	"ONH2",	"SCH1",	"TXH1-Dry",	"TXH1-Early",	"TXH1-Late",	"TXH2",	"WIH1",	"WIH2", 'MOH1-rep1', 'TXH1', 'MOH1-rep2', 'ONH1', 'KSH2', 'KSH3',
                           'COH1', 'NEH3', 'NEH4', 'ILH2', 'NEH1', 'NEH3-Irrigated', 'NEH4-NonIrrigated', 'NYH4', 'AZH1',
                           'AZI1',
                           'AZI2',
                           'PAI1',
                           'PAI2',
                           'DEI1',
                           'GAI1',
                           'NCI1',
                           'NYI2',
                           'TXI1',
                           'IAI1',
                           'IAI2',
                           'IAI3',
                           'IAI4',
                           'ILI1',
                           'INI1',
                           'KSI1',
                           'MNI1',
                           'MOH2',
                           'MOI1',
                           'MOI2',
                           'NYI1',
                           'SDH1',
                           'SDI1',
                           'TXI2',
                           'TXI3',
                           'WII1',
                           'WII2',
                           'IA(?)2',
                           'IA(?)3',
                           'IA(H4)',
                           'MN(?)1',
                           'NY?',

                           'GAI2', 'IAH1 IAI1', 'ILH1 ILI1', 'INH1 INI1', 'MNI2', 'MOH1 MOI1', 'MOH2 MOI2 MOI3', 'NEH1 NEI1', 'NYH1 NYI1', 'TXH1  TXI1  TXI2', 'TXH2  TXI3', 'WIH1 WII1', 'G2FWI-HYB', 'G2FDE1', 'nan', 'GXE_inb_IA1', 'GXE_inb_IA2', 'G2FIA3', 'G2FIL1', 'G2FIN1', 'G2FMN2', 'GXE_inb_MO1', 'GXE_inb_BO2', 'GXE_inb_MO3', 'NC1', 'G2FNE1', 'G2FNY1', 'GxE_inb_PA1', 'G2F_IN_TX1', 'G2FWI1', 'G2FWI2', 'IAH1a', 'IAH1b', 'IAH1c',
                           'AZI1  AZI2', 'ILH1  ILI1  ILH2', 'INH1  INI1', 'KSH1  KSI1', 'MOH1  MOI1  MOH2  MOI2', 'NEH1  NEH4', 'NYH1  NYI1', 'NYH3  NYI2', 'PAI1  PAI2', 'WIH1  WII1', 'WIH2  WII2'
                           ]


replaceExperimentCodes = {
    'MOH1- rep 1': 'MOH1-rep1',
    'MOH1- rep 2': 'MOH1-rep2',
    'TXH1- Early': 'TXH1-Early',
    'TXH1- Late': 'TXH1-Late',
    'TXH1- Dry': 'TXH1-Dry',
    'NEH3 (IRRIGATED)': 'NEH3-Irrigated',
    'NEH4 (NON IRRIGATED)': 'NEH4-NonIrrigated'

}

# 'AZI1  AZI2': 'AZ1',
#  'ILH1  ILI1  ILH2': '',
#  'INH1  INI1': '',
#  'KSH1  KSI1': '',
#  'MOH1  MOI1  MOH2  MOI2': '',
#  'NEH1  NEH4': '',
#  'NYH1  NYI1': '',
#  'NYH3  NYI2': '',
#  'PAI1  PAI2': '',
#  'TXH1  TXI1  TXI2': '',
#  'WIH1  WII1': '',
#  'WIH2  WII2': '',


# 'GAI2', 'IAH1 IAI1', 'ILH1 ILI1', 'INH1 INI1', 'MNI2', 'MOH1 MOI1', 'MOH2 MOI2 MOI3', 'NEH1 NEI1', 'NYH1 NYI1', 'TXH1  TXI1  TXI2', 'TXH2  TXI3', 'WIH1 WII1', 'G2FWI-HYB', 'G2FDE1', 'nan', 'GXE_inb_IA1', 'GXE_inb_IA2', 'G2FIA3', 'G2FIL1', 'G2FIN1', 'G2FMN2', 'GXE_inb_MO1', 'GXE_inb_BO2', 'GXE_inb_MO3', 'NC1', 'G2FNE1', 'G2FNY1', 'GxE_inb_PA1', 'G2F_IN_TX1', 'G2FWI1', 'G2FWI2', 'IAH1a', 'IAH1b', 'IAH1c'

# For specific datasets ----

# For

# For management ====
expectedManagementCols = [
    'ExperimentCode',
    'Application',
    'Product',
    'Date',
    'QuantityPerAcre',
    'ApplicationUnit',
    'Year',
    'Plot']


# management.Application ====

newApplicationNames = {
    'irrigation': ['irrigation',
                   'Furrow Irrigation'],

    'fertilizer': ['Post-fertilization',
                   'Post-fertilization for DG2F',
                   'Pre-emergent Herbicide',
                   'Pre-plant fertilization',
                   'broadcast fertilizer',
                   'fertigation',
                   'fertilize',
                   'sidedress Nitrogen application',
                   'Fertlizer',
                  'soil remediant'],

    'herbicide': ['Pre-emergent Herbicide',
                  'Herbicide pre-emergence application',
                  'Post-Herbicide',
                  'pre-plant herbicide',
                  'post-plant herbicide',
                  'Post-Plant Herbicide',
                  'adjuvant'],

    'insecticide': ['insecticide'],
    'fungicide': ['Fungicide '],

    'ignored': ['FOR G2LA', 'FOR G2FE', 'For DG2F']
}



# newApplicationNames

# management.ApplicationUnit ====

expectedApplicationUnits = ['lbs/Acre', 'oz/Acre',
                            'in/Acre', 'gal/Acre', 'units/Acre', 'ton/Acre', 'GPA']


replaceApplicationUnits = {'lbs': 'lbs/Acre',
                           'oz': 'oz/Acre',
                           'gallons/acre': 'gal/Acre', 
                           'Gallons/acre': 'gal/Acre'
                           }


# management.Product ====
newIrrigationNames = {
    'Water': ['water',
              'Water',
              'preirrigation',
              'Overhead; recorded by weather station',
              'irrigate',
              'irrigated',
              'RECORDED BY WEATHER STATION!',
              'H2O']
}
 
newFertilizerNames = {
    'Manure Poultry': ['Poultry manure'],
    # npk like
    'nkpLike21-0-0-24': ['21-0-0-24'],
    'nkpLike28-0-0-0.5': ['28-0-0-0.5'],
    'nkpLike28-0-0-5': ['28-0-0-5'],
    'Starter nkpLike50-45-10-10-2': ['Starter 50-45-10-10-2'],

    'npkLike24-0-0-3': ['24-0-0-3'],
    'npkLike20-10-0-1': ['20-10-0-1s'],
    'npkLike27-0-0-6': ['27-0-0-6s'],
    'npkLike30-26-0-6': ['npkLike30-26-0-6', 'Starter2x2', 'Liquid Starter 2x2 planter', 'Planter starter 2x2'],
    # per 2019
    # Location	Application_or_treatment	Product_or_nutrient_applied	Date_of_application	Quantity_per_acre	Application_unit
    # OHH1	fertilize	Planter starter 2x2	Friday, June 7, 2019	30-26-0-6S	lbs/Acre
    # "Planter starter 2x2" may == "30-26-0-6S"
    
    'npkLike6.7-13-33.5-4.4': ['6.7-13-33.5 4.4S dry fertilizer broadcast incorp', '6.7-13-33.5 with 4.4 sulfur'],

    # Readily Converted NPK ----------------------------------------------------------------------------
    'npk0-0-32': ['0-0-32'],
    'npk0-0-60': ['0-0-60',
                  '0-0-60 (NPK)',
                  '0-0-60, preplant incorporated',
                  '0-0-60 / pre plant incorporate'],
    'npk4-10-24': ['4-10-24'],
    'npk5.5-26-30': ['5.5-26-30 fertilizer'],
    'npk7-0-40': ['npk7-0-40'],
    'npk7.5-30-30': ['npk7.5-30-30'],
    'npk7-33-21': ['7-33-21 broadcast prior to planting',
                   '7-33-21 broadcasted',
                   '7-33-21�', 
                   '7-33-21Ê' ],
    'npk8-20-30': ['8/20/1930'],
    'npk8.9-42.3-48.8': ['npk8.9-42.3-48.8'],
    'npk9-24-27': ['9/24/2027'],
    'npk9.3-31.7-0': ['9.3-31.7-0'],
    'npk10-34-0': ['10-34-0'],
    'npk10-10-30': ['npk10-10-30', '10-10-30 N-P-K'],
    'npk11-37-0': ['11-37-0'],



    'npk11-52-0': ['11-52-0, preplant incorporated', '11-52-0 / pre plant incorporate'],
    'npk12.72-9.8-23.57': ['12.72-9.8-23.57'],
    'npk13-19-24': ['13-19-24',
                    '13-19-24�', 
                    '13-19-24Ê'],
    'npk14-4-29': ['14-4-29'],
    'npk14-0-0': ['140+0+0'],  # NOTE! Presumed typo. 140 -> 14
    'npk15-18-24': ['15-18-24'],
    'npk15-34-0': ['15-34-0'],
    'npk16-0-24': ['16-0-24',
                   'Applied 16-0-24'],
    'npk18-46-0': ['18-46-0'],
    'npk19-18-0': ['19-18-0� (liquid planter starter)', 
                   '19-18-0\xa0 (liquid planter starter)', 
                   '19-18-0Ê (liquid planter starter)'
                  ],
     
    'npk20-9-3': ['npk20-9-3'],
    'npk20-10-0': ['20-10-0 (NPK)'],
    'npk27-26-0': ['npk27-26-0'],
    'npk28-0-0': ['28-0-0', '28%'],

    'Manure npk31.5-4.3-16.3': ['Manure N31.5-P4.3-K16.3 from Digester'],
    'npk32-0-0': ['32-0-0'],
    'npk46-0-0': ['46-0-0'],

    'npk50-50-0': ['npk50-50-0'],


    'Starter 16-16-16': ['Starter 16-16-16'],
    'Starter 16.55-6.59-2.03': ['Starter Fertilizer 16.55-6.59-2.03'],
    'Starter2x2 16-16-16': ['Starter ferilizer 16-16-16 2x2'],
    'Starter 20-10-0': ['Starter fertilizer 20-10-0'],

    'npk10-0-30+12S': ['10-0-30-12%S'],  # Note extra Sulfer %
    # Note extra Molybdenum
    'Mono-Ammonium Phosphate npk11-52-0+Mo0.001': ['MAP 11-52-0'],
    'npk11-37-0+4Zn': ['11-37-0 + 4 Zn'],  # Note extra zinc

    # Nitrogen -----------------------------------------------------------------------------------------
    'N28% npk28-0-0': ['28% Nitrogen',
                       '28% Liquid Nitrogen',
                       '28% Nitrogen',
                       r'28% Nitrogen, Sidedress coulter injected',
                       '28% Liquid Nitrogen',
                      '28% N sidedress'],

    'NH3': ['NH3', 'Anhydrous Ammonia'],
    'NH3 N-serve': ['NH3 with N-serve'],
 

    'N': ['N',
          'Nitrogen',
          'Nitrogen sidedress',
          'Side dressed Nitrogen',
          'Nitrogen - SuperU',
          'nitrogen knifed',
          'nitrogen (date approximate, pre plant)', 
          'nitrogen (date approximate, post plant)'],


    'Urea46%': ['46% Urea'],
    'Super Urea': ['Super U',
                   'Super Urea'],

    'Urea': ['Urea', 'urea', 'Granual Urea', 'Granular Urea'],

    'UAN': ['UAN', 'UAN ', 'Nitrogen base (UAN)'],

    'UAN28% npk28-0-0': ['UAN 28%',
                         '28% UAN'],

    'UAN30% npk30-0-0': ['30% UAN',
                         'Side Dressed N with 30% UAN',
                         'UAN side dressed  30-0-0',
                         '30% UAN protected with nitrapyrin nitrification inhibitor (Nitrogen 42 GPA knifed between rows)',
                         'UAN 30%'],
    'UAN32% npk32-0-0': ['UAN; 32-0-0'
                         ],

    # Phosphorus ---------------------------------------------------------------------------------------
    'P': ['P2O5', 'Phosphorous', 'P2O'],
    'P+K': ['Phosphorus & Potassium (unknown form)'],

    # Potassium ----------------------------------------------------------------------------------------
    'Potash': ['Pot Ash', 'potash'],
    'Potash npk0-0-60': ['potash 0-0-60'],
    'K': ['K', 'K2O'],
    # Other --------------------------------------------------------------------------------------------
    'Agrotain': ['Agrotain'],
    # Boron ===========================================================================================
    'B10%': ['10% Boron', 'Boron 10%'],
    # Zinc ============================================================================================
    'Zn': ['Zinc', 'zinc', 'Zinc\xa0\xa0 (with planter fertilizer)'],
    'Zn planter fertilizer': ['Zinc�� (with planter fertilizer)', 'ZincÊÊ (with planter fertilizer)'],
    'Zn sulfate': ['zinc sulfate'],

    # Sulfer ==========================================================================================
    '24S': ['24S'],
    '24S': ['Applied 24S'],
    'S': ['S'],
    # Manganese =======================================================================================
    'KickStand Manganese 4% Xtra': ['KickStand� Manganese 4% Xtra', 'KickStand® Manganese 4% Xtra'],
    # Ammonium Sulfate ================================================================================
    'Ammonium sulfate': ['AMS 21-0-0-24',
                         'Am Sulfate',
                         'Ammonium sulfate',
                         'ammonium sulfate'
                         ],

    # Multiple Components =============================================================================
    'Orangeburg Mix': ['Orangeburg Mix'],
    # pH ==============================================================================================
    'Lime': ['Lime', 'lime', 'Ag Lime', 'Lime / pre plant incorporate']

}

newHerbicideNames = {
    # Aatrex -------------------------------------------------------------------------------------------
    'Aatrex': ['Aatrex',
               'Attrex',
               'post emergence- Aatrex'],

    # Atrazine -----------------------------------------------------------------------------------------
    'Atrazine': ['Atrazine',
                 'Atrazine ',
                 'Atrazine 4L',
                 'atrazine',
                 'Atrazine ',
                 'Applied Atrazine',
                 'Atrazine '],

    # Ammonium sulfate ---------------------------------------------------------------------------------
    'Ammonium sulfate': ['AMS',
                         'Am Sulfate',
                         'ammonium sulfate',
                         'AMS 21-0-0-24'],
    'Ammonium sulfate .5%': ['AMS@.5%'],
    # Accent Q -----------------------------------------------------------------------------------------
    'Accent Q': ['Accent',
                 'Accent Q',
                 'Accent Q (DuPont)'],

    # Acuron -------------------------------------------------------------------------------------------
    'Acuron': ['Accuron',
               'Acuron',
               'Acuron (S-Metalochlor, Atrazine, Mesotrione)',
               'Acuron herbicide'],
    # Banvel -------------------------------------------------------------------------------------------
    'Banvel': ['Banvel',
               'post emergece- Banvel'],
    # Bicep --------------------------------------------------------------------------------------------

    'Bicep': ['Bicep', 'bicep'],
    'Bicep Magnum': ['Bicep Magnum'],
    'Bicep II': ['Bicep II'],
    'Bicep II Magnum': ['Bicep II Magnum',
                        'pre-emergence -Bicep II Magnum'],

    'Bicep II Magnum Lite': ['Bicep Lite II magnum'],
    # Buctril ------------------------------------------------------------------------------------------
    'Buctril': ['Buctril',
                'Buctril 4EC'],

    # Callisto -----------------------------------------------------------------------------------------
    'Callisto': ['Calisto', 'Callisto'],
    'Callisto+Atrazine+AMS+COC': ['Calisto/Atrazine/AMS/COC'],

    # Crop Oil -----------------------------------------------------------------------------------------
    'Crop Oil': ['Crop Oil',
                 'crop oil'],

    # Dual ---------------------------------------------------------------------------------------------
    'Dual': ['Daul ',
             'Dual'],

    'Dual Magnum': ['Dual Magnum',
                    'Applied Dual Magnum'],

    'Dual II Magnum': ['Dual II - Magnum',
                       'Dual II Mag.',
                       'Dual II Magnum',
                       'Dual 2 Magnum'],

    'Degree Xtra': ['Degree Extra',
                    'Degree Xtra; Acetochlor + Atrazine'],
    # Harness ------------------------------------------------------------------------------------------
    'Harness': ['Harness'],
    'Harness Xtra': ['Harness Xtra'],
    'Harness Xtra 5.6L': ['Harness Xtra 5.6L; Acetachlor and Atrazine', 'Harness Xtra 5.6L;  Acetachlor and Atrazine'],
    # Liberty 280SL ------------------------------------------------------------------------------------
    'Liberty 280SL': ['Liberty 280SL'],
    # Medal II -----------------------------------------------------------------------------------------
    'Medal II EC': ['Medal II EC', 'Medal II (S- Metolachlor)'],
    # NOTE: [],Presumed typo
    'Medal II EC+Simazine+Explorer': ['Medal II, sirrazine, Explorer'],
    # Primextra ----------------------------------------------------------------------------------------
    'Primextra': ['Primextra'],
    'Primextra Magnum': ['Primextra Magnum'],
    # Prowl H2O ----------------------------------------------------------------------------------------
    'Prowl H2O': ['Prowl',
                  'Prowl H20 ',
                  'Prowl H2O'],

    # Roundup + Roundup PowerMAX ---------------------------------------------------------------------------------
    'Roundup': ['Roundup', 
                'Round up'],
    'Roundup PowerMAX': ['Roundup PowerMAX',
                         'Roundup PowerMax',
                         'Applied Round Up Pwrmax',
                         'Roundup Power Max'],
    'Roundup PowerMax II': ['Roundup PowerMax II'],
    # Simazine -----------------------------------------------------------------------------------------
    'Simazine': ['Simazine',
                 'Simizine'],
    'Simazine 4L': ['Simazine 4L'],

    # Lexar --------------------------------------------------------------------------------------------
    'Lexar': ['Lexar'],
    'Lexar EZ': ['Lexar EZ',
                 'Lexar E Z'],
    # Status -------------------------------------------------------------------------------------------
    'Status': ['Status',
               'status'],

    'Warrant': ['Warrant',
                'warrant'],
    # Only One Entry -----------------------------------------------------------------------------------
    '24D': ['24D', '24-D'],
    'AD-MAX 90': ['AD-MAX 90'],
    'AccuQuest WM': ['AccuQuest WM'],
    'Aremzon': ['Aremzon'],
    'Balance Flex herbicide': ['Balance Flex herbicide'],
    'Basagran': ['Basagran'],
    'Bicep Light II': ['Bicep Light II'],
    'Brawl': ['Brawl'],
    'Broadloom': ['Broadloom'],
    'Buccaneer Plus': ['Buccaneer Plus'],
    'Evik DF': ['Evik DF', 'Applied Evik DF'],
    'Explorer': ['Explorer'],  
    'Glyphosate': ['Glyphosate'],  
    'Evik': ['Evik'],
    'Integrity': ['Integrity'],
    'Gramoxone': ['Gramoxone'],
    'Guardsman': ['Guardsman (atrazine, metalochlor)'],
    'Impact': ['Impact', 'Applied Impact'],

    'Keystone': ['Keystone'],
    'Laudis': ['Laudis'],

    'Methylated Seed Oil 1%': ['MSO @1%'],
    'Me Too Lachlor II': ['Me Too Lachlor II'],
    'Nonionic Surfactant': ['NIS'],  # adjuvant
    'Option': ['Option'],
    'Outlook+Infantry': ['Outlook/Infantry'],
    'Permit': ['Permit'],
    'Princep': ['Princep', 'Pricep'],
    'Satellite HydroCap': ['Satellite HydroCap'],  # Pendimethalin 38.7%

    'Steadfast Q': ['Steadfast Q'],

    # has mixed components
    'Makaze+Glyphosate+Medal II EC+S-metolachlor+Eptam+S-ethyl dipropylthiocarbamate+Trifluralin HF+Trifluralin+Atrazine 4L+Atrazine': ['Makaze, Glyphosate (isopropylamine salt); Medal II EC, S-metolachlor; Eptam, S-ethyl dipropylthiocarbamate; Trifluralin HF, Trifluralin; Atrazine 4L, Atrazine'],
    'Cadet+Fluthiacet-methyl': ['Cadet; Fluthiacet-methyl'],
    'Coragen+Chlorantraniliprole': ['Coragen; Chlorantraniliprole'],
    'Prowl+Atrazine+Basagran': ['Prowl, Atrazine, Basagran'],
    'Impact+Atrazine': ['Impact and Atrazine'],
    'Instigate+Bicep': ['Instigate and Bicep'],
    'Glyphosate+2,4D+Dicamba': ['Glyphosato, 2,4D, Dicambo', 'Glyphosate, 2,4D, Dicambo'],
    'harness+atrazine': ['harness (acetechlor) + atrazine'],
    '2-4,D+Round Up': ['2-4,D, Round Up'],
    'Bicep II+RoundUp': ['Bicep II, Round Up'],
    'Laudis+Atrazine': ['Laudis, Atrazine'],
    'Counter+Radiant+Asana XL+Prevathon': ['Counter(terbufos), Radiant (spinetoram), Asana XL (Esfenvalerate), Prevathon (Chlorantraniliprole)'],
    'Status+Accent': ['Status + Accent (diflufenzopyr, dicamba, nicosulfuron)'],
    'Primextra+Callisto': ['Primextra+Callisto(s-metolachlor/benoxacor/atrazine+mesotrione)'],
    'Accent+Banvel': ['Accent and Banvel - Nicosulfuron and Dicamba'],
    'Callisto+AI Mesotrione+Dual 2 Magnum+AI S-metolachlor+Simazine+AI Simazine': ['Tank Mix (Callisto, AI Mesotrione) (Dual 2 Magnum, AI S-metolachlor) (Simazine, AI Simazine:2-chloro-4 6-bis(ethylamino)-s-triazine)'],
    'Lumax+Atrazine': ['Lumax + Atrazine'],
    'Lumax+glyphosate': ['Lumax + glyphosate'],
    'Converge Flexx+Converge 480': ['Converge Flexx; Converge 480'],
    'Callisto+Dual 2 Magnum+Simazine 4L': ['Tank Mix (Callisto; Dual 2 Magnum; Simazine 4L)'],
    'Roundup+Evik': ['Roundup and Evik hand sprayed as needed'],
    'Atrazine+DualIIMagnum': ['Atrazine 4L and Dual II Magnum']

}

newInsecticideNames = {
    'Counter 20G': ['Counter 20G',
                    'Counter 20g',
                    'Applied Counter 20G',
                    'Coutner 20G'],
    'Counter': ['Counter'],
    'Force 3G': ['Force 3G',
                 'Counter Force 3G',
                 'Counter Force 3G (AI Tefluthrin)',
                 'Force 3G (Tefluthrin)',
                 'Force 3G; Tefluthrin'],
    'Force 2G': ['Force 2G'],
    'Lorsban 4E': ['Lorsban 4E'],
    'Sevin XLR': ['Sevin XLR'],
    'Sniper': ['Sniper', 'Sniper LFRÊ'],
    # Liquid Fertilizer Ready #TODO how much fertilizer is being added through this?
    'Sniper LFR': ['Sniper LFR�', 'Snipper LFR']
}



newFungicideNames = {
    'Delaro 325 SC': ['Fungicide - Delaro 325 SC']
}

newMiscNames = {
    # ignore
    'nan': ['Disk',
            'Field Cultivator',
            'Hip and Rolled',
            'Planted Corn Test',
            'Planted Corn Filler Dyna-Grow D57VC51 RR',
            'Begin Planting Research',
            'nan']
}


newProductNames = {}
newProductNames.update(newIrrigationNames)
newProductNames.update(newFertilizerNames)
newProductNames.update(newHerbicideNames)
newProductNames.update(newInsecticideNames)
newProductNames.update(newFungicideNames)
newProductNames.update(newMiscNames)
# newProductNames

# %%

# %% [markdown]
# ## Act on Data

# %%
##### variable #####


expectedYear = ['2014']

##### ######## #####

# %% code_folding=[3, 6, 260, 778, 976]
def clean_managment(
    expectedYear,
    management
):
    
    # TLC for 2015 goes here
    if expectedYear == ['2015']:

        # drop herbicide/pesticide cols
        management = management.drop(columns = [entry for entry in list(management
                                     ) if entry in ['HerbicidePrePlant', 'HerbicidePostPlant', 'InsecticideType']])

        management.Product.drop_duplicates()

        management= management.loc[management.Product != 'UAN; 32-0-0', ] # this corresponds to AZI1 & AZI2. Both have info in the fertilizer cols so keeping this would double count. 


        #% #

        fertilizerSubset = management.loc[:, [
            'ExperimentCode',
            'FertilizerType', #
            'NApplied',       #
            'TotalN',
            'TotalP',
            'TotalK',

            'FertilizerDay1',
            'FertilizerDay2',
            'FertilizerDay3',
            'FertilizerDay4',
            'FertilizerDay5',
            'FertilizerDay6',
            'FertilizerDay7',
            'FertilizerDay8']]

        # fertilizerSubset.dropna()
        fertilizerSubset = fertilizerSubset.drop_duplicates()


        print('This col has no value')
        list(fertilizerSubset.NApplied.drop_duplicates())


        # fertilizerSubset.loc[(fertilizerSubset.NApplied != 'nan') | (fertilizerSubset.NApplied.notna())]

        # fertilizerSubset.dropna(subset = ['NApplied'])


        #% #

        # Deal with entries lacking FertilizerDay1 first

        temp = fertilizerSubset.loc[fertilizerSubset.FertilizerDay1 == 'nan'].copy()

        for col in list(temp):
            mask = temp[col] == 'nan'
            temp.loc[mask, col] = np.nan

        temp = temp.dropna(subset=list(temp.drop(columns='ExperimentCode')), how='all')

        for col in ['TotalN', 'TotalP', 'TotalK']:
            temp.loc[temp[col].isna(), col] = 0
            temp.loc[:, col] = temp.loc[:, col].astype(str)


        temp = temp.loc[:, ['ExperimentCode', 'TotalN', 'TotalP', 'TotalK']]
        temp = temp.rename(columns={'TotalN': 'N',
                                    'TotalP': 'P',
                                    'TotalK': 'K'})

        temp = temp.melt(id_vars=['ExperimentCode'], var_name="Product").rename(columns={'value': 'QuantityPerAcre'})
        temp['ApplicationUnit'] = 'lbs/Acre'

        fertilizerNoDate = temp

        # stack
        fertilizerSubset = fertilizerSubset.loc[fertilizerSubset.FertilizerDay1 != 'nan']

        output = pd.DataFrame()

        # the values which will be overwritten are: 'nan', 'Preplant', 'Planting', '', 'unknown'
        fertilizerDayColumns = ['FertilizerDay1', 'FertilizerDay2', 'FertilizerDay3', 'FertilizerDay4', 
                                'FertilizerDay5', 'FertilizerDay6', 'FertilizerDay7', 'FertilizerDay8']

        for column in fertilizerDayColumns:
            temp= fertilizerSubset.loc[fertilizerSubset[column] != '', ].copy()
            temp['Date']= temp[column]
            temp= temp.drop(columns = fertilizerDayColumns)
            if output.shape[0] == 0:
                    output= temp
            else:
                output= pd.concat([output, temp])

        output = output.loc[(output.Date.notna()) & (output.Date != 'nan'), ]


        # add in div for total
        output= output.merge(output.loc[:, ['ExperimentCode', 'Date']
                                       ].groupby('ExperimentCode'
                                       ).count(
                                       ).reset_index(
                                       ).rename(columns = {'Date':'DivBy'}
                                       ), how = 'outer')


        #% #

        searchString = '120lbs/acre in UAN applied prior to spring tillage.'
        output.loc[output.TotalN == searchString, 'TotalN'] = 0

        output = output.merge(pd.DataFrame().from_dict(
            {'ExperimentCode': ['IAI3'],
                'TotalN': ['120'],
                'Date': ['Preplant'],
                'DivBy': [1]}),
                     how = 'outer')


        # # Drop this experiment:
        mask= ((output.TotalN == '150 lbs/acre in UAN form applied with chemical ahead of tillage') & (
                output.TotalP == '18-60-120-15-1.5 (n-p-k-sulfur-zinc)') & (
                output.TotalK == '120lbs, fall applied'))
        output['Dropped'] = False
        output.loc[mask, 'Dropped'] = True

        droppedExp = output.loc[mask, ['ExperimentCode', 'Dropped']].drop_duplicates()
        output = output.loc[~mask, ].merge(droppedExp, how = 'outer')


        #% #

        # these can be converted to numeric?
        dfName = 'output'
        print('TotalN not converted into numeric!')
        quantitiesFound = list(eval(dfName).TotalN.drop_duplicates())
        print([entry for entry in quantitiesFound if convertable_to_float(value = entry) == False])

        print('TotalP not converted into numeric!')
        quantitiesFound = list(eval(dfName).TotalP.drop_duplicates())
        print([entry for entry in quantitiesFound if convertable_to_float(value = entry) == False])

        print('TotalK not converted into numeric!')
        quantitiesFound = list(eval(dfName).TotalK.drop_duplicates())
        print([entry for entry in quantitiesFound if convertable_to_float(value = entry) == False])


        #% #

        # convert to numeric, normalize by number of applications
        for col in ['TotalN', 'TotalP', 'TotalK']:
            output.loc[:, [col]] = output.loc[:, [col]].astype(float)

        mask = ((output.DivBy != 0) & (output.DivBy.notna()))
        for col in ['TotalN', 'TotalP', 'TotalK']:
            output.loc[mask, col] = output.loc[mask, col] / output.loc[mask, 'DivBy']  

        # drop treatments too complicated to parse
        for fertilizerType in [
            'urea (46-0-0); applied to south half (1/2 Acre, Hi-N side) of field only: actual applied was 140 lb urea to the Hi-N side of the field',
            'urea; applied to south half  (Hi-N side) of field only']:
            output = output.loc[output.FertilizerType != fertilizerType, ]



        #% #
        # What has no npk? -- can we just use npks?
        mask = (((output.TotalN.isna()) | (output.TotalN == 0)) &
                ((output.TotalP.isna()) | (output.TotalP == 0)) &
                ((output.TotalK.isna()) | (output.TotalK == 0)))
        print('There are', 
              output.loc[(mask) & (output.ExperimentCode!='IAI4'), ].shape[0], 
              'rows without NPK info where NPK info is expected.')


        output = output.loc[:, [
            'ExperimentCode',
            'TotalN',
            'TotalP',
            'TotalK',
            'Date',
            'Dropped'
        ]].rename(columns={'TotalN': 'N',
                           'TotalP': 'P',
                           'TotalK': 'K'}
                  ).melt(
            id_vars=['ExperimentCode', 'Date', 'Dropped'],
            var_name="Product"
        ).rename(columns={'value': 'QuantityPerAcre'})



        #% #

        output['ApplicationUnit'] = 'lbs/Acre'

        output


        fertilizerNoDate.loc[:, 'QuantityPerAcre'] = fertilizerNoDate.QuantityPerAcre.astype(float)

        # finally ready to join! 
        #TODO don't overwrite  managementyet. Need to pull irrigation 
        output = output.merge(fertilizerNoDate, how = 'outer').copy()


        #% #

        irrigationSubset = management.loc[:, [
        'ExperimentCode',
        'Date',   
        'Year',
        'Plot',

        'Irrigation',
        'WeatherStationIrrigation',
        'Application',
        'Product',

        'QuantityPerAcre',
        'ApplicationUnit'


        ]].drop_duplicates()

        irrigationSubset = irrigationSubset.loc[irrigationSubset.Irrigation != 'No', ]
        irrigationSubset= irrigationSubset.drop_duplicates()

        list(irrigationSubset.Irrigation.drop_duplicates())

        list(irrigationSubset.WeatherStationIrrigation.drop_duplicates())

        list(irrigationSubset.Application.drop_duplicates())

        #% #

        irrigationSubset= irrigationSubset.dropna(axis = 0, subset = [
            'Irrigation', 
            'WeatherStationIrrigation',
            'Application',
            'Product',
            'QuantityPerAcre',
            'ApplicationUnit'])

        irrigationSubset.loc[:, 'QuantityPerAcre']= irrigationSubset.loc[:, 'QuantityPerAcre'].astype(str)

        #% #

        output.loc[:, 'QuantityPerAcre'] = output.loc[:, 'QuantityPerAcre'].astype(str)


        #% #

        management = irrigationSubset.merge(output, how = 'outer')
        management['Application'] = 'fertilizer'
        management['Year'] = '2015'
        print('ran 2015')

    # TLC for 2014 goes here

    if expectedYear == ['2014']: 
        # drop herbicide/pesticide cols
        management = management.drop(columns = [entry for entry in list(management
                                     ) if entry in ['HerbicidePrePlant', 'HerbicidePostPlant', 'InsecticideType']])

        management


        #% # Fertilizer


        list(management.loc[management.FertilizerSchedule == 'see above', 'TotalN'])




        # alter so as to index off of only one variable without making a composite
        mask = ((management.FertilizerSchedule == 'see above'
                ) & (management.TotalN == '180lb per acre of NH3; and 18lb per acre from application of di-ammonium phosphate (DAP) at 100lb/acre'))
        management.loc[mask, 'FertilizerSchedule'] = 'see above1'


        mask = ((management.FertilizerSchedule == 'see above'
                ) & (management.TotalN == '180lb per acre of NH3; and 18lb per acre from application of diammonium phosphate (DAP) at 100lb/acre'))
        management.loc[mask, 'FertilizerSchedule'] = 'see above2'



        # from previous version
        # Consider:
        # <1> 'urea (46-0-0); applied to south half (1/2 Acre, Hi-N side) of field only: actual applied was 140 lb urea to the Hi-N side of the field |  | Ntot= 130 | Ptot= 0 | Ktot= 0',
        # <2> 'urea; applied to south half  (Hi-N side) of field only |  | Ntot= 0 | Ptot= 0 | Ktot= 0',
        # -- # <3> ' | Low-N history of fields 105 and 85: Fields 105 and 85 recieved an application of hardwood sawdust (May 6 field 85 and May 9 field 105) in 2011; and both fields received a second application of sawdust on May 23; 2012: not sawdust applied in 2013 or 2014 | Ntot= 130 lb/a; urea; on High N sides of each field only; no N applied on low N reps | Ptot= None | Ktot= None',
        # <4> ' |  | Ntot= 100 lb per acre (reps 1 3 5 7 9 11); 0 lb per acre (reps 2 4 6 8 10 12) | Ptot= 0 | Ktot= 0'



        # There are some entries which are too complicated and or vague to be of use to us. 
        # For these we'll want to drop all observations from that particular location and year.
        management['Dropped'] = False
        mask = management.FertilizerSchedule == 'Low-N history of fields 105 and 85: Fields 105 and 85 recieved an application of hardwood sawdust (May 6 field 85 and May 9 field 105) in 2011; and both fields received a second application of sawdust on May 23; 2012: not sawdust applied in 2013 or 2014'

        # Erase these cells to prevent usage
        for col in [entry for entry in list(management) if entry not in ['ExperimentCode', 'Year', 'Location', 'Dropped']]:
            management.loc[mask, col] = np.nan

        management.loc[mask, 'Dropped'] = True


        management.loc[mask,]


        replaceFertilizerSchedules = {
            'Preplant 120 lb K2O/ac as 0-0-60; 117 lb PAN/ac as poultry litter; 194 lb P2O5/ac as poultry litter; At plant: 5/5/2014; 26 lb N/ac as 20-10-0-1; 13 lb P2O5/ac as 20-10-0-1; At sidedress: 195 lb N/ac as 30% UAN\n':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize', 'fertilize', 'fertilize', 'fertilize'],
                    ['Preplant',  'Preplant',  'Preplant',
                     '2014-5-5',  '2014-5-5',  'Sidedress'],
                    ['K2O',       'N',         'P2O5',      'N',         'P2O5',      'N'],
                    ['120',         '117',         '194',
                     '26',          '13',          '195'],
                    ['lbs/Acre',  'lbs/Acre',  'lbs/Acre',
                     'lbs/Acre',  'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            '5/8:  140-0-0':
            pd.DataFrame(
                zip(['fertilize'],
                    ['2014-5-8'],
                    ['N'],
                    ['140'],
                    ['lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            '4/22:  dry fert 36-92-120; 4/23: 150 lbs Ammonia':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize', 'fertilize'],
                    ['2014-4-12', '2014-4-12', '2014-4-12', '2014-4-23'],
                    ['N', 'P', 'K',  'N'],
                    ['36', '92', '120', '39'],
                    ['lbs/Acre', 'lbs/Acre', 'lbs/Acre', 'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            '300#/acre of 10-0-30-12%S preplant and 26 gallons of 30% UAN (85# of N) was applied pre- emerge. Total N applied was 115 pounds per acre.':
            pd.DataFrame(
                zip(['fertilize'],
                    ['Preplant',     'Pre-emerge'],
                    ['10-0-30-12%S', 'N'],
                    ['300',          '85'],
                    ['lbs/Acre',     'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            '250 lbs/acre of 10-20-20 / 110lbs/acre  N 33.8 gallons of 30% UAN':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize'],
                    ['unknown',  'unknown',    'unknown'],
                    ['N',         'P',         'K'],
                    ['135.1204',       '50',        '50'],
                    ['lbs/Acre',  'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'UAN broadcast Preplant/ P+K 2x2 band Starter Fertilizer':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize'],
                    ['Preplant',  'Preplant', 'Preplant'],
                    ['N',         'P',        'K'],
                    ['165',         '54',         '24'],
                    ['lbs/Acre',  'lbs/Acre', 'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            '180 lb Nitrogen/acre preplant':
            pd.DataFrame(
                zip(['fertilize'],
                    ['Preplant'],
                    ['N'],
                    ['180'],
                    ['lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'none other than preplant above':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize'],
                    ['unknown',  'unknown',    'unknown'],
                    ['N',         'P',         'K'],
                    ['118',       '46',        '62'],
                    ['lbs/Acre',  'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'NOTES on fertilizer:  At planting 5/27/14 12-25-0-3.3S and 0.3 ZN was banded at 7.5 gallons per acre which comes out to 11 lbs of N and 22 lbs of P2O per acre.  Then side dress 6/30/14 33.8 gallons of 30 % UAN which is 110 lbs of N.':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize'],
                    ['2014-5-27', '2014-5-27', '2014-6-30'],
                    ['N',         'P2O',       'N'],
                    ['11',          '22',          '110'],
                    ['lbs/Acre',  'lbs/Acre', 'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'Preplant applied at 150 lbs 11-37-0+4lbs Zn then applied at V7 = 200 lbs 46-0-0':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize'],
                    ['Preplant', 'Preplant', 'V7'],
                    ['11-37-0',  'Zn',       '46-0-0'],
                    ['150',        '4',          '200'],
                    ['lbs/Acre', 'lbs/Acre', 'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            '3/6/14-UAN/10-34-0 applied with spreader truck. V5 stage 32-0-0 applied side dress':
            pd.DataFrame(
                zip(['fertilize',  'fertilize'],
                    ['2014-3-6',   'V5'],
                    ['npk10-34-0', 'npk32-0-0'],
                    ['235.3',        '863.96875'],
                    ['lbs/Acre',   'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'See above':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize'],
                    ['2014-04-17',  '2014-04-17',    '2014-04-17'],
                    ['N',         'P',         'K'],
                    ['450',       '192',        '210'],
                    ['lbs/Acre',  'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            '120 lbs/acre':
            pd.DataFrame(
                zip(['fertilize'],
                    ['unknown'],
                    ['N'],
                    ['120'],
                    ['lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'none other than applied in fall':
            pd.DataFrame(
                zip(['fertilize', 'fertilize',   'fertilize', 'fertilize'],
                    ['fall 2013', 'spring 2014', 'fall 2013', 'fall 2013'],
                    ['N',         'N',           'P',         'K'],
                    ['22',        '110',         '74',        '111'],
                    ['lbs/Acre',  'lbs/Acre',    'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'Pre plant incorporated fertilizer 10-10-30 at 474 lbs per acre; 24S liquid nitrogen 12 gals per acre side dress; 24S liquid nitrogen 12 gals per acre top dress':
            pd.DataFrame(
                zip(['fertilize',   'fertilize', 'fertilize'],
                    ['Preplant',    'Preplant',  'Preplant'],
                    ['npk10-10-30', 'N',         'N'],
                    ['474',           '26.3',        '26.3'],
                    ['lbs/Acre',    'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'NOTES on fertilizer:  At planting 5/27/14 12-25-0-3.3S and 0.3 ZN was banded at 7.5 gallons per acre which comes out to 11 lbs of N and 22 lbs of P2O per acre.� Then side dress 6/30/14 33.8 gallons of 30 % UAN which is 110 lbs of N.':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize'],
                    ['2014-5-27', '2014-5-27', '2014-6-30'],
                    ['N',         'P2O',       'N'],
                    ['11',          '22',          '110'],
                    ['lbs/Acre',  'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'Pre-plant (150 lbs 11-37-0 + 4lbs Zn) applied on 1/19/2014 in liquid form; and 200 lbs 32-0-0 applied in liquid form on 4-19-2014 ~V-7.':
            pd.DataFrame(
                zip(['fertilize',  'fertilize', 'fertilize'],
                    ['2014-1-14',  '2014-1-14', '2014-4-19'],
                    ['npk11-37-0', 'Zn',        'N'],
                    ['150',          '4',           '91.5'],
                    ['lbs/Acre',   'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'Pre-plant (150 lbs 11-37-0 + 4lbs Zn) applied on 1/29/2014 in liquid form; and 200 lbs 32-0-0 applied in liquid form on 4-15-2014.':
            pd.DataFrame(
                zip(['fertilize',  'fertilize', 'fertilize'],
                    ['2014-1-29',  '2014-1-29', '2014-4-15'],
                    ['npk11-37-0', 'Zn',        'N'],
                    ['150',          '4',           '91.5'],
                    ['lbs/Acre',   'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'see above1': pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize'],
                    ['unknown',  'unknown',    'unknown'],
                    ['N',         'P',         'K'],
                    ['198',       '46',        '51'],
                    ['lbs/Acre',  'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            'see above2': pd.DataFrame(
                zip(['fertilize', 'fertilize'],
                    ['unknown',   'unknown'],
                    ['N',         'P'],
                    ['198',       '46'],
                    ['lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            '3/24/2014 8/20/1930 broadcast 1000#/A; 4/4/2014 10-34-0 row band 10gal/A; 4/4/2014 Agri Starter Orangeburg row band 4qt/A; 5/7/2014;28-0-0-0.5;row band;100#N/A; 5/7/2014;Boron;row band;20 ounce/A; 5/14/2014;28-0-0-0.5;row band;100#N/A':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize', 'fertilize', 'fertilize', 'fertilize', 'fertilize'],
                    ['2014-3-24', '2014-3-24', '2014-3-24', '2014-4-4',
                        '2014-4-4',  '2014-5-7',  '2014-5-14'],
                    ['N',         'P',         'K',         'N',
                        'P',         'N',         'N'],
                    ['80',        '200',       '300',       '12',
                        '40',        '100',       '100'],
                    ['lbs/Acre',  'lbs/Acre',  'lbs/Acre',  'lbs/Acre',
                        'lbs/Acre',  'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            # Any data pertaining to the previous year is ignored. We assume nutrients listed are available to the plants.
            'P&K applied in fall 2013; Liquid N applied preplant;  Dry starter fertilizer applied with planter':
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize'],
                    ['unknown',   'unknown',   'unknown'],
                    ['N',         'P',         'K'],
                    ['156.131',   '60.6682',   '107.061'],
                    ['lbs/Acre',  'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit']),

            "fall '13:  NH3; 4/12:  11-52-0  Map; 0-0-60 granular potash":
            pd.DataFrame(
                zip(['fertilize', 'fertilize', 'fertilize'],
                    ['unknown',   'unknown',   'unknown'],
                    ['N',         'P',         'K'],
                    ['156',       '80',        '75'],
                    ['lbs/Acre',  'lbs/Acre',  'lbs/Acre']
                    ), columns=['Application', 'Date', 'Product', 'QuantityPerAcre', 'ApplicationUnit'])
        }


        fertilizerAccumulator = pd.DataFrame()

        for currentKey in list(replaceFertilizerSchedules.keys()):
            currentValue = replaceFertilizerSchedules[currentKey]

            temp = management.loc[management.FertilizerSchedule == currentKey, ['ExperimentCode', 'Year', 'Location', 'Plot']]
            temp['join'] = True
            currentValue['join'] = True

            temp = temp.merge(currentValue, how = 'outer')

            if ((fertilizerAccumulator.shape[0] == 0) | (fertilizerAccumulator.shape[1] == 0)):
                fertilizerAccumulator = temp
            else:
                fertilizerAccumulator = fertilizerAccumulator.merge(temp, how = 'outer')


                fertilizerAccumulator


        #% # Fertigation

        print('FertigationSchedule does not overlap with growing:\n', list(management.FertigationSchedule.drop_duplicates()))


        #% # Irrigation


        manageIrrigation = management.loc[:, [
            'ExperimentCode',
            'Year',
            'Location',
            'Irrigation',
            'WeatherStationIrrigation',
            'IrrigationSchedule',
            'Plot']]


        replaceIrrigationSchedules = {
            '6-17- .5"; 6-29- .5"; 7-1 - .5\'; 7-8 - .3"; 7-21 - .3"; 7-22 - .25"; 7-29 - .25"; 8-1 - .25"; 8-12 - .3"':
            pd.DataFrame(zip(
                ['2014-6-17', '2014-6-29', '2014-7-1', '2014-7-8', '2014-7-21',
                 '2014-7-22', '2014-7-29', '2014-8-1', '2014-8-12'],
                ['.5', '.5', '.5', '.3', '.3', '.25', '.25', '.25', '.3']), columns=['Date', 'QuantityPerAcre']),

            'May 9; May 28; June 5; June 16; July 7.  Approx. 1 inch of rain equivalent was applied each time.':
            pd.DataFrame(zip(
                ['2014-5-9', '2014-5-28', '2014-6-5', '2014-6-16', '2014-7-7'],
                ['1',        '1',         '1',        '1',         '1']), columns=['Date', 'QuantityPerAcre']),

            '7/21/14  0.5" target; measured 0.37"':
            pd.DataFrame(zip(
                    ['2014-7-21'],
                    ['0.37']), columns=['Date', 'QuantityPerAcre']),

            '4/29/2014 and 6/17/2014; flood irrigated':
            pd.DataFrame(zip(
                ['2014-4-29', '2014-6-17'],
                ['unknown',   'unknown']), columns=['Date', 'QuantityPerAcre']),

            'May 4; 2014 and June 19; 2014; flood irrigated':
            pd.DataFrame(zip(
                ['2014-5-4', '2014-6-19'],
                ['unknown',  'unknown']), columns=['Date', 'QuantityPerAcre']),

            'Accidently irrigated first 2 ranges (plots 1-20) on 5/5/2014; overflow.': 
            pd.DataFrame(zip(
                ['2014-5-5' for i in range(20)],
                ['unknown' for i in range(20)],
                [str(i+1) for i in range(20)]), columns=['Date', 'QuantityPerAcre', 'Plot']),

            '0.5" 5/9/2014; 0.5" 5/23/2014;  0.5" 5/30/2014; 1" 6/6/2014; 1" 6/25/14; 1" 6/30/14; 1" 7/2/2014; 1" 7/5/14; 1" 7/7/14; ':
            pd.DataFrame(zip(
                    ['2014-5-9', '2014-5-23', '2014-5-30', '2014-6-6', '2014-6-25',
                     '2014-6-30', '2014-7-2', '2014-7-5', '2014-7-7'],
                    [str(i) for i in [0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1]]), columns=['Date', 'QuantityPerAcre'])
        }



        # extend those entries that need extending

        irrigationAccumulator = pd.DataFrame()

        for currentKey in list(replaceIrrigationSchedules.keys()):
            currentValue = replaceIrrigationSchedules[currentKey]

            temp = manageIrrigation.loc[manageIrrigation.IrrigationSchedule == currentKey, 
                                        ['ExperimentCode', 'Year', 'Location', 'Irrigation', 'WeatherStationIrrigation']]

            temp['join'] = True
            currentValue['join'] = True

            temp = temp.merge(currentValue, how = 'outer')
            temp['IrrigationSchedule' ]= currentKey

            if ((irrigationAccumulator.shape[0] == 0) | (irrigationAccumulator.shape[1] == 0)):
                irrigationAccumulator = temp
            else:
                irrigationAccumulator = irrigationAccumulator.merge(temp, how = 'outer')

        irrigationAccumulator = irrigationAccumulator.drop(columns = ['join'])



        irrigationAccumulator
        # manageIrrigation.merge(irrigationAccumulator, how = 'outer')


        # Remove same from manageIrrigation so we can join without duplicating entries
        for currentKey in list(replaceIrrigationSchedules.keys()):
            manageIrrigation.loc[manageIrrigation.IrrigationSchedule != currentKey, :]




        manageIrrigation = manageIrrigation.merge(irrigationAccumulator, how = 'outer')
        manageIrrigation = manageIrrigation.drop(columns = ['IrrigationSchedule'])
        manageIrrigation['Product'] = np.nan
        manageIrrigation.loc[manageIrrigation.Date.notna(), 'Product'] = 'Water'



        #% # Merging

        # if irriation is captured by weahter station then we don't need it here.

        manageIrrigation = manageIrrigation.loc[manageIrrigation.Date.notna()]


        manageIrrigation = manageIrrigation.loc[(
            (manageIrrigation.Irrigation != 'no'
        ) & (manageIrrigation.Irrigation != 'No'
        ) & (manageIrrigation.Irrigation != 'none')) ]


        manageIrrigation = manageIrrigation.drop(columns = ['Irrigation'])

        manageIrrigation['Application'] = 'irrigation'


    #     list(manageIrrigation.Irrigation.drop_duplicates())

        management = fertilizerAccumulator.merge(manageIrrigation, how = 'outer').drop_duplicates()
        print('ran 2014')


    #### General Management
    # Manual row fixing goes here

    
    # 2019
    # Product
    if expectedYear == ['2019']:
        searchString = 'Glyphosate 1Qt/A burndown 4/4/19'
        management.loc[management.Product == searchString, 'Date'] = '4/4/19'
        management.loc[management.Product == searchString, 'QuantityPerAcre'] = 1
        management.loc[management.Product == searchString, 'Product'] = 'Glyphosate'
                
        searchString = '100 of N'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'N'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 100       
        
        searchString = '30-26-0-6S'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = '30-26-0-6S'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 100
        
    
    
    # 2018
    # Product
    if expectedYear == ['2018']:
        searchString = '27N, 26P'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 142

        searchString = '180 N'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'N'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 180

        searchString = 'NPK 0-0-39'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'K'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 39

        searchString = 'NPK 5-25-0'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 48

        searchString = '130 (42 GPA)'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 130 

        searchString = '120-0-0'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 120 

        searchString = '0-100-100'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'npk50-50-0'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 200


    # 2017
    if expectedYear == ['2017']:
        ## Products ====
        dropTheseProducts= [
            'Squarts Acuron', 
            '1 quart/acre of Atrazine and 2 quart/acre of Warrant', 
            'Accent 2/3 ounce/acre and Prowl H2O 1quart/acre', 
            'Harness Xtra 2.6L']
        for product in dropTheseProducts:
            management = management.loc[management.Product != product, ]

        management.loc[management.Product == '38.5 lb of Nitrogen, 182 lb of P2O5, 210 lb of K2O', 'QuantityPerAcre'] = '430.5'
        management.loc[management.Product == '38.5 lb of Nitrogen, 182 lb of P2O5, 210 lb of K2O', 'ApplicationUnit'] = 'lbs/Acre'
        management.loc[management.Product == '38.5 lb of Nitrogen, 182 lb of P2O5, 210 lb of K2O', 'Product'] = 'npk8.9-42.3-48.8'


        ## QuantititesPerAcre
        management.loc[management.QuantityPerAcre == '100 units per acre', 'ApplicationUnit'] = 'units/Acre'
        management.loc[management.QuantityPerAcre == '100 units per acre', 'QuantityPerAcre'] = 100

        management.loc[management.QuantityPerAcre == '80 units per acre', 'ApplicationUnit'] = 'units/Acre'
        management.loc[management.QuantityPerAcre == '80 units per acre', 'QuantityPerAcre'] = 80

        management.loc[management.QuantityPerAcre == '260 lb/ac', 'ApplicationUnit'] = 'lbs/Acre'
        management.loc[management.QuantityPerAcre == '260 lb/ac', 'QuantityPerAcre'] = 260

        searchString = '375 lbs./A'
        management.loc[management.QuantityPerAcre == searchString, 'ApplicationUnit'] = 'lbs/Acre'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 375

        searchString = '42 GPA'
        management.loc[management.QuantityPerAcre == searchString, 'ApplicationUnit'] = 'GPA'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 42

        searchString = '110 lbs./N'
        management.loc[management.QuantityPerAcre == searchString, 'ApplicationUnit'] = 'lbs/Acre'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 110

        searchString = 'NPK 14-65-0'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 125 # so that %*lbs match expectation

        searchString = 'NPK 0-0-51'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 85

        searchString = '27N, 26P'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 142

        ## ApplicationUnits ====

        ## Times ====
        management.loc[management.Date == '13-Apr', 'Date'] = '13-Apr-2017'

    # 2016
    if expectedYear == ['2016']:
        dropTheseProducts= [
            'apllied 32oz/acre Liberty 280SL as a burndown', 
            'applied 32oz/acre Liberty 280 SL ', 
            'applied 24oz/acre Prowl H2O for emergency weed control', 
            'applied 48oz / acre Lexar EZ herbicide', 
            'applied 48oz Roundup PowerMax II / acre as a burn down/pre-emerge', 
            'applied 32oz Liberty 280SL / acre as a burn down/pre-emerge', 
            'applied 64oz Warrant / acre as a burn down/pre-emerge', 
            'Harness Extra 5.6l']
        for product in dropTheseProducts:
            management = management.loc[management.Product != product, ]


        ### Product
        searchString = '5/20/2020'
        management.loc[management.Product == searchString, 'QuantityPerAcre'] = 100
        management.loc[management.Product == searchString, 'Product'] = 'npk7.5-30-30'

        searchString = 'Side dress applicaton 28%'
        management.loc[management.Product == searchString, 'QuantityPerAcre'] = 134 # Previously 134-0-0, presumably NPK lbs
        management.loc[management.Product == searchString, 'Product'] = 'N'

        searchString = 'incorporated 100lbs N / acre '
        management.loc[management.Product == searchString, 'QuantityPerAcre'] = 100 # Previously 134-0-0, presumably NPK lbs
        management.loc[management.Product == searchString, 'Product'] = 'N'

        searchString = 'incorporated 80lbs P2O5 / acre'
        management.loc[management.Product == searchString, 'QuantityPerAcre'] = 80 # Previously 134-0-0, presumably NPK lbs
        management.loc[management.Product == searchString, 'Product'] = 'P'

        searchString = 'incorporated 100lbs K2O / acre'
        management.loc[management.Product == searchString, 'QuantityPerAcre'] = 100 # Previously 134-0-0, presumably NPK lbs
        management.loc[management.Product == searchString, 'Product'] = 'K'


        searchString = 'applied 97.789lbs N / acre from urea treated with 2.4oz Agrotain / 50lbs'
        management.loc[management.Product == searchString, 'QuantityPerAcre'] =  97.789 # Previously 134-0-0, presumably NPK lbs
        management.loc[management.Product == searchString, 'Product'] = 'N'



        ### QuantityPerAcre
        searchString = '60 gallons/acre'
        management.loc[management.QuantityPerAcre == searchString, 'ApplicationUnit'] = 'gal/Acre'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  60

        searchString = '20-9-3'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'npk20-9-3'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  100

        searchString = '1/2 ton/A'
        management.loc[management.QuantityPerAcre == searchString, 'ApplicationUnit'] = 'ton/Acre'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  0.5

        searchString = '375 lbs./A'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  375

        searchString = '50 GPA'
        management.loc[management.QuantityPerAcre == searchString, 'ApplicationUnit'] = 'GPA'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  50

        searchString = '.75 inch'
        management.loc[management.QuantityPerAcre == searchString, 'ApplicationUnit'] = 'in/Acre'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  0.75

        searchString = 'NPK 27,26,0\n'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'npk27-26-0'
        # management.loc[management.QuantityPerAcre == searchString, 'ApplicationUnit'] = 'ton/Acre'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  100

        searchString = '91 lb'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 91

        searchString =  '180 N'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'N'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  180


        searchString =  '100 Actual N'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'N'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  100

        searchString =  '90 Actual K'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'K'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  90

        searchString =  '30 Actual P'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'P'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  30

        searchString =  '60 Actual N'
        management.loc[management.QuantityPerAcre == searchString, 'Product'] = 'N'
        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] =  60


        searchString = '150+5 Zn'
        temp = management.loc[management.QuantityPerAcre == searchString, ].copy()
        temp.loc[:, ['Product', 'QuantityPerAcre']] = ['Zn', 5]
        management= management.merge(temp, how = 'outer')

        management.loc[management.QuantityPerAcre == searchString, 'QuantityPerAcre'] = 150

    # 2015

    # 2014


    #% #
    print('Unexpected cols in mangement:')
    [col for col in list(management) if col not in expectedManagementCols]



    #% #
    management.loc[:, 'Application'] = rename_col_entries(
            df= management,
            colIn='Application',
            newToOldDict=newApplicationNames)



    #% #
    management.loc[:, 'Product'] = rename_col_entries(
            df= management,
            colIn='Product',
            newToOldDict=newProductNames)



    #% #
    # Filter again --

    # We only care about irrigation and fertilizer so we'll drop anything that matches other keys
    discardTheseProducts = list(newHerbicideNames.keys()
                         )+list(newInsecticideNames.keys()
                         )+list(newFungicideNames.keys()
                         )+list(newMiscNames.keys())

    # bool to keep
    mask = [True if entry not in discardTheseProducts else False for entry in management.Product]
    management = management.loc[mask, :]

    # we don't care about rows missing both Application + Product. Those are from merging.
    mask = (management.Application.notna()) | (management.Product.notna())
    management= management.loc[mask, :]



    #% #
    dfName = 'management'

    expectedApplications = list(newApplicationNames.keys())
    # replaceApplications = {}

    for key in list(replaceApplicationUnits.keys()):
        management.loc[management.ApplicationUnit == key, 'ApplicationUnit'] = replaceApplicationUnits[key]


    print('unexpected applications')
    print([exp for exp in list(eval(dfName).Application.drop_duplicates()) if exp not in expectedApplications])

    print('unexpected units')
    print([exp for exp in list(eval(dfName).ApplicationUnit.drop_duplicates()) if exp not in expectedApplicationUnits])



    #% #
    print('Not converted into times by default!')
    # [entry for entry in eval(dfName).Date.drop_duplicates() if pd.isnull(pd.to_datetime(entry)) ]
    [entry for entry in eval(dfName).Date.drop_duplicates() if not convertable_to_datetime_not_NaT(entry) ]


    #% #
    print('QuantityPerAcre not converted into numeric!')
    quantitiesFound = list(eval(dfName).QuantityPerAcre.drop_duplicates())
    [entry for entry in quantitiesFound if convertable_to_float(value = entry) == False]


    #% #
    # we have to check the products some other way after parsing. Maybe check based on grouping of applications. 

    expectedProducts = list(newProductNames.keys())
    # replaceProducts = {}
    print([exp for exp in list(eval(dfName).Product.drop_duplicates()) if exp not in expectedProducts])


    return(management)


def prep_data(expectedYear=['2014'],
              expectedExperimentCodes=expectedExperimentCodes,
              replaceExperimentCodes=replaceExperimentCodes,
              expectedManagementCols=expectedManagementCols,
              newApplicationNames=newApplicationNames,
              expectedApplicationUnits=expectedApplicationUnits,
              replaceApplicationUnits=replaceApplicationUnits,
              newProductNames=newProductNames
):
    
    
    # update names ----
    if expectedYear == ['2019']:
        data2018 =      [agro2019,    weather2019,   metadata2019,   soil2019,   pheno2019]
        dataNames2018 = ['agro2019', 'weather2019', 'metadata2019', 'soil2019', 'pheno2019']
    if expectedYear == ['2018']:
        data2018 =      [agro2018,    weather2018,   metadata2018,   soil2018,   pheno2018,   geno2018]
        dataNames2018 = ['agro2018', 'weather2018', 'metadata2018', 'soil2018', 'pheno2018', 'geno2018']
    if expectedYear == ['2017']:
        data2018 =      [agro2017, weather2017, metadata2017, soil2017, pheno2017, geno2017]
        dataNames2018 = ['agro2017', 'weather2017', 'metadata2017', 'soil2017', 'pheno2017', 'geno2017']
    if expectedYear == ['2016']:
        data2018 =      [agro2016, weather2016, metadata2016, soil2016, pheno2016]
        dataNames2018 = ['agro2016', 'weather2016', 'metadata2016', 'soil2016', 'pheno2016']  
    if expectedYear == ['2015']:
        data2018 =      [agro2015, weather2015, metadata2015, soil2015, pheno2015] 
        dataNames2018 = ['agro2015', 'weather2015', 'metadata2015', 'soil2015', 'pheno2015']    
    if expectedYear == ['2014']:
        data2018 =      [          weather2014, metadata2014,           pheno2014]
        dataNames2018 = [            'weather2014', 'metadata2014',             'pheno2014']

    for i in range(len(data2018)):
        data2018[i] = rename_columns(
            dataIn=data2018[i],
            colNamePath='../data/external/UpdateColNames.csv',
            logfileName=dataNames2018[i]
        )


    # regroup columns ----
    colGroupings = pd.read_csv('../data/external/UpdateColGroupings.csv')
    if expectedYear not in [['2014'], ['2015']]:
        out2018 = sort_cols_in_dfs(
            inputData=data2018,
            colGroupings=colGroupings)
    else:
        out2018 = sort_cols_in_dfs(
            inputData=data2018,
            colGroupings=colGroupings.drop(columns = ['Genotype', 'GenotypeType']))



    # Try to fix experiment codes --
    for dfKey in list(out2018.keys()):
        df = out2018[dfKey]
        if 'ExperimentCode' in list(df):
            for key in list(replaceExperimentCodes.keys()):
                df.loc[df.ExperimentCode == key, 'ExperimentCode'] = replaceExperimentCodes[key]
        out2018[dfKey] = df


    # separate to make these easier to work with ----

    geno = pd.DataFrame()
    pheno = pd.DataFrame()
    management = pd.DataFrame()
    metadata = pd.DataFrame()
    soil = pd.DataFrame()
    weather = pd.DataFrame()

    if 'Genotype' in out2018.keys():
        geno = out2018['Genotype'] 
    if 'Phenotype' in out2018.keys():
        pheno = out2018['Phenotype'] 
    if 'Management' in out2018.keys():
        management = out2018['Management'] 
    if 'Metadata' in out2018.keys():
        metadata =out2018['Metadata'] 
    if 'Soil' in out2018.keys():
        soil = out2018['Soil'] 
    if'Weather' in out2018.keys():
        weather = out2018['Weather']



   


    # Any alterations to correct experiment codes should go here --
    if expectedYear == ['2018']:
        pheno= pheno.loc[pheno.ExperimentCode.notna(), ] # rows without expcode lack info (from including pedigree)
    if expectedYear == ['2017']:
        pheno= pheno.loc[pheno.ExperimentCode.notna(), ] # rows without expcode lack info (from including pedigree)
    # if expectedYear == ['2016']:
    # if expectedYear == ['2015']:
    # if expectedYear == ['2014']:



    metadataExps = list(metadata.ExperimentCode.drop_duplicates())

    print('Exps In Management missing in Metadata')
    print([exp for exp in list(management.ExperimentCode.drop_duplicates()) if exp not in metadataExps] )
    print('Exps In Soil missing in Metadata')
    print([exp for exp in list(soil.ExperimentCode.drop_duplicates()) if exp not in metadataExps] )
    print('Exps In Weather missing in Metadata')
    print([exp for exp in list(weather.ExperimentCode.drop_duplicates()) if exp not in metadataExps] )
    print('Exps In Phenotype missing in Metadata')
    print([exp for exp in list(pheno.ExperimentCode.drop_duplicates()) if exp not in metadataExps] ) 


    print('')
    print('Exps In Metadata missing in Management')
    print([exp for exp in metadataExps if exp not in list(management.ExperimentCode.drop_duplicates())] )
    print('Exps In Metadata missing in Soil')
    print([exp for exp in metadataExps if exp not in list(soil.ExperimentCode.drop_duplicates())] )
    print('Exps In Metadata missing in Weather')
    print([exp for exp in metadataExps if exp not in list(weather.ExperimentCode.drop_duplicates())] )
    print('Exps In Metadata missing in Phenotype')
    print([exp for exp in metadataExps if exp not in list(pheno.ExperimentCode.drop_duplicates())] ) 
        
        
        
        



    # General Checks --
    ## Experiment Codes ==
    for dfName in ['pheno', 'management', 'metadata', 'soil', 'weather']:
        print(dfName)
        print([exp for exp in list(eval(dfName).ExperimentCode.drop_duplicates()) if exp not in expectedExperimentCodes])


    # check that the year is right. (everywhere except geno)
    for dfName in ['pheno', 'management', 'metadata', 'soil', 'weather']:
        print(dfName)
        print([exp for exp in list(eval(dfName).Year.drop_duplicates()) if exp not in expectedYear + [np.nan]])

    # set year
    for df in [pheno, management, metadata, soil, weather]:
        df.loc[:, 'Year'] = expectedYear[0]
        
        
#     if (('Pedigree' in list(geno)) & ('Pedigree' in list(pheno))):
#         # Are there any pedigrees in phenotype that aren't in genotype?
#         print([pg for pg in list(geno.Pedigree.drop_duplicates()) if pg not in list(pheno.Pedigree.drop_duplicates())])
#         # the other way around?
#         print([pg for pg in list(pheno.Pedigree.drop_duplicates()) if pg not in list(geno.Pedigree.drop_duplicates())])


    # Filter --

    # Make sure we're only looking at something we care about.
    pheno= pheno.dropna(subset = ['GrainYield'])
    pheno= pheno.loc[pheno.PlotDiscarded != 'Yes']


    list(pheno.PlotDiscarded.drop_duplicates())


    # management --------------------------------------------------------------------------------------
    management = clean_managment(
        expectedYear = expectedYear,
        management = management
    )

    print('management contains:')
    print(list(management))



    # metadata --------------------------------------------------------------------------------------
    metadata['ExpLon'] = np.nan
    metadata['ExpLat'] = np.nan

    gpsCols = [
     'WeatherStationLat',
     'InfieldStationLat',
     'FieldCorner1Lat',
     'FieldCorner2Lat',
     'FieldCorner3Lat',
     'FieldCorner4Lat',

     'WeatherStationLon',
     'InfieldStationLon',
     'FieldCorner1Lon',
     'FieldCorner2Lon',
     'FieldCorner3Lon',
     'FieldCorner4Lon']
    metadata.loc[:, gpsCols] = metadata.loc[:, [entry for entry in list(metadata) if entry in gpsCols]].astype(float)


    mask = ((metadata['ExpLon'].isna()) | (metadata['ExpLat'].isna()))

#     metadata.loc[mask, 'ExpLon'] = (metadata.loc[mask, 'FieldCorner1Lon'
#                 ] + metadata.loc[mask, 'FieldCorner2Lon'
#                 ] + metadata.loc[mask, 'FieldCorner3Lon'
#                 ] + metadata.loc[mask, 'FieldCorner4Lon'
#                 ])/4

#     metadata.loc[mask, 'ExpLat'] = (metadata.loc[mask, 'FieldCorner1Lat'
#                 ] + metadata.loc[mask, 'FieldCorner2Lat'
#                 ] + metadata.loc[mask, 'FieldCorner3Lat'
#                 ] + metadata.loc[mask, 'FieldCorner4Lat'
#                 ])/4
    
    metadata.loc[mask, 'ExpLon'] = metadata.loc[mask, ['FieldCorner1Lon', 
                                                       'FieldCorner2Lon', 
                                                       'FieldCorner3Lon', 
                                                       'FieldCorner4Lon']].mean(axis = 1)

    metadata.loc[mask, 'ExpLat'] = metadata.loc[mask, ['FieldCorner1Lat', 
                                                       'FieldCorner2Lat', 
                                                       'FieldCorner3Lat', 
                                                       'FieldCorner4Lat']].mean(axis = 1)
    
    mask = ((metadata['ExpLon'].isna()) | (metadata['ExpLat'].isna()))
    metadata.loc[mask, 'ExpLon'] = metadata.loc[mask, 'WeatherStationLon']
    metadata.loc[mask, 'ExpLat'] = metadata.loc[mask, 'WeatherStationLat']
    
    # Ensure that within a year each experiment code maps onto exactly one gps coordinate set.
    for code in list(metadata['ExperimentCode'].drop_duplicates()):
        metadata.loc[metadata.ExperimentCode == code, 'ExpLat'] = metadata.loc[metadata.ExperimentCode == code, 'ExpLat'].mean()
        metadata.loc[metadata.ExperimentCode == code, 'ExpLon'] = metadata.loc[metadata.ExperimentCode == code, 'ExpLon'].mean()
    
    # Weather --------------------------------------------------------------------------------------
    #TODO data reduction and summarizing goes here

    # reduce output --------------------------------------------------------------------------------------

    # What's the minimum set that we actually want?

    # geno -------- all expected
    # management -- all expected
    # soil -------- all expected
    # weather ----- all expected, but with data reduction
    # metadata ---- 'ExperimentCode', 'Year' 
    #          new: 'ExpLon', 'ExpLat'
    #     possible: 'PreviousCrop', 
                  # 'PreplantTillage', 
                  # 'InseasonTillage',    
                  # 'MoistureSystem', 
                  # 'MoistureNeeded',

    # phenotype --- == Identifiers ==
                    # 'ExperimentCode',
                    # 'Year',
                    # 'PlotLength',
                    # 'AlleyLength',
                    # 'RowSpacing',
                    # 'RecId',
                    # 'Source',
                    # 'Pedigree',
                    # 'Family',
                    # 'Replicate',
                    # 'Block',
                    # 'Plot',
                    # 'Range',
                    # 'Pass',

                   #== Timing == 
                    # 'DatePlanted',
                    # 'DateHarvested',
                    # 'Anthesis',
                    # 'Silking',
                    # 'DaysToPollen',
                    # 'DaysToSilk',

                   #== Measures of interest ==
                    # 'Height',
                    # 'EarHeight',
                    # 'StandCount',
                    # 'PercentStand',
                    # 'RootLodging',
                    # 'StalkLodging',
                    # 'PercentGrainMoisture',
                    # 'TestWeight',
                    # 'PlotWeight',
                    # 'GrainYield'


    metadataSelectionList = ['ExperimentCode', 'Year', 'Location',
    'ExpLon', 
    'ExpLat',
    'PreviousCrop', 
    'PreplantTillage', 
    'InseasonTillage',    
    'MoistureSystem', 
    'MoistureNeeded']



    phenoSelectionList = ['ExperimentCode',
    'Year',
    'PlotLength',
    'AlleyLength',
    'RowSpacing',
    'RecId',
    'Source',
    'Pedigree',
    'Family',
    'Replicate',
    'Block',
    'Plot',
    'Range',
    'Pass',

    'DatePlanted',
    'DateHarvested',
    'Anthesis',
    'Silking',
    'DaysToPollen',
    'DaysToSilk',

    'Height',
    'EarHeight',
    'StandCount',
    'PercentStand',
    'RootLodging',
    'StalkLodging',
    'PercentGrainMoisture',
    'TestWeight',
    'PlotWeight',
    'GrainYield']

    

    metadata = metadata.loc[:, [entry for entry in list(metadata) if entry in metadataSelectionList]]          
        
    print('-------- --------')
    print('The following metadata entries lack ExperimentCodes:')
    mask = ((metadata.ExperimentCode.isna()) | (metadata.ExperimentCode == 'nan'))
    printOut = metadata.loc[mask, :].reset_index().drop(columns = ['index'])
    for i in range(printOut.shape[0]):
        print(printOut.loc[i, :])
    print('-------- --------')

    pheno = pheno.loc[:, [entry for entry in list(pheno) if entry in phenoSelectionList]] 





    # return these packaged like output2018 was.
    returnDict = {
        'Genotype': geno,
        'Phenotype': pheno,
        'Management': management,
        'Metadata': metadata,
        'Soil': soil,
        'Weather': weather
    }
    return(returnDict)


# %%
res2014 = prep_data(expectedYear=['2014'])

# %%
res2015 = prep_data(expectedYear=['2015'])

# %%
res2016 = prep_data(expectedYear=['2016'])

# %%
res2017 = prep_data(expectedYear=['2017'])

# %%
res2018 = prep_data(expectedYear=['2018'])

# %%
res2019 = prep_data(expectedYear=['2019'])

# %% [markdown]
# ## Use result dicts to create new dataframes

# %%
newGenotype = res2018['Genotype'
        ].merge(res2017['Genotype'], how = 'outer')

# %%
newPhenotype = res2019['Phenotype'
        ].merge(res2018['Phenotype'], how = 'outer'
        ).merge(res2017['Phenotype'], how = 'outer'
        ).merge(res2016['Phenotype'], how = 'outer'
        ).merge(res2015['Phenotype'], how = 'outer'
        ).merge(res2014['Phenotype'], how = 'outer')

# %%
newManagement = res2019['Management'
        ].merge(res2018['Management'], how = 'outer'
        ).merge(res2017['Management'], how = 'outer'
        ).merge(res2016['Management'], how = 'outer'
        ).merge(res2015['Management'], how = 'outer'
        ).merge(res2014['Management'], how = 'outer')

# %%

# %%
newMetadata = res2019['Metadata'
        ].merge(res2018['Metadata'], how = 'outer'
        ).merge(res2017['Metadata'], how = 'outer'
        ).merge(res2016['Metadata'], how = 'outer'
        ).merge(res2015['Metadata'], how = 'outer'
        ).merge(res2014['Metadata'], how = 'outer')

# %%
newSoil = res2019['Soil'
        ].merge(res2018['Soil'], how = 'outer'
        ).merge(res2017['Soil'], how = 'outer'
        ).merge(res2016['Soil'], how = 'outer'
        ).merge(res2015['Soil'], how = 'outer'
        ).merge(res2014['Soil'], how = 'outer')

# %%
newWeather = res2019['Weather'
        ].merge(res2018['Weather'], how = 'outer'
        ).merge(res2017['Weather'], how = 'outer'
        ).merge(res2016['Weather'], how = 'outer'
        ).merge(res2015['Weather'], how = 'outer'
        ).merge(res2014['Weather'], how = 'outer')


# %%
# res2017G = res2017['Genotype']
# res2017G = res2017G.loc[res2017G.GenotypeCode.notna()]

# %%
# temp.loc[temp.Pedigree == 'PHW03/PHRE1']
# temp = res2018['Genotype']
# temp#.loc[temp.GenotypeCode.notna()]

# %%
# geno2017.loc[geno2017['Female Pedigree'] == "", :]

# %%
# newGenotype.loc[newGenotype.Pedigree == 'PHW03/PHRE1']
# geno2017.info()

# %%
def track_newPheno_grainyield_obs(
    df = newPhenotype,
    markerName = 'Epoch2',
    expObsTrackerDf = ''#expObsTrackerDf
):
    df = df.loc[df.GrainYield.notna(), :].drop_duplicates()
    if type(expObsTrackerDf) == str:
        expObsTrackerDf = df.loc[:, ['GrainYield', 'Year']].groupby(['Year']).count().reset_index().rename(columns = {'GrainYield':markerName})
    else:
        expObsTrackerDf = expObsTrackerDf.merge(df.loc[:, ['GrainYield', 'Year']].groupby(['Year']).count().reset_index().rename(columns = {'GrainYield':markerName}))


    for entry in [entry for entry in list(expObsTrackerDf) if entry not in ['Year']]:
        plt.plot(expObsTrackerDf.Year, expObsTrackerDf[entry], label = entry)    
    plt.legend(loc = 'upper left')
    plt.show()

    return(expObsTrackerDf)
    


# %%
expObsTrackerDf = track_newPheno_grainyield_obs(
    df = newPhenotype,
    markerName = 'Start 3.4',
    expObsTrackerDf = ''#expObsTrackerDf
)

# %%

# %% [markdown]
# # Can we match up the phenotypes with their genotype correctly?

# %%
nP = newPhenotype.copy()
nP['GBSYear'] = ''
mask = (nP.Year == '2019') | (nP.Year == '2018')
nP.loc[mask, 'GBSYear'] = '2018'
nP.loc[~mask, 'GBSYear'] = '2017'

# %%
nP2018 = nP.loc[nP.GBSYear == '2018', ].copy()
nP2017 = nP.loc[nP.GBSYear == '2017', ].copy()

gbs2018 = res2018['Genotype']
gbs2017 = res2017['Genotype']

# 'PHN11_PHW65_0212'

# %%
# nP2018

# %%
# geno2018

# %%

# %%
## Clean up pedigree column to aid in matching

import re
geno2018['PedigreeClean'] = [re.sub("[*]", "_", str(entry).replace(' ', '').upper()) for entry in geno2018['Inbred Genotype Names']]
geno2017 = geno2017.loc[:, ['Female Pedigree', 'Female GBS']
                       ].rename(columns = {'Female Pedigree':'PedigreeClean', 'Female GBS':'Sample Names'}
   ).merge(geno2017.loc[:, ['Male Pedigree', 'Male GBS']
                       ].rename(columns = {'Male Pedigree':'PedigreeClean', 'Male GBS':'Sample Names'}
   ), how = 'outer').drop_duplicates()

geno2017['PedigreeClean'] = [re.sub("[*]", "_", str(entry).replace(' ', '').upper()) for entry in geno2017['PedigreeClean']]


geno2018 = geno2018.reset_index().drop(columns = ['index'])
geno2017 = geno2017.reset_index().drop(columns = ['index'])

# %% [markdown]
# ### Work with 2018 (genomic) data

# %%
temp = nP2018.loc[:, ['Pedigree']].drop_duplicates()
temp['PedigreeClean'] = [re.sub("[*]", "_", str(entry).replace(' ', '').upper()) for entry in temp['Pedigree']]
splitPedigrees = [str(entry).split('/') for entry in temp['PedigreeClean']]
splitPedigrees = pd.DataFrame(splitPedigrees, columns = ['F', 'M'])

uniqueParents = pd.DataFrame(splitPedigrees['F']).merge(pd.DataFrame(splitPedigrees.loc[:, 'M']
                                  ).rename(columns = {'M':'F'}), how = 'outer').drop_duplicates()
                             
uniqueParents = uniqueParents.rename(columns = {'F':'PedigreeClean'})
uniqueParents

# %%

# %%
uniqueParents2018 = uniqueParents.copy()

# %% [markdown]
# ### Add in 2017 (genomic) data

# %%
temp = nP2017.loc[:, ['Pedigree']].drop_duplicates()
temp['PedigreeClean'] = [re.sub("[*]", "_", str(entry).replace(' ', '').upper()) for entry in temp['Pedigree']]
splitPedigrees = [str(entry).split('/') for entry in temp['PedigreeClean']]
splitPedigrees = pd.DataFrame(splitPedigrees, columns = ['F', 'M'])


uniqueParents = pd.DataFrame(splitPedigrees['F']).merge(pd.DataFrame(splitPedigrees.loc[:, 'M']
                                  ).rename(columns = {'M':'F'}), how = 'outer').drop_duplicates()
                             
uniqueParents = uniqueParents.rename(columns = {'F':'PedigreeClean'})
uniqueParents


# %%
uniqueParents2018['GenotypeYear'] = '2018'
uniqueParents['GenotypeYear'] = '2017'

# %%
uniqueParents = pd.concat([uniqueParents, uniqueParents2018])

# %%
uniqueParents['Match2018'] = ''
uniqueParents['Match2017'] = ''
uniqueParents = uniqueParents.reset_index().drop(columns = ['index'])

# %%
uniqueParents['Match2018n'] = 0
uniqueParents['Match2017n'] = 0

for i in range(uniqueParents.shape[0]):
    match2018 = geno2018.loc[geno2018.PedigreeClean == uniqueParents.loc[i, 'PedigreeClean'], 'Sample Names']
    match2017 = geno2017.loc[geno2017.PedigreeClean == uniqueParents.loc[i, 'PedigreeClean'], 'Sample Names']

    match2018 = list(match2018)
    match2017 = list(match2017)

    uniqueParents.loc[i, 'Match2018n'] = len(match2018)#.shape[0]
    uniqueParents.loc[i, 'Match2017n'] = len(match2017)#.shape[0]

    if len(match2018) > 0:
        uniqueParents.loc[i, 'Match2018'] = match2018[0]

    if len(match2017) > 0:
        uniqueParents.loc[i, 'Match2017'] = match2017[0]


# %%
uniqueParents

# %%
mask = ((uniqueParents.PedigreeClean.notna()
    ) & (uniqueParents.PedigreeClean != ''
    ) & (uniqueParents.PedigreeClean != 'None'))
uniqueParents = uniqueParents.loc[mask, ]

uniqueParents['Matched'] = False
uniqueParents.loc[((uniqueParents.Match2017 != '') | (uniqueParents.Match2018 != '')), 'Matched'] = True


print('Within each genotyping year how many unique parents have gbs codes?\nNote: The same code may be in 2017 & 2018.')
uniqueParents.loc[:, ['GenotypeYear', 'Matched', 'PedigreeClean']
                 ].groupby(['GenotypeYear', 'Matched']
                 ).count(
                 ).reset_index()


# %% [markdown]
# ### Across the board who's missing?

# %%
missingParents = uniqueParents.loc[uniqueParents.Matched == False, ].drop(columns = ['GenotypeYear']).drop_duplicates()

# %%
missingParents = missingParents.sort_values('PedigreeClean')

missingParents

# %% code_folding=[79]
pedigreeReplacementDict = {
#     '1319YHR' :[''],
    '2369' :['2369.0'], # <-------------------- Replacement Found
#     '31G66' :[''],
#     '5618STXRIB' :[''],
    '740' :['740.0'], # <-------------------- Replacement Found
    '78010' :['DK78010'], # <-------------------- Replacement Found
#     '87916' :[''],
#     '9855HR' :[''],
    '???TX205' :['TX205'], # <-------------------- Replacement Found
#     'A7270G8' :[''],
#     'AGRATECH_903VIP' :[''],
#     'B47' :['PHB47'],
#     'B73XPHP02' :[''],
#     'BECK5140HR' :[''],
#     'BH8477SS' :[''],
#     'BH8900VIP31112016' :[''],
#     'CML442_CML343' :[''],
#     'CML450_TX110' :[''],
#     'CROPLAN3899VT2PRIB' :[''],
#     'CROPLAND3899VT2PRIB' :[''],
#     'D44VC36' :[''],
#     'DEKALB64-69' :[''],
#     'DEKALBDKC38-03' :[''],
#     'DEKALBDKC53-68GENSSRIB' :[''],
#     'DEKALBDKC60-67RIB' :[''],
#     'DEKALBDKC61-54RIB' :[''],
#     'DEKALB_64-35' :[''],
#     'DK3IIH6' :['3IIH6'],
#     'DK45-79' :[''],
#     'DKB64-69' :[''],
#     'DKC38-03RIB' :[''],
#     'DKC42-60RIB' :[''],
#     'DKC45-79' :[''],
#     'DKC46-61-RM96(RR2)' :[''],
#     'DKC52-62' :[''],
#     'DKC53-56RIB-RM103(SMARTSTAX)' :[''],
#     'DKC53-56RIB-SMARTSTAX-RM103' :[''],
#     'DKC53-68GENSSRIB' :[''],
#     'DKC53-78RIB-RM103(RR)' :[''],
#     'DKC60-67' :[''],
#     'DKC60-67RIB' :[''],
#     'DKC61-54RIB' :[''],
#     'DKC62-08' :[''],
#     'DKC62-20' :[''],
#     'DKC62-20RIB' :[''],
#     'DKC64-35RIB' :[''],
#     'DKC64-69' :[''],
#     'DKC65-19RIBGENVT3P' :[''],
#     'DKC65-71' :[''],
#     'DKC65-71RIB' :[''],
#     'DKC65-95' :[''],
#     'DKC66-29' :[''],
#     'DKC66-40RIB' :[''],
#     'DKC66-87' :[''],
#     'DKC67-44' :[''],
#     'DKC67-72' :[''],
#     'DKC67-72RIB' :[''],
#     'DKC67-88' :[''],
#     'DKC69-16' :[''],
#     'DKC69-72' :[''],
#     'DKC70-27' :[''],
#     'DKL46-611' :[''],
#     'DKPB80' :[''],
#     'DYNA-GRO_D58VC37' :[''],
#     'EX3101' :[''],
#     'EX4101' :[''],
#     'EX5101' :[''],
#     'FS6296VT3' :[''],
#     'G10D98' :[''],
#     'G12J11' :[''],
#     'G6611VTTP2017' :[''],
#     'G7601VTTP2017' :[''],
    'GEMS-0264' :['GEMN-0264'], # <-------------------- Replacement Found
    'GEMS-0265' :['GEMN-0265'], # <-------------------- Replacement Found
    'GEMS-0269' :['GEMN-0269'], # <-------------------- Replacement Found
#     'GREATLAKESGLH5283' :[''],
#     'LG5499' :[''],
#     'LG5618STXRIB' :[''],
#     'LG5643STXRIB' :[''],
#     'LGSEEDS5618STXRIB' :[''],
#     'LH119' :[''],
#     'LH119XPHP02' :[''],
#     'LH132XPHP02' :[''],
#     'LH195XPHP02' :[''],
#     'LOCALCHECK' :[''],
#     'LOCALCHECK1' :[''],
#     'LOCAL_CHECK' :[''],
#     'LOCAL_CHECK_1' :[''],
#     'LOCAL_CHECK_2' :[''],
#     'LOCAL_CHECK_3' :[''],
#     'LOCAL_CHECK_4' :[''],
#     'LOCAL_CHECK_5' :[''],
#     'M&W45M43' :[''],
#     'MAIZE' :[''],
#     'MAIZEXMZ4092DBR' :[''],
#     'MAIZEXMZ4525SMX' :[''],
#     'MBS3785' :[''],
#     'MBS5618' :[''],
#     'MBS5787HXT' :[''],
#     'MC4354VT2PRIB' :[''],
    # 'MO44_PHW65_0001' :[''],
    # 'MO44_PHW65_0010' :[''],
    # 'MO44_PHW65_0012' :[''],
    # 'MO44_PHW65_0029' :[''],
    # 'MO44_PHW65_0031' :[''],
    # 'MO44_PHW65_0056' :[''],
    # 'MO44_PHW65_0060' :[''],
    # 'MO44_PHW65_0067' :[''],
    # 'MO44_PHW65_0068' :[''],
    # 'MO44_PHW65_0073' :[''],
    # 'MO44_PHW65_0080' :[''],
    # 'MO44_PHW65_0104' :[''],
    # 'MO44_PHW65_0113' :[''],
    # 'MO44_PHW65_0116' :[''],
    # 'MO44_PHW65_0118' :[''],
    # 'MO44_PHW65_0120' :[''],
    # 'MO44_PHW65_0121' :[''],
    # 'MO44_PHW65_0128' :[''],
    # 'MO44_PHW65_0136' :[''],
    # 'MO44_PHW65_0144' :[''],
    # 'MO44_PHW65_0150' :[''],
    # 'MO44_PHW65_0151' :[''],
    # 'MO44_PHW65_0159' :[''],
    # 'MO44_PHW65_0164' :[''],
    # 'MO44_PHW65_0165' :[''],
    # 'MO44_PHW65_0187' :[''],
    # 'MO44_PHW65_0203' :[''],
    # 'MO44_PHW65_0218' :[''],
    # 'MO44_PHW65_0220' :[''],
    # 'MO44_PHW65_0252' :[''],
    # 'MO44_PHW65_0253' :[''],
    # 'MO44_PHW65_0256' :[''],
    # 'MO44_PHW65_0259' :[''],
    # 'MO44_PHW65_0260' :[''],
    # 'MO44_PHW65_0263' :[''],
    # 'MO44_PHW65_0271' :[''],
    # 'MO44_PHW65_0288' :[''],
    # 'MO44_PHW65_0321' :[''],
    # 'MO44_PHW65_0324' :[''],
    # 'MO44_PHW65_0333' :[''],
    # 'MO44_PHW65_0336' :[''],
    # 'MO44_PHW65_0343' :[''],
    # 'MO44_PHW65_0349' :[''],
    # 'MO44_PHW65_0358' :[''],
    # 'MO44_PHW65_0391' :[''],
    # 'MO44_PHW65_0392' :[''],
    # 'MO44_PHW65_0404' :[''],
    # 'MO44_PHW65_0416' :[''],
    # 'MO44_PHW65_0455' :[''],
    # 'MO44_PHW65_0471' :[''],
    # 'MO44_PHW65_0475' :[''],
#     'MZ2311DBR' :[''],
#     'MZ4092DBR' :[''],
#     'N585-3111' :[''],
    'N68B' :['B68'], # <-------------------- Replacement Found
#     'NAN' :[''],
    # 'NILASQ4G11I01S3' :[''],
    # 'NILASQ4G11I02S2' :[''],
    # 'NILASQ4G11I03S2' :[''],
    # 'NILASQ4G11I04S2' :[''],
    # 'NILASQ4G11I05S2' :[''],
    # 'NILASQ4G11I06S3' :[''],
    # 'NILASQ4G11I07S3' :[''],
    # 'NILASQ4G11I08S3' :[''],
    # 'NILASQ4G11I09S2' :[''],
    # 'NILASQ4G11I10S2' :[''],
    # 'NILASQ4G11I11S2' :[''],
    # 'NILASQ4G11I12S3' :[''],
    # 'NILASQ4G21I01S2' :[''],
    # 'NILASQ4G21I02S2' :[''],
    # 'NILASQ4G21I03S2' :[''],
    # 'NILASQ4G21I04S2' :[''],
    # 'NILASQ4G21I05S2' :[''],
    # 'NILASQ4G21I06S2' :[''],
    # 'NILASQ4G21I07S2' :[''],
    # 'NILASQ4G21I08S2' :[''],
    # 'NILASQ4G21I09S2' :[''],
    # 'NILASQ4G21I10S2' :[''],
    # 'NILASQ4G21I11S2' :[''],
    # 'NILASQ4G21I12S2' :[''],
    # 'NILASQ4G31I01S2' :[''],
    # 'NILASQ4G31I02S3' :[''],
    # 'NILASQ4G31I03S2' :[''],
    # 'NILASQ4G31I04S2' :[''],
    # 'NILASQ4G31I05S3' :[''],
    # 'NILASQ4G31I06S2' :[''],
    # 'NILASQ4G31I07S3' :[''],
    # 'NILASQ4G31I08S2' :[''],
    # 'NILASQ4G31I09S2' :[''],
    # 'NILASQ4G31I10S2' :[''],
    # 'NILASQ4G31I11S3' :[''],
    # 'NILASQ4G31I12S2' :[''],
    # 'NILASQ4G41I01S2' :[''],
    # 'NILASQ4G41I02S3' :[''],
    # 'NILASQ4G41I03S2' :[''],
    # 'NILASQ4G41I04S3' :[''],
    # 'NILASQ4G41I05S3' :[''],
    # 'NILASQ4G41I06S3' :[''],
    # 'NILASQ4G41I07S2' :[''],
    # 'NILASQ4G41I08S2' :[''],
    # 'NILASQ4G41I09S2' :[''],
    # 'NILASQ4G41I10S2' :[''],
    # 'NILASQ4G41I11S2' :[''],
    # 'NILASQ4G41I12S3' :[''],
    # 'NILASQ4G51I01S2' :[''],
    # 'NILASQ4G51I02S3' :[''],
    # 'NILASQ4G51I03S2' :[''],
    # 'NILASQ4G51I04S3' :[''],
    # 'NILASQ4G51I05S3' :[''],
    # 'NILASQ4G51I06S2' :[''],
    # 'NILASQ4G51I07S3' :[''],
    # 'NILASQ4G51I08S3' :[''],
    # 'NILASQ4G51I09S3' :[''],
    # 'NILASQ4G51I10S2' :[''],
    # 'NILASQ4G51I11S2' :[''],
    # 'NILASQ4G51I12S2' :[''],
    # 'NILASQ4G61I01S2' :[''],
    # 'NILASQ4G61I02S3' :[''],
    # 'NILASQ4G61I03S3' :[''],
    # 'NILASQ4G61I04S3' :[''],
    # 'NILASQ4G61I05S3' :[''],
    # 'NILASQ4G61I06S3' :[''],
    # 'NILASQ4G61I07S3' :[''],
    # 'NILASQ4G61I08S2' :[''],
    # 'NILASQ4G61I09S2' :[''],
    # 'NILASQ4G61I10S3' :[''],
    # 'NILASQ4G61I11S2' :[''],
    # 'NILASQ4G61I12S2' :[''],
    # 'NILASQ4G71I01S2' :[''],
    # 'NILASQ4G71I02S2' :[''],
    # 'NILASQ4G71I03S2' :[''],
    # 'NILASQ4G71I04S2' :[''],
    # 'NILASQ4G71I05S2' :[''],
    # 'NILASQ4G71I06S2' :[''],
    # 'NILASQ4G71I07S2' :[''],
    # 'NILASQ4G71I08S3' :[''],
    # 'NILASQ4G71I09S2' :[''],
    # 'NILASQ4G71I10S2' :[''],
    # 'NILASQ4G71I11S3' :[''],
    # 'NILASQ4G71I12S3' :[''],
#     'P0216' :[''],
#     'P0216AM' :[''],
#     'P0993HR' :[''],
#     'P1162YHR' :[''],
#     'P1197' :[''],
#     'P1197AM' :[''],
#     'P1221AMXT' :[''],
#     'P1311AMXT' :[''],
#     'P1498' :[''],
#     'P1498R' :[''],
#     'P1498YHR' :[''],
#     'P1916' :[''],
#     'P1916YHR' :[''],
#     'P2023HR' :[''],
#     'P2023R' :[''],
#     'P9188AM' :[''],
#     'P9411XR-RM94(RR2)' :[''],
#     'P9675AMXT' :[''],
#     'P9697AM~S' :[''],
#     'P9789AMXT' :[''],
#     'P9855HR' :[''],
#     'P9990XR-RM99(RR2)' :[''],
#     'PGM49' :[''],
#     'PH210' :[''],
#     'PHG39XPHP02' :[''],
    # 'PHN11_PHW65_0001' :[''],
    # 'PHN11_PHW65_0022' :[''],
    # 'PHN11_PHW65_0027' :[''],
    # 'PHN11_PHW65_0028' :[''],
    # 'PHN11_PHW65_0032' :[''],
    # 'PHN11_PHW65_0035' :[''],
    # 'PHN11_PHW65_0050' :[''],
    # 'PHN11_PHW65_0075' :[''],
    # 'PHN11_PHW65_0081' :[''],
    # 'PHN11_PHW65_0089' :[''],
    # 'PHN11_PHW65_0094' :[''],
    # 'PHN11_PHW65_0103' :[''],
    # 'PHN11_PHW65_0109' :[''],
    # 'PHN11_PHW65_0129' :[''],
    # 'PHN11_PHW65_0133' :[''],
    # 'PHN11_PHW65_0139' :[''],
    # 'PHN11_PHW65_0141' :[''],
    # 'PHN11_PHW65_0146' :[''],
    # 'PHN11_PHW65_0154' :[''],
    # 'PHN11_PHW65_0155' :[''],
    # 'PHN11_PHW65_0168' :[''],
    # 'PHN11_PHW65_0179' :[''],
    # 'PHN11_PHW65_0189' :[''],
    # 'PHN11_PHW65_0194' :[''],
    # 'PHN11_PHW65_0205' :[''],
    # 'PHN11_PHW65_0212' :[''],
    # 'PHN11_PHW65_0223' :[''],
    # 'PHN11_PHW65_0224' :[''],
    # 'PHN11_PHW65_0227' :[''],
    # 'PHN11_PHW65_0241' :[''],
    # 'PHN11_PHW65_0260' :[''],
    # 'PHN11_PHW65_0276' :[''],
    # 'PHN11_PHW65_0279' :[''],
    # 'PHN11_PHW65_0286' :[''],
    # 'PHN11_PHW65_0300' :[''],
    # 'PHN11_PHW65_0301' :[''],
    # 'PHN11_PHW65_0303' :[''],
    # 'PHN11_PHW65_0352' :[''],
    # 'PHN11_PHW65_0369' :[''],
    # 'PHN11_PHW65_0378' :[''],
    # 'PHN11_PHW65_0398' :[''],
    # 'PHN11_PHW65_0407' :[''],
    # 'PHN11_PHW65_0408' :[''],
    # 'PHN11_PHW65_0410' :[''],
    # 'PHN11_PHW65_0418' :[''],
    # 'PHN11_PHW65_0421' :[''],
    # 'PHN11_PHW65_0431' :[''],
    # 'PHN11_PHW65_0443' :[''],
    # 'PHN11_PHW65_0451' :[''],
    # 'PHN11_PHW65_0453' :[''],
    # 'PHN11_PHW65_0478' :[''],
    # 'PHN11_PHW65_0486' :[''],
    # 'PHN11_PHW65_0494' :[''],
    # 'PHN11_PHW65_0499' :[''],
    # 'PHN11_PHW65_0505' :[''],
    # 'PHN11_PHW65_0514' :[''],
    # 'PHN11_PHW65_0541' :[''],
    # 'PHN11_PHW65_0548' :[''],
    # 'PHN11_PHW65_0552' :[''],
    # 'PHN11_PHW65_0577' :[''],
    # 'PHN11_PHW65_0591' :[''],
    # 'PHN11_PHW65_0604' :[''],
    # 'PHN11_PHW65_0612' :[''],
    # 'PHN11_PHW65_0614' :[''],
    # 'PHN11_PHW65_0626' :[''],
    # 'PHN11_PHW65_0637' :[''],
    # 'PHN11_PHW65_0639' :[''],
    # 'PHN11_PHW65_0663' :[''],
#     'PHOENIX_7402A3' :[''],
    # 'PHW65_MOG_0003' :[''],
    # 'PHW65_MOG_0006' :[''],
    # 'PHW65_MOG_0009' :[''],
    # 'PHW65_MOG_0014' :[''],
    # 'PHW65_MOG_0015' :[''],
    # 'PHW65_MOG_0016' :[''],
    # 'PHW65_MOG_0021' :[''],
    # 'PHW65_MOG_0025' :[''],
    # 'PHW65_MOG_0032' :[''],
    # 'PHW65_MOG_0034' :[''],
    # 'PHW65_MOG_0038' :[''],
    # 'PHW65_MOG_0040' :[''],
    # 'PHW65_MOG_0048' :[''],
    # 'PHW65_MOG_0051' :[''],
    # 'PHW65_MOG_0052' :[''],
    # 'PHW65_MOG_0054' :[''],
    # 'PHW65_MOG_0055' :[''],
    # 'PHW65_MOG_0056' :[''],
    # 'PHW65_MOG_0059' :[''],
    # 'PHW65_MOG_0060' :[''],
    # 'PHW65_MOG_0061' :[''],
    # 'PHW65_MOG_0062' :[''],
    # 'PHW65_MOG_0063' :[''],
    # 'PHW65_MOG_0064' :[''],
    # 'PHW65_MOG_0067' :[''],
    # 'PHW65_MOG_0068' :[''],
    # 'PHW65_MOG_0069' :[''],
    # 'PHW65_MOG_0071' :[''],
    # 'PHW65_MOG_0072' :[''],
    # 'PHW65_MOG_0075' :[''],
    # 'PHW65_MOG_0086' :[''],
    # 'PHW65_MOG_0087' :[''],
    # 'PHW65_MOG_0089' :[''],
    # 'PHW65_MOG_0098' :[''],
    # 'PHW65_MOG_0106' :[''],
    # 'PHW65_MOG_0108' :[''],
    # 'PHW65_MOG_0109' :[''],
    # 'PHW65_MOG_0115' :[''],
    # 'PHW65_MOG_0116' :[''],
    # 'PHW65_MOG_0117' :[''],
    # 'PHW65_MOG_0118' :[''],
    # 'PHW65_MOG_0119' :[''],
    # 'PHW65_MOG_0121' :[''],
    # 'PHW65_MOG_0124' :[''],
    # 'PHW65_MOG_0125' :[''],
    # 'PHW65_MOG_0127' :[''],
    # 'PHW65_MOG_0128' :[''],
    # 'PHW65_MOG_0129' :[''],
    # 'PHW65_MOG_0132' :[''],
    # 'PHW65_MOG_0133' :[''],
    # 'PHW65_MOG_0139' :[''],
    # 'PHW65_MOG_0140' :[''],
    # 'PHW65_MOG_0143' :[''],
    # 'PHW65_MOG_0145' :[''],
    # 'PHW65_MOG_0146' :[''],
    # 'PHW65_MOG_0148' :[''],
    # 'PHW65_MOG_0150' :[''],
    # 'PHW65_MOG_0152' :[''],
    # 'PHW65_MOG_0153' :[''],
    # 'PHW65_MOG_0155' :[''],
    # 'PHW65_MOG_0156' :[''],
    # 'PHW65_MOG_0157' :[''],
    # 'PHW65_MOG_0160' :[''],
    # 'PHW65_MOG_0162' :[''],
    # 'PHW65_MOG_0173' :[''],
    # 'PHW65_MOG_0175' :[''],
    # 'PHW65_MOG_0178' :[''],
    # 'PHW65_MOG_0180' :[''],
    # 'PHW65_MOG_0181' :[''],
    # 'PHW65_MOG_0183' :[''],
    # 'PHW65_MOG_0190' :[''],
    # 'PHW65_MOG_0191' :[''],
    # 'PHW65_MOG_0192' :[''],
    # 'PHW65_MOG_0194' :[''],
    # 'PHW65_MOG_0195' :[''],
    # 'PHW65_MOG_0199' :[''],
    # 'PHW65_MOG_0200' :[''],
    # 'PHW65_MOG_0203' :[''],
    # 'PHW65_MOG_0206' :[''],
    # 'PHW65_MOG_0209' :[''],
    # 'PHW65_MOG_0213' :[''],
    # 'PHW65_MOG_0218' :[''],
    # 'PHW65_MOG_0221' :[''],
    # 'PHW65_MOG_0223' :[''],
    # 'PHW65_MOG_0232' :[''],
    # 'PHW65_MOG_0236' :[''],
    # 'PHW65_MOG_0241' :[''],
    # 'PHW65_MOG_0242' :[''],
    # 'PHW65_MOG_0250' :[''],
    # 'PHW65_MOG_0253' :[''],
    # 'PHW65_MOG_0268' :[''],
    # 'PHW65_MOG_0287' :[''],
    # 'PHW65_MOG_0291' :[''],
    # 'PHW65_MOG_0296' :[''],
    # 'PHW65_MOG_0297' :[''],
    # 'PHW65_MOG_0300' :[''],
    # 'PHW65_MOG_0307' :[''],
    # 'PHW65_MOG_0310' :[''],
    # 'PHW65_MOG_0314' :[''],
    # 'PHW65_MOG_0315' :[''],
    # 'PHW65_MOG_0316' :[''],
    # 'PHW65_MOG_0321' :[''],
    # 'PHW65_MOG_0326' :[''],
    # 'PHW65_MOG_0332' :[''],
    # 'PHW65_MOG_0335' :[''],
    # 'PHW65_MOG_0340' :[''],
    # 'PHW65_MOG_0341' :[''],
    # 'PHW65_MOG_0343' :[''],
    # 'PHW65_MOG_0350' :[''],
    # 'PHW65_MOG_0352' :[''],
    # 'PHW65_MOG_0354' :[''],
    # 'PHW65_MOG_0359' :[''],
    # 'PHW65_MOG_0375' :[''],
    # 'PHW65_MOG_0379' :[''],
    # 'PHW65_MOG_0381' :[''],
    # 'PHW65_MOG_0384' :[''],
    # 'PHW65_MOG_0388' :[''],
    # 'PHW65_MOG_0392' :[''],
    # 'PHW65_MOG_0393' :[''],
    # 'PHW65_MOG_0410' :[''],
    # 'PHW65_MOG_0420' :[''],
    # 'PHW65_MOG_0423' :[''],
    # 'PHW65_MOG_0424' :[''],
    # 'PHW65_MOG_0426' :[''],
    # 'PHW65_MOG_0438' :[''],
    # 'PHW65_MOG_0439' :[''],
    # 'PHW65_MOG_0444' :[''],
    # 'PHW65_MOG_0448' :[''],
    # 'PHW65_MOG_0449' :[''],
    # 'PHW65_MOG_0461' :[''],
    # 'PHW65_MOG_0463' :[''],
    # 'PHW65_MOG_0467' :[''],
    # 'PHW65_MOG_0476' :[''],
    # 'PHW65_MOG_0490' :[''],
    # 'PHW65_MOG_0512' :[''],
    # 'PHW65_MOG_0513' :[''],
    # 'PHW65_MOG_0517' :[''],
    # 'PHW65_MOG_0522' :[''],
    # 'PHW65_MOG_0534' :[''],
    # 'PHW65_MOG_0544' :[''],
    # 'PHW65_MOG_0545' :[''],
    # 'PHW65_MOG_0551' :[''],
    # 'PHW65_MOG_0553' :[''],
    # 'PHW65_MOG_0554' :[''],
    # 'PHW65_MOG_0560' :[''],
    # 'PHW65_MOG_0575' :[''],
    # 'PHW65_MOG_0583' :[''],
    # 'PHW65_MOG_0591' :[''],
    # 'PHW65_MOG_0599' :[''],
    # 'PHW65_MOG_0620' :[''],
    # 'PHW65_MOG_0629' :[''],
    # 'PHW65_MOG_0633' :[''],
    # 'PHW65_MOG_0651' :[''],
    # 'PHW65_MOG_0660' :[''],
    # 'PHW65_MOG_0662' :[''],
    # 'PHW65_MOG_0675' :[''],
    # 'PHW65_MOG_0696' :[''],
    # 'PHW65_MOG_0716' :[''],
    # 'PIONEER1319YHR' :[''],
    # 'PIONEER1637YHR' :[''],
    # 'PIONEER2160YHR' :[''],
    # 'PIONEERP0157AM' :[''],
    # 'PIONEERP0216AM' :[''],
    # 'PIONEERP0993HR' :[''],
    # 'PIONEERP9855HR' :[''],
#     'PPHP38' :[''],
#     'PRIDEA7270G8' :[''],
#     'REV28HR202016' :[''],
#     'SCS1085YHR' :[''],
#     'TERRRAL_REV25BHR26' :[''],
#     'TROPHYHYBRID' :[''],
#     'TX110' :[''],
#     'TX601' :[''],
#     'TX775' :[''],
#     'TX779' :['']

    }

# exclude those already examined. 
# fuzz search
# examine
    # newDict

# %%

# %%
missingParents = missingParents.reset_index().drop(columns = 'index')

# %%

# %%
# Use the above dict to update the names which have been manually identified.
missingParents['PedigreeIgnore'] = ''
for key in pedigreeReplacementDict.keys():
    mask = (missingParents.PedigreeClean == key)
    missingParents.loc[mask, 'PedigreeIgnore'] = missingParents.loc[mask, 'PedigreeClean'] 
    missingParents.loc[mask, 'PedigreeClean'] = pedigreeReplacementDict[key]


# %%

# %%
missingParentsList = list(missingParents.loc[missingParents.PedigreeClean != 'NoGoodReplacement', 'PedigreeClean'])

parentsWithGBSCodesList =  list(geno2018.PedigreeClean) + [entry for entry in list(geno2017.PedigreeClean) if entry not in list(geno2018.PedigreeClean)]

foundParentsList = [entry for entry in missingParentsList if entry in parentsWithGBSCodesList]
missingParentsList = [entry for entry in missingParentsList if entry not in parentsWithGBSCodesList]


print('Out of ', len(list(missingParents.PedigreeClean)), ' parents,'
      'ignoring entries where no good replacement was found:\n', 
      len(foundParentsList), ' matched following update.\n', 
      len(missingParentsList), ' still to be matched or removed.', 
      sep = '')

# %%
## searching goes here ----
from fuzzywuzzy import fuzz, process
takeTopX = 10
minRatio = 1#40

# i = 0
matchingDict = dict()
for i in range(len(missingParentsList)):
    fuzzScores = [fuzz.ratio(missingParentsList[i], str(possibleMatch)) for possibleMatch in parentsWithGBSCodesList]
    res = pd.DataFrame(zip(parentsWithGBSCodesList, fuzzScores), columns = ['PossibleMatch', 'Score'])
    res = res.sort_values('Score', ascending = False)
    res = res.head(takeTopX)
    res = res.loc[res.Score >= minRatio, ]
    res = list(res['PossibleMatch'])
    res = res + ['' for i in range(takeTopX - len(res))]
    matchingDict.update({missingParentsList[i]: res})

# fuzz.ratio(unmatchedParents.loc[i, 'FClean'], str(gbs))]

# [[str(gbs), fuzz.ratio(unmatchedParents.loc[i, 'FClean'], str(gbs))] for gbs in gbsLookup.ParentClean if fuzz.ratio(unmatchedParents.loc[i, 'FClean'], str(gbs)) > threshold]

# %%
res = pd.DataFrame().from_dict(matchingDict).T
# res.to_clipboard()
res.head(5)

# %%
# pd.DataFrame().from_dict(matchingDict)

# %%
#findme!

# %%

# %%
### Prepare data for one-hot encoding or for gbs code.

# %%
# Allow all entries with gbs data
foundParentsList

# also allow a subset of the missing ones. 
missingParentsList

# %%

# %%
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Disallowed Parents: #
# Anything we suspect we'll not get genomic data for goes here.

# this and the next part are critical the above informs what we retain.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

dropIfParentIs = [
    'LOCALCHECK',
    'LOCALCHECK1',
    'LOCALCHECKDE',
    'LOCALCHECKIA1C',
    'LOCALCHECKNE2',
    'LOCALCHECKNE3',
    'LOCALCHECKTX2',
    'LOCAL_CHECK',
    'LOCAL_CHECK1',
    'LOCAL_CHECK2',
    'LOCAL_CHECK3',
    'LOCAL_CHECK4',
    'LOCAL_CHECK5',
    'LOCAL_CHECK_1',
    'LOCAL_CHECK_2',
    'LOCAL_CHECK_3',
    'LOCAL_CHECK_4',
    'LOCAL_CHECK_5',
    'MAIZE',
    'MAIZEXMZ4092DBR',
    'MAIZEXMZ4525SMX',
    'NAN',
]



# %%

# %%
temp = newPhenotype.loc[:, 'Pedigree'].drop_duplicates()
temp = pd.DataFrame(temp)
temp = temp.reset_index().drop(columns = ['index'])

temp['PedigreeClean'] = [re.sub("[*]", "_", str(entry).replace(' ', '').upper()) for entry in temp['Pedigree']]
splitPedigrees = [str(entry).split('/') for entry in temp['PedigreeClean']]
splitPedigrees = pd.DataFrame(splitPedigrees, columns = ['F', 'M'])
temp['F'] = splitPedigrees['F']
temp['M'] = splitPedigrees['M']

# fix inbred line parsing:
temp.loc[temp.M.isna(), 'M'] = temp.loc[temp.M.isna(), 'F']
temp = temp.loc[temp.PedigreeClean != 'NAN', ]

temp['FPass'] = False
temp['MPass'] = False
temp = temp.reset_index().drop(columns = ['index'])

okayParents = pd.DataFrame(temp['F']
                          ).merge(pd.DataFrame(temp['M']
                                 ).rename(columns = {'M':'F'}), how = 'outer'
                          ).drop_duplicates(
                          ).reset_index(
                          ).drop(columns = ['index'])

# filter out the unwanted parents from the okay df.
# note that we're using a df here to make sure we don't have to fuss with dtypes as much
for i in range(len(dropIfParentIs)):
    okayParents = okayParents.loc[okayParents.F != dropIfParentIs[i], ]

for i in range(temp.shape[0]):
    if sum(okayParents.F == str(temp.loc[i, 'F'])) == 1:
        temp.loc[i, 'FPass'] = True
    if sum(okayParents.F == str(temp.loc[i, 'M'])) == 1:
        temp.loc[i, 'MPass'] = True
    

temp['AllPass'] = False
mask = (temp.FPass & temp.MPass)
temp.loc[mask, 'AllPass'] = True

# %%
print('Here is the split of passing/failing pedigree entries:')
temp.loc[:, ['Pedigree', 'AllPass']].groupby(['AllPass']).count().reset_index()

# %%
temp = temp.loc[temp.AllPass, :]

# %%
temp = temp.loc[:, ['Pedigree', 'F', 'M']]

# filtering -- 
newPhenotype = temp.merge(newPhenotype)

# %%
newPhenotype

# %%

# %%
expObsTrackerDf = track_newPheno_grainyield_obs(
    df = newPhenotype,
    markerName = '4.0.3',
    expObsTrackerDf = expObsTrackerDf
)

# %% [markdown]
# # Apply any needed interpolation/filtering to newly merged data.

# %% [markdown]
# ## General

# %%
print('Replacing TXH4 with Lubbock TX cooridnates. (was set to near madison WI)\n')
print('---------- Before ---------')
print(newMetadata.loc[newMetadata.ExperimentCode == 'TXH4', ['ExpLon', 'ExpLat']])

newMetadata.loc[newMetadata.ExperimentCode == 'TXH4', 'ExpLon'] = -101.8552
newMetadata.loc[newMetadata.ExperimentCode == 'TXH4', 'ExpLat'] =   33.5779

print('---------- After ----------')
print(newMetadata.loc[newMetadata.ExperimentCode == 'TXH4', ['ExpLon', 'ExpLat']])

# %%
newMetadata.loc[newMetadata.ExperimentCode.isna(), ]

# %%
# Fill in missing gps codes
gpsGrouping = {
    'AZI1': ['AZH1', 'AZI1  AZI2'],
    'GAH1': ['GAI2'],
    
    'GEH1': ['GEH1', 'GEH2'], # GEH2 is missing here

    
    'IAH1': ['IAH1 IAI1', 'IAH1a', 'IAH1b', 'IAH1c'],
    'IAH2': ['IA(?)2', 'IAH2 '],
    'IAH3': ['IA(?)3', 'IAH3 '],
    'IAH4': ['IA(H4)','IAH4 '],
    'ILH1': ['ILH1  ILI1  ILH2', 'ILH1 ILI1'],
    'INH1': ['INH1  INI1', 'INH1 INI1'],
    'KSH1': ['KSH1  KSI1', 'KSH2', 'KSH3'],
    'MNH1': ['MN(?)1', 'MNI2'],
    'MOH1': ['MOH1  MOI1  MOH2  MOI2', 'MOH1 MOI1','MOH1 '],
    'MOH2': ['MOH2 MOI2 MOI3'],
    'NEH1': ['NEH1  NEH4', 'NEH1 NEI1'],
    'NEH3': ['NEH3-Irrigated'],
    'NEH4': ['NEH4-NonIrrigated'],
    'NYH1': ['NY?', 'NYH1  NYI1', 'NYH1 NYI1', 'NYI1'],
    'NYH2': ['NYH3  NYI2', 'NYH4'],
    'PAI1': ['PAI1  PAI2'],
    'SDH1': ['SDI1'],
    'TXH1': ['TXH1  TXI1  TXI2'],
    'TXH2': ['TXH2  TXI3', 'TXI3'],
    'WIH1': ['WIH1  WII1', 'WIH1 WII1', 'W1H1'],
    'WIH2': ['WIH2  WII2', 'W1H2'],
}
  
with open('../data/interim/gpsGrouping.txt', 'w') as convert_file:
     convert_file.write(json.dumps(gpsGrouping))

# %%
# for every missing experiment code, add in the same experiment code from a different year. 
# If that doesn't exist then fill in the average of the relevant key from gpsGroupings

meanExpGps = newMetadata.groupby(['ExperimentCode']
                        ).agg(mExpLat = ('ExpLat', 'mean'), 
                              mExpLon = ('ExpLon', 'mean')
                        ).reset_index(
                        ).dropna(subset = ['mExpLat']
                        ).reset_index(
                        ).drop(columns = ['index'])
meanExpGps

needGps = newMetadata.loc[newMetadata.ExpLon.isna(), ['ExperimentCode', 'Year']
                         ].reset_index(
                         ).drop(columns = ['index'])

for i in range(needGps.shape[0]):
    curExp  = needGps.loc[i, 'ExperimentCode']
    curYear = needGps.loc[i, 'Year']

    repGps = meanExpGps.loc[meanExpGps.ExperimentCode == curExp, :]
    
    repExp = curExp
    if repGps.shape[0] != 1:
        
        for key in gpsGrouping.keys():
            if curExp in gpsGrouping[key]:
                repExp = key

        if repExp == curExp:
            print('No replacment found for ', curExp)
            
            
            
        else:
            repGps = newMetadata.loc[newMetadata.ExperimentCode == repExp, 
                                    ].groupby(['ExperimentCode']
                                    ).agg(mExpLat = ('ExpLat', 'mean'), 
                                          mExpLon = ('ExpLon', 'mean'))

    if repGps.shape[0] == 1:
        print('Replacing', curExp, curYear, 'with', repExp, 'average.')
        mask = (newMetadata.ExperimentCode == curExp) & (newMetadata.Year == curYear)
        newMetadata.loc[mask, 'ExpLon'] = float(repGps['mExpLon'])
        newMetadata.loc[mask, 'ExpLat'] = float(repGps['mExpLat'])


# %%
print('This should be empty if all values have been filled in.')
newMetadata.loc[newMetadata.ExpLon.isna(), :]

# %%
# Which experiment codes are in which datasets?

expsFound = newMetadata.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']]

expsFound['inMetadata'] = True


temp = newManagement.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()
temp['inManagement'] = True
expsFound = expsFound.merge(temp, how = 'outer')


temp = newPhenotype.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()
temp['inPhenotype'] = True
expsFound = expsFound.merge(temp, how = 'outer')


temp = newSoil.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()
temp['inSoil'] = True
expsFound = expsFound.merge(temp, how = 'outer')


temp = newWeather.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()
temp['inWeather'] = True
expsFound = expsFound.merge(temp, how = 'outer')

expsFound = expsFound.merge(temp, how = 'outer')

expsFound#.to_clipboard()
# list(expsFound.ExperimentCode.drop_duplicates().sort_values())
# newPhenotype
# newSoil
# newWeather

# entry for entry in list(newManagement.ExperimentCode.drop_duplicates()) not in 

# %%
# Drop any disallowed experiments
# i.e. those without daymet data

for disallowedExp in ['GEH1', 'GEH2']:
    newPhenotype =  newPhenotype.loc[newPhenotype.ExperimentCode != disallowedExp, ]
    newMetadata =   newMetadata.loc[newMetadata.ExperimentCode != disallowedExp, ]
    newSoil =       newSoil.loc[newSoil.ExperimentCode != disallowedExp, ]
    newManagement = newManagement.loc[newManagement.ExperimentCode != disallowedExp, ]
    newWeather =    newWeather.loc[newWeather.ExperimentCode != disallowedExp, ]

# %%

# %%

# %%
# temp = newMetadata.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']] # <--

# temp = newManagement.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()
# temp = newPhenotype.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()
# temp = newSoil.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()       # <--
# temp = newWeather.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()

# temp.loc[temp.ExperimentCode == 'nan' ]


# %%
expObsTrackerDf = track_newPheno_grainyield_obs(
    df = newPhenotype,
    markerName = '5.1',
    expObsTrackerDf = expObsTrackerDf
)

# %% [markdown]
# ## Phenotype

# %%
# infer missing planting dates

# %%
newPhenotype

# %%
print('Planting day of year and missing values (red)')
newPhenotype.DatePlanted = pd.to_datetime(newPhenotype.DatePlanted)
newPhenotype.DateHarvested = pd.to_datetime(newPhenotype.DateHarvested)

fig, axs = plt.subplots(1, 2)  
fig.set_size_inches(18.5, 10.5)

axs[0].scatter(newPhenotype.DatePlanted.dt.day_of_year, newPhenotype.ExperimentCode)


axs[1].scatter(newPhenotype.Year, newPhenotype.ExperimentCode)
temp = newPhenotype.loc[newPhenotype.DatePlanted.isna(), ['ExperimentCode', 'Year']].drop_duplicates()
axs[1].scatter(temp.Year, temp.ExperimentCode, color = 'red')

# %%
print('Known planting dates for SCH1', list(newPhenotype.loc[newPhenotype.ExperimentCode == 'SCH1', 'DatePlanted'].drop_duplicates()))

# %%
print('If we guess a central value 2017+~', (pd.to_datetime('2018-04-20 00:00:00') - pd.to_datetime('2018-05-13 00:00:00'))/2, ' is likely reasonable')
print('ie', pd.to_datetime('2017-04-20 00:00:00')+timedelta(13))

newPhenotype.loc[(newPhenotype.DatePlanted.isna()) & (newPhenotype.Year == '2017'), 'DatePlanted'] = pd.to_datetime('2017-04-20 00:00:00')+timedelta(13)

print('SCH1 2017 planting date imputed.')

# %%
plt.hist(newPhenotype.DatePlanted, bins = 12*5)
print()


# %%
expObsTrackerDf = track_newPheno_grainyield_obs(
    df = newPhenotype,
    markerName = '5.2',
    expObsTrackerDf = expObsTrackerDf
)

# %% [markdown]
# ## Soil

# %%
newSoil

# %%
temp = newSoil.loc[:, [
    #  'SoilTaxonomicId',
 'SoilpH', # <- 
 'CationExchangeCapacity', # <- 
    #  'TextureMineralFraction', # <- 
#  'SampleType', # <- 
#  'SoilTextType', # <- 
#  'SoilTexture' # <- 
]]

for col in list(temp):
    print(col)
    print(list(temp.loc[:, col].drop_duplicates()))
    print('')

# %%
col = 'SoilpH'
tempValues = newSoil.loc[:, col].drop_duplicates()
print('replacing the following values')
print([entry for entry in tempValues if not convertable_to_float(entry)])

searchString = '1:2 soil:water 7'
newSoil.loc[newSoil[col] == searchString, col] = 7

searchString = '1:2 soil:water 6.5'
newSoil.loc[newSoil[col] == searchString, col] = 6.5

searchString = 'not recently tested'
newSoil.loc[newSoil[col] == searchString, col] = np.nan

searchString = 'soil sample submitted 6/11/15'
newSoil.loc[newSoil[col] == searchString, col] = np.nan

searchString = 'PH = 8.00'
newSoil.loc[newSoil[col] == searchString, col] = 8


# %%
col = 'CationExchangeCapacity'
tempValues = newSoil.loc[:, col].drop_duplicates()
print('replacing the following values')
print([entry for entry in tempValues if not convertable_to_float(entry)])


searchString = '14.05 cmolc/kg'
newSoil.loc[newSoil[col] == searchString, col] = 14.05

searchString = '9 cmocc/kg' 
newSoil.loc[newSoil[col] == searchString, col] = 9

searchString = '296 umho/cm'
newSoil.loc[newSoil[col] == searchString, col] = np.nan

searchString = '269 umho/cm'
newSoil.loc[newSoil[col] == searchString, col] = np.nan

# %%

# %%
soilDataCols = [
#  'SoilTaxonomicId',
#  'Grower',
#  'DateReceived',
#  'DateReported',
 'EDepth',
 'SoilpH', # <- 
 'WDRFpH',
 'SSalts',
 'TexturePSA',
 'PercentOrganic',
 'ppmNitrateN',
 'NitrogenPerAcre',
 'ppmK',
 'ppmSulfateS',
 'ppmCa',
 'ppmMg',
 'ppmNa',
 'CationExchangeCapacity', # <- 
 'PercentH',
 'PercentK',
 'PercentCa',
 'PercentMg',
 'PercentNa',
 'ppmP',
 'PercentSand',
 'PercentSilt',
 'PercentClay',
#  'TextureMineralFraction', # <- 
#  'SampleType', # <- 
 'ppmZn',
 'ppmFe',
 'ppmMn',
 'ppmCu',
 'ppmB',
#  'SoilSampleDate',
#  'LabId',
#  'LabSampleId',
 'PlowDepth',
 'SikorapH',
#  'Location',
#  'SoilTextType', # <- 
#  'SoilTexture' # <- 
                  ]

for col in soilDataCols:
    print(col)
    try:
        newSoil.loc[:, col].astype(float)
#         print('Can be converted to float.')
    except:
        print('------- Conversion failed.')

# %%
for col in soilDataCols:
#     print(col)
    try:
        newSoil.loc[:, col] = newSoil.loc[:, col].astype(float)
#         print('Can be converted to float.')
    except:
        print('------- Conversion failed.')

# %%

# %%

# %%

# %%
# We can use the nearest location a to interpolate missing values
# (e.g. 'AZI1' & 'AZI1 AZI2' have diffeing names but the same coordinates)

meanLonLat = newMetadata.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']].groupby('ExperimentCode').agg('mean').reset_index()
# distanceLon = 
# distanceLat = distanceLon.copy().drop(columns = ['ExpLon'])
# distanceLon = distanceLon.drop(columns = ['ExpLat'])

# np.matrix(np.nan, row = 4, col = 4)


# meanLonLat.shape[0]
nExp = meanLonLat.shape[0]
distanceMatrix = pd.DataFrame(np.zeros((nExp, nExp)))


for i in range(nExp):
    for j in range(nExp):
        if i == j:
            distanceMatrix.loc[i, j] = np.nan
        else:
            lonDif = meanLonLat.loc[i, 'ExpLon'] - meanLonLat.loc[j, 'ExpLon']
            latDif = meanLonLat.loc[i, 'ExpLat'] - meanLonLat.loc[j, 'ExpLat']

            distanceMatrix.loc[i, j] = np.sqrt((lonDif**2) + (latDif**2))

# distanceMatrix

# %%
# this is to help in choosing which experiment codes can be grouped together
# Requires these global parameters! Only trust if used right after defining them.
def lookupNearestExp(
    lookupExpCode = 'ARH1'
):
    i = meanLonLat.loc[meanLonLat.ExperimentCode == lookupExpCode, :].index[0]
    # meanLonLat.loc[i, :]
    distance = distanceMatrix.loc[i, :].min()
    nearestExp = meanLonLat.loc[distanceMatrix.loc[i, :].idxmin(), 'ExperimentCode']
    
    return([nearestExp, distance])

# lookupNearestExp()

meanLonLat['NearestExp'] = ''
meanLonLat['Distance'] = np.nan
for i in range(nExp):
    res = lookupNearestExp(lookupExpCode = meanLonLat.loc[i, 'ExperimentCode'])
    meanLonLat.loc[i, 'NearestExp'] = res[0]
    meanLonLat.loc[i, 'Distance'] = res[1]

meanLonLat

# %%

plt.plot(meanLonLat.Distance)


# %%
# set up a grouping variable to use -- inspect distances

# meanLonLat.loc[meanLonLat.Distance < 0.1, ].to_clipboard()

# %%
def ifColIn(
    df,
    colList
):
    return([col for col in list(df) if col in colList])

soilCols = [
    'SoilpH',                 'WDRFpH',      'SSalts',   'TexturePSA',  'PercentOrganic',
    'CationExchangeCapacity', 'PercentH',    'PercentK', 'PercentCa',   'PercentMg',
    'PercentNa',              'ppmNitrateN', 'ppmK',     'ppmSulfateS', 'ppmCa', 
    'ppmMg',                  'ppmNa',       'ppmP',     'PercentSand', 'PercentSilt',
    'PercentClay']

interpolateSoil = newMetadata.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']].merge(newSoil, how = 'right')

# interpolateSoil

# %%

# %%
## set up a grouping variable to use ====
groupingDict = {
'gAZ': ['AZH1', 'AZI1', 'AZI1  AZI2', 'AZI2'],  # <--------------------------------------------
'gDE': ['DEI1',    'G2FDE1'],
'gGA': ['GAH1','GAI1','GAH2','GAI2','GXE_inb_IA2','IAI1'],  
'gIA_2': ['IA(?)2','IAH2'],
'gIA_3': ['IA(?)3','IAH3', 'IAI3', 'IAI4','GXE_inb_IA1','G2FIA3'],
'gIA_4': ['IA(H4)','IAH4'],
'gIA_1': ['IAH1','IAH1a','IAH1b','IAH1c','IAH1 IAI1', 'IAI2'],
'gIL': ['ILH1 ILI1','ILH2','ILH1', 'ILH1  ILI1  ILH2', 'ILI1','G2FIL1'], # <--------------------------------------------
'gIN': ['INH1', 'INH1  INI1', 'INH1 INI1', 'INI1', 'G2FIN1'], # <--------------------------------------------
'gKS': ['KSH1  KSI1', 'KSH2', 'KSH3', 'KSI1', 'KSH1'],
'gMN': ['MNI1','G2FMN2','MNI2','MN(?)1','MNH1'], # <--------------------------------------------
'gMO': ['GXE_inb_MO3','GXE_inb_BO2', 'MOH1', 'MOH1  MOI1  MOH2  MOI2', 'MOH1 MOI1', 'GXE_inb_MO1', 'MOH1-rep1', 'MOH1-rep2','MOI1', 'MOI2', 'MOH2', 'MOH2 MOI2 MOI3'], 
'gNC': ['NC1', 'NCI1'],  # <--------------------------------------------
'gNE': ['NEH1', 'NEH1  NEH4', 'NEH1 NEI1', 'G2FNE1', 'NEH2', 'NEH3', 'NEH3-Irrigated', 'NEH4', 'NEH4-NonIrrigated'], 
'gNY_1': ['NY?','NYH1','NYH1  NYI1', 'NYH1 NYI1', 'G2FNY1', 'NYI1'],
'gNY_2': ['NYH2','NYH3  NYI2','NYH3','NYH4', 'NYI2'],
'gPA': ['PAI1  PAI2','PAI2', 'GxE_inb_PA1', 'PAI1'],
'gSD': ['SDH1','SDI1'], # <--------------------------------------------   
'gTX': ['TXH1','TXH1-Dry','TXH1-Late','TXH1-Early', 'TXH2  TXI3',
        'TXI3', 'TXH1  TXI1  TXI2', 'TXI1', 'TXH2','G2F_IN_TX1','TXI2'],
'gWI_1': ['WIH1','WIH1 WII1','WIH1  WII1','WII1','G2FWI1'],
'gWI_2': ['WIH2','WIH2  WII2', 'WII2','G2FWI2', 'G2FWI-HYB']
}

# %%

# %%
# list(interpolateSoil.ExperimentCode.drop_duplicates().sort_values())

interpolateSoil['Groups'] = interpolateSoil['ExperimentCode']

for key in groupingDict.keys():
    interpolateSoil['Groups'] = [entry if entry not in groupingDict[key] else key for entry in interpolateSoil['Groups'] ]

# %%
## Actually perform the interpolation ====

# meanLonLat.loc[meanLonLat.Distance < 1, :]

expCodes = list(interpolateSoil.Groups.drop_duplicates())

expCode = expCodes[0]
for expCode in expCodes:
    temp = interpolateSoil.loc[interpolateSoil.Groups == expCode, ]
    interpolateSoil = interpolateSoil.loc[interpolateSoil.Groups != expCode, ]

    tempRef = temp.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']]
    temp = temp.drop(columns = ['ExperimentCode', 'ExpLon', 'ExpLat'])
    temp = temp.groupby(['Groups','Year']).agg('mean').reset_index()

    temp = pd.DataFrame.from_dict({'Year':list(newPhenotype.Year.drop_duplicates())}
                                          #[, '2018', '2017', '2016', '2015', '2014']}
                              ).merge(temp, how = 'left')

    temp = temp.interpolate(method='linear', limit_direction = 'both', axis = 0)
    temp = temp.loc[temp.Groups.notna(), :]
    temp = tempRef.merge(temp)
    interpolateSoil = pd.concat([interpolateSoil, temp])
    
# interpolateSoil

# %%
# interpolateSoil.to_clipboard()


# %%
interpolateSoil= interpolateSoil.drop(columns = ['Groups'])

# %%
print(interpolateSoil.shape)
print(newSoil.shape) 
# newSoil = interpolateSoil.copy()

# %%
interpolateSoil.info()

# %%
# reduce by dropping empty/cols with many missing values.
interpolateSoil= interpolateSoil.drop(columns = [
'SoilTaxonomicId',
'Grower',
'DateReceived',
'DateReported',
'EDepth',
'TextureMineralFraction',
'SampleType',
'ppmZn',
'ppmFe',
'ppmMn',
'ppmCu',
'ppmB',
'SoilSampleDate',
'LabId',
'LabSampleId',
'Location',
'SoilTextType',
'SoilTexture',
'PlowDepth',
'SikorapH'])

# %%
print('There are sites with multiple observations per year')
temp = interpolateSoil
temp = temp.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat', 'SoilpH']].groupby(['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']).count().reset_index()
temp.loc[temp.SoilpH > 1, ].head(5)


# %%
print('But this seems to be due to duplicate rows and is easily corrected')
temp = interpolateSoil.drop_duplicates()
temp = temp.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat', 'SoilpH']].groupby(['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']).count().reset_index()
temp.loc[temp.SoilpH > 1, ].head(5)


# %%
interpolateSoil = interpolateSoil.drop_duplicates()

# %%

# %%
from sklearn.impute import KNNImputer

# %%
interpolateSoilIds = interpolateSoil.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']]
interpolateSoilData = interpolateSoil.drop(columns= ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat'])
interpolateSoilDataNames = list(interpolateSoilData)
# knn impute
imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
imputer.fit(interpolateSoilData)
interpolateSoilData = imputer.transform(interpolateSoilData)
# convert back to df and add back col names
interpolateSoilData = pd.DataFrame(interpolateSoilData)
interpolateSoilData.columns = interpolateSoilDataNames
#reset indexes to prevent issues in binding cols
interpolateSoilIds = interpolateSoilIds.reset_index().drop(columns = ['index'])
interpolateSoilData = interpolateSoilData.reset_index().drop(columns = ['index'])
# replace
interpolateSoil = pd.concat([interpolateSoilIds, interpolateSoilData], axis=1)

# %%
interpolateSoil

# %%
# summarize down so there is only one entry for each experimentcode / year combination

temp = interpolateSoil.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat', 'SoilpH']].groupby(['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']).count().reset_index()
temp.loc[temp.SoilpH > 1, ]

# %%
newSoil = interpolateSoil.copy()

# %%

# %%
# multipleSoilEntries = pd.DataFrame(phenotype.loc[:, ['ExperimentCode', 'Year'] ]).drop_duplicates().merge(soil.loc[:, ['ExperimentCode', 'Year', 'SoilpH']]).groupby(['ExperimentCode', 'Year']).count().reset_index()

# multipleSoilEntries.loc[multipleSoilEntries.SoilpH >1, ]

# %%
expObsTrackerDf = track_newPheno_grainyield_obs(
    df = newPhenotype,
    markerName = '5.3',
    expObsTrackerDf = expObsTrackerDf
)

# %% [markdown]
# ## Weather

# %% [markdown]
# ### Reduce down weather

# %%
weatherDataCols = ['Temp','DewPoint','RelativeHumidity','SolarRadiation',
                   'Rainfall','WindSpeed','WindDirection','WindGust',
                   'SoilTemp','SoilMoisture','UVL','PAR',
                   'Photoperiod'
                #  'SoilEC','ppmCO2',
                  ]

for col in weatherDataCols:
    print(col)
    try:
        newWeather.loc[:, col].astype(float)
#         print('Can be converted to float.')
    except:
        print('------- Conversion failed.')

# %%
print("drop empty rows")
col = 'StationId'
newWeather = newWeather.loc[newWeather[col].notna(), ]

# %%
col = 'Rainfall'
rainfallValues = newWeather.loc[:, col].drop_duplicates()
print([entry for entry in rainfallValues if not convertable_to_float(entry)])

print('The following have rainfall encoded as "T" rather. Replacing with np.nan and will infer them.')
print(newWeather.loc[newWeather[col] == 'T', ['ExperimentCode', 'Year']].drop_duplicates())

newWeather.loc[newWeather[col] == 'T', col] = np.nan

# %%
# some values for rainfall are ludicrous. Check scaling.
# temp= newManagement.loc[(newManagement.Application == 'irrigation'
#                   ) | (newManagement.Product == 'Water'), :]
                              
# temp.sort_values('QuantityPerAcre', ascending = False)

temp = newWeather
temp.Rainfall = temp.Rainfall.astype('float')
temp.sort_values('Rainfall',ascending = False )
plt.scatter(temp.Rainfall, temp.Year)

# %%
temp = temp.loc[temp.Year == '2018', ]

# %%

# %%
from matplotlib.pyplot import figure
figure(figsize=(8, 6), dpi=80)

plt.scatter(temp.loc[temp.ExperimentCode.notna(), 'Rainfall'], temp.loc[temp.ExperimentCode.notna(), 'ExperimentCode'])

# %%

experimentCode = 'IAH2'
plt.scatter(temp.loc[temp.ExperimentCode == experimentCode, 'Month'],
            temp.loc[temp.ExperimentCode == experimentCode, 'Rainfall'],
            label= experimentCode)
plt.title('Normal Range')
plt.legend(loc='lower right')


# %%
experimentCode = 'IAH3'
plt.scatter(temp.loc[temp.ExperimentCode == experimentCode, 'Month'],
            temp.loc[temp.ExperimentCode == experimentCode, 'Rainfall'],
            label= experimentCode)
plt.title('Abnormal Range')
plt.legend(loc='lower right')


# %%
print('100x error?')
experimentCode = 'IAH3'
plt.scatter(temp.loc[temp.ExperimentCode == experimentCode, 'Month'],
            temp.loc[temp.ExperimentCode == experimentCode, 'Rainfall']/100,
            label= experimentCode)
plt.title('Abnormal Range')
plt.legend(loc='lower right')

# %%
print('Adjusting IAH3\'s 2018 Rainfall by x = x/100')

newWeather.loc[(newWeather.ExperimentCode == 'IAH3') & (newWeather.Year == '2018'), 
               'Rainfall'] = newWeather.loc[(newWeather.ExperimentCode == 'IAH3') & (newWeather.Year == '2018'), 
               'Rainfall']/100

print('Should be ~10 order of magnitude:', np.max(newWeather.loc[(newWeather.ExperimentCode == 'IAH3') & (newWeather.Year == '2018'), 
               'Rainfall']))



# %%
col = 'Photoperiod'
photoValues = newWeather.loc[:, col].drop_duplicates()

temp = pd.DataFrame([[entry]+str(entry).split(sep = ':') for entry in photoValues if not convertable_to_float(entry)]
            ).rename(columns = {0:'Raw',
                               1:'H',
                               2:'M'})
temp['H'] = temp['H'].astype(float)
temp['M'] = temp['M'].astype(float)
temp['H.P'] = np.nan
temp['H.P'] = (temp['H'] + temp['M']/60)

temp

for i in temp.index:
#     print(i+1, '/', len(temp.index))
    newWeather.loc[newWeather[col] == temp.loc[i, 'Raw'], col] = temp.loc[i, 'H.P']

# %%
col = 'WindGust'
windValues = newWeather.loc[:, col].drop_duplicates()

windValues

# %%
print('WindGust values unable to be converted into floats:')
for entry in list(windValues):
    try:
        float(entry)
    except :
        print(entry)

newWeather.loc[newWeather.WindGust == '#REF!', ['ExperimentCode', 'Year']].groupby(['ExperimentCode']).count()#.drop_duplicates()

print('Droping non-convertable WindGust values')

newWeather.loc[newWeather.WindGust == '#REF!', 'WindGust'] = np.nan

# %% code_folding=[]
for col in weatherDataCols:
    print(col)
    try:
        newWeather.loc[:, col].astype(float)
#         print('Can be converted to float.')
    except:
        print('------- Conversion failed.')
        
for col in weatherDataCols:
    newWeather.loc[:, col] = newWeather.loc[:, col].astype(float)
print('data cols converted!')

# %%
# There are missing values where we know there was rainfall because it was coded as 'T' earlier. 
plt.hist(newWeather.loc[newWeather.Rainfall > 0, 'Rainfall'], bins = 100)
plt.title('Mean non zero rainfall is '+str(round(newWeather.loc[newWeather.Rainfall > 0, 'Rainfall'].mean(), 3)) + 'mm')

# %%
newWeather.loc[newWeather.Rainfall.isna(), 'Rainfall'] = newWeather.loc[newWeather.Rainfall > 0, 'Rainfall'].mean()

# %%
temp = newWeather.groupby(
    ['ExperimentCode',
#      'StationId', # there appear to be multiple stations recorded. See IAH4, 2018	5	21. StationId == 9082, 9085
     'Year',
     'Month',
     'Day']).agg(
                            TempMin=('Temp','min' ),
                            TempMean=('Temp', 'mean'),
                              TempMax=('Temp','max' ),
                    DewPointMean=('DewPoint', 'mean'),
    RelativeHumidityMean=('RelativeHumidity', 'mean'),
        SolarRadiationMean=('SolarRadiation', 'mean'),
                    RainfallTotal=('Rainfall','sum' ),
                    WindSpeedMax=('WindSpeed','max' ),
          WindDirectionMean=('WindDirection', 'mean'),
                      WindGustMax=('WindGust','max' ),
                    SoilTempMean=('SoilTemp', 'mean'), # not in 2014
            SoilMoistureMean=('SoilMoisture', 'mean'), # not in 2014
                              UVLMean=('UVL', 'mean'), # not in 2014
                              PARMean=('PAR', 'mean'),# not in 2017
              PhotoperiodMean=('Photoperiod', 'mean')
)

temp= temp.reset_index()

newWeather = temp.copy()

# %%

# %%

# %% [markdown]
# ### Prep for weather Interpolation


# %%
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns



# %%
import wget
import os

def retrieve_daymet_data(inputLat,
                         inputLon,
                         inputYear,
                        saveCsv= False):
    # Download and write
    url = 'https://daymet.ornl.gov/single-pixel/api/data?lat='+ str(inputLat
                                                    ) +'&lon=' + str(inputLon
                                                  ) +'&years=' + str(inputYear)
    output_directory= r'../data/interim'
    filename = wget.download(url, out=output_directory)
    # print(filename)
    # read and reformat
    data= pd.read_csv(filename, skiprows=list(range(0,6)))
    data= data.fillna(method='ffill')

    # On a leap year there is no Dec 31.   
    data['year']=data['year'].astype('str')
    data['yday']=data['yday'].astype('str')
    data['Date']= pd.to_datetime(data['year']+data['yday'], format='%Y%j')
    data.drop(columns = ['year', 'yday'], inplace= True)

    data['Latitude']= inputLat
    data['Longitude']= inputLon
    
    if saveCsv == False:
        if os.path.exists(filename):
#             print('Removing '+filename)
            os.remove(filename)

    return(data[[
        'Latitude',
        'Longitude',
        'Date',
        'dayl (s)',
        'prcp (mm/day)',
        'srad (W/m^2)',
        'swe (kg/m^2)',
        'tmax (deg c)',
        'tmin (deg c)',
        'vp (Pa)']])


# %%

# %%

# %%
uniqueLoc = newMetadata.loc[:, ['Year', 'ExpLon', 'ExpLat']
                           ].drop_duplicates()
# uniqueLoc['YearSite'] = uniqueLoc['Year'].astype(str)+'_'+uniqueLoc['ExpLon'].astype(str)+'-'+uniqueLoc['ExpLat'].astype(str)
uniqueLoc = uniqueLoc.reset_index().drop(columns = ['index']
                           ).rename(columns = {'ExpLon':'DaymetLon', 'ExpLat':'DaymetLat'})

# %%
# uniqueLoc['Year'] =uniqueLoc['Year'].astype(int)

# uniqueLocBefore = uniqueLoc.copy()
# uniqueLocAfter = uniqueLoc.copy()

# uniqueLocBefore['Year'] = uniqueLocBefore['Year']-1
# uniqueLocAfter['Year'] = uniqueLocAfter['Year']+1

# uniqueLoc = pd.concat([uniqueLocBefore, uniqueLoc, uniqueLocAfter]).drop_duplicates()
# uniqueLoc['Year'] =uniqueLoc['Year'].astype(str)

# %% code_folding=[]
import tqdm

if remakeDaymet == True:
    if os.path.exists('../data/interim/daymetPart.pkl'):
        daymet= pd.read_pickle('../data/interim/daymetPart.pkl')
    else:    
        daymet= pd.DataFrame.from_dict({'Latitude': [],
                                        'Longitude': [],
                                        'Year': []})

    for i in tqdm.tqdm(uniqueLoc.index):
        # Confirm record does not exist already    
        if daymet.loc[(daymet.Latitude == uniqueLoc.loc[i, 'DaymetLat']) &
                      (daymet.Longitude == uniqueLoc.loc[i, 'DaymetLon']) &
                      (daymet.Year == uniqueLoc.loc[i, 'Year']), :].shape[0] == 0:
#             print(i)
            try:
                temp = retrieve_daymet_data(
                    inputLat=uniqueLoc.loc[i, 'DaymetLat'],
                    inputLon=uniqueLoc.loc[i, 'DaymetLon'],
                    inputYear=uniqueLoc.loc[i, 'Year'])
            except:
                print("Broke with input",
                     '\nLat ', uniqueLoc.loc[i, 'DaymetLat'],
                     '\nLon ', uniqueLoc.loc[i, 'DaymetLon'],
                     '\nYear', uniqueLoc.loc[i, 'Year'])
                break

            temp.loc[:, 'Year'] = uniqueLoc.loc[i, 'Year']

            if daymet.shape[0] == 0:
                daymet = temp
            else:
                daymet= pd.concat([daymet, temp])

    daymet.to_pickle('../data/interim/daymetPart.pkl')


# %%
if remakeDaymet == True:    
    daymet.to_pickle('../data/interim/daymet.pkl')
daymet= pd.read_pickle('../data/interim/daymet.pkl')

# %% code_folding=[]
daymet = daymet.rename(columns = {'DaymetLon':'ExpLon', 'DaymetLat':'ExpLat'})


# %%
# merge daymet with weather by pairing up the experimental codes with the lon/lat

# %%
connectingCols = newMetadata.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']].drop_duplicates()
connectingCols

# %%

# %%
# connectingCols.loc[connectingCols.ExperimentCode == 'DEH1']

# retrieve_daymet_data(
#                 inputLat=38.637405,  #uniqueLoc.loc[i, 'DaymetLat'],
#                 inputLon=-75.204048, #uniqueLoc.loc[i, 'DaymetLon'],
#                 inputYear='2014')     #uniqueLoc.loc[i, 'Year'])


# %%

# %%
newDaymetNames = {
            'tmin (deg c)': 'TempMinEst', # Daymet <-- 'tmin (deg c)', Temp Air Min
            'tmax (deg c)': 'TempMaxEst', # Daymet <-- 'tmax (deg c)', Temp Air Max  
            'srad (W/m^2)': 'SolarRadEst', # Daymet  <-- 'srad (W/m^2)', Shortwave Radiation   
           'prcp (mm/day)': 'PrecipEst',      # Daymet <-- 'prcp (mm/day)', Precipitation
            'swe (kg/m^2)': 'SnowWaterEqEst', # Daymet <-- 'swe (kg/m^2)', Snow Water Equivalent    
                 'vp (Pa)': 'VaporPresEst',   # Daymet <-- 'vp (Pa)' Daily average water vapor partial pressure    
                'dayl (s)': 'DayLenEst', # Daymet <-- 'dayl (s)', Day length    
               'Longitude': 'ExpLon', 
                'Latitude': 'ExpLat'
}

for key in newDaymetNames.keys():
    daymet = daymet.rename(columns = {key: newDaymetNames[key]})

# %%
# add Date column for easy merging

# %%
newWeather['Date'] = newWeather['Year'].astype(str)+ '-' + newWeather['Month'].astype(str)+ '-' + newWeather['Day'].astype(str)
newWeather['Date'] = pd.to_datetime(newWeather['Date'])

# %%

# %%

# %%
tempWeather = connectingCols.merge(newWeather, how = 'outer' # merge left to ignore any missing sites
                                  )
# tempWeather = tempWeather.merge(daymet, how = 'outer') # merge outer to include daymet's data for days that have no measured data

# %%
# Change how we add in daymet data to prevent missing days in data from multiple experimentcodes pointing to the same gps coordinates.

renameDaymetParams = tempWeather.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']].drop_duplicates()
renameDaymetParams = renameDaymetParams.reset_index().drop(columns = ['index'])

# %%

accumulator = pd.DataFrame()

for i in range(len(renameDaymetParams)):
    daymetPart = pd.DataFrame(renameDaymetParams.loc[i:i, :]).merge(daymet, how = 'left')
    if accumulator.shape[0] == 0:
        accumulator = daymetPart
    else:
        accumulator = pd.concat([accumulator, daymetPart])

# %%
tempWeather = tempWeather.merge(accumulator, how = 'outer')

# %%

# %%

# %%
tempWeather['doy']= pd.to_datetime(tempWeather.Date).dt.day_of_year
# get this in the same units as PhotoperiodMean
tempWeather.loc[:, 'DayLenEst'] = (tempWeather.loc[:, 'DayLenEst'] / (60*60))

# %%
tempWeather['YearSite'] = tempWeather['Year'].astype(str)+'_'+tempWeather['ExpLon'].astype(str)+'-'+tempWeather['ExpLat'].astype(str)
print(len(tempWeather['YearSite'].drop_duplicates()), 'unique year/lonlat comibnations exist.', '\nExamples:', '\n'.join(list(tempWeather['YearSite'].drop_duplicates())[0:5]))



# %%
list(tempWeather)

# %%
# make a backup for comparison and selectively drop measurements that are likey to be measurement errors.
backupWeather = tempWeather.copy()

# %%
# Here's an example of exactly what we're trying to get rid of. 
temp = tempWeather.loc[(tempWeather.ExperimentCode == 'ARH1') & (tempWeather.Year == '2018')]

temp= temp.sort_values('Date')

print('This sort of clear instrumental failure is exactly what we want to avoid.')
print('Orange/Blue == observed max/min. Gray bands are daymet estimates.')

plt.fill_between(temp.Date, temp['TempMinEst'], temp['TempMaxEst'], color='lightgray', alpha=0.5)
plt.plot(temp.Date, temp['TempMinEst'], color= 'black', alpha = 0.25)
plt.plot(temp.Date, temp['TempMaxEst'], color= 'black', alpha = 0.25)

plt.plot(temp.Date, temp.TempMax, color= 'darkorange', alpha = 0.75, label='Max')
plt.plot(temp.Date, temp.TempMin, color= 'skyblue', alpha = 0.75, label='Min')

plt.legend(loc='lower right')

plt.ylim(-20, 60)
plt.savefig('DemoTempError_1.svg',dpi=350)

# %%
# We want to QC based on expected temp.
sns.pairplot(data=tempWeather,
             y_vars=['TempMin', 'TempMean', 'TempMax'],
             x_vars=['TempMaxEst', 'TempMinEst',])

# %%
weatherModDat = tempWeather
# weatherModDat.info()

# %%
# make all the changes relative to the estimated minimum. 
# This lets us ignore seasonality because the prediction should already have that.
for col in ['TempMin', 'TempMean', 'TempMax']:
    
    weatherModDat[col] = weatherModDat[col] - weatherModDat['TempMinEst']

# use the middle x quantile to diminish the effect of outliers:
ql = 0.2
qh = 0.8
weatherModDatQuant = weatherModDat.drop(columns=['ExpLon', 'TempMinEst']
                                        ).groupby(['YearSite']
                                        ).quantile([ql, qh]
                                        ).reset_index(
                                        ).pivot(index='YearSite', columns=['level_1'])

# Reduce down to middle 60%
# Before: (16901, 8)
# After:   (5221, 8)
for exp in tqdm.tqdm(list(weatherModDat.YearSite.drop_duplicates())):
    # exp = 'ARH1'
    for col in ['TempMin', 'TempMean', 'TempMax']:
        # col = 'TempMaxEst'
        mask = (weatherModDat.YearSite != exp
          ) | ((weatherModDat.YearSite == exp
               ) & (weatherModDat[col] >= weatherModDatQuant.loc[exp, (col, ql)]
               ) & (weatherModDat[col] <= weatherModDatQuant.loc[exp, (col, qh)])
              )

        weatherModDat= weatherModDat.loc[mask, :]
        
        
# restore back to raw values from differences relative to estimated min
for col in ['TempMin', 'TempMean', 'TempMax']:
    weatherModDat[col] = weatherModDat[col] + weatherModDat['TempMinEst']

# %%
# Replace so the rest of the code runs as is. We'll restore it later.
tempWeather = weatherModDat

# %%

# %%
dropCols= ['ExperimentCode', 'Year', #'StationId', 
           'Month', 'Day', 'Date', 'doy']
mask = np.triu(np.ones_like(tempWeather.drop(columns = dropCols).corr(), dtype=np.bool))

plt.figure(figsize=(18, 9))

heatmap = sns.heatmap(
    tempWeather.drop(columns = dropCols).corr(), 
    vmin=-1, vmax=1, annot=True, 
    fmt='.1g',
    mask = mask,
    cmap='vlag_r') # <--- "_r" reverses it 
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

# %%
# sns.pairplot(data=tempWeather,
#              y_vars=[
#                  'TempMin',
#                  'TempMean',
#                  'TempMax',
#              ],
#              x_vars=[
#                  'TempMaxEst',
#                  'TempMinEst',
#              ])

# %%
# Probably also want to concatenate YearLocation in case weather stations were altered from year to year.


# %%
# tempWeather['YearSite'] = tempWeather['Year'].astype(str)+'_'+tempWeather['ExpLon'].astype(str)+'-'+tempWeather['ExpLat'].astype(str)


# print(len(tempWeather['YearSite'].drop_duplicates()), 'unique year/lonlat comibnations exist.', '\nExamples:', '\n'.join(list(tempWeather['YearSite'].drop_duplicates())[0:5]))

# %% [markdown]
# ### Find best model to use for weather interpolation

# %% code_folding=[16]
import statsmodels.api as sm
import statsmodels.formula.api as smf

def compare_model_fits(
    weatherModDat=tempWeather,
    modelStatements=[
        'TempMin ~ 1',
        'TempMin ~ TempMinEst',
        'TempMin ~ TempMaxEst',
        'TempMin ~ TempMaxEst + TempMinEst',
        'TempMin ~ TempMaxEst * TempMinEst'
    ]
):
    # Note: this has one pass the model statement in as a key to get the fitted model out.
    modelDict = {}
    # Fit all models ====
    for i in range(len(modelStatements)):
        fm = smf.ols(modelStatements[i], data=weatherModDat).fit()
        # modelDict.update({'model_'+str(i): fm})
        modelDict.update({modelStatements[i]: fm})

    # Calculate AICcs and make summary DF ====
    modelAICcs = []
    for key in modelDict.keys():
        fmAICc = sm.tools.eval_measures.aicc(
            llf=modelDict[key].llf,
            nobs=modelDict[key].nobs,
            df_modelwc=modelDict[key].df_model)

        modelAICcs = modelAICcs + [fmAICc]

    modelComparison = pd.DataFrame(zip(modelStatements, modelAICcs, (modelAICcs - min(modelAICcs)))
                                   ).rename(columns={0: 'Model',
                                                     1: 'AICc',
                                                     2: 'DeltaAICc'})
    return([modelComparison, modelDict])


# %% code_folding=[]
# for a metric
weatherStationMetrics = [
    'TempMin',
    'TempMean',
    'TempMax',
    'DewPointMean',
    'RelativeHumidityMean',
    'SolarRadiationMean',
    'RainfallTotal',
    'WindSpeedMax',
    'WindDirectionMean',
    'WindGustMax',
    'SoilTempMean',
    'SoilMoistureMean',
    'UVLMean',
    'PARMean',
    'PhotoperiodMean']

daymetMetrics = [
        'DayLenEst',
        'PrecipEst',
        'SolarRadEst',
        'SnowWaterEqEst',
        'TempMaxEst',
        'TempMinEst',
        'VaporPresEst',
        # 'ExpLon', # implicit (albeit catagorical) in YearSite 
        # 'ExpLat'
    ]



import itertools # for itertools.combinations, used in making formula to test in data interpolation.

# Alternate version: try up to r predictors chosen from the top n predictors 
def make_formula_combinations(
    tWeatherCor = tempWeather.corr(),
    wSMetric = 'DewPointMean',
    nTopPredictors = 3 ,
    rMax = 3, # max number of non-intercept non-site/year predictors (n choose r)
    consideredPredictors = [
        'DayLenEst',
        'PrecipEst',
        'SolarRadEst',
        'SnowWaterEqEst',
        'TempMaxEst',
        'TempMinEst',
        'VaporPresEst',
        'ExpLon', 
        'ExpLat'
    ],
        # how many of the most strongly correlated or predictors should we allow?
        # these will be tested by addition in the order that they're most strongly correlated:
        # y~1, y~1+a, y~1+a+b ...
    locationPredictor = 'YearSite' # leave empty if not desired
):
    # use cor to get the set of predictors under consideration:
    temp = pd.DataFrame(tWeatherCor.loc[wSMetric, [col for col in list(tWeatherCor) if col in consideredPredictors]])
    temp.columns = ['Cor']
    temp['AbsCor'] = np.abs(temp['Cor'])
    temp = temp.sort_values('AbsCor', ascending=False)
    temp = temp.reset_index()

    sortedPredictors = list(temp['index'])

    predictors = sortedPredictors[0:nTopPredictors]
    ## Make combinations for formulas ====
    allCombinations = []
    for r in range(rMax):
        r = r+1
        combinations = [combo for combo in itertools.combinations(predictors, r = r)]
        combinations = [list(combo) for combo in combinations]
        allCombinations = allCombinations + combinations

    allCombinations = [['1']] + allCombinations # add intercept only model
    rightHands      = [' + '.join(combo) for combo in allCombinations] # convert to right hand of model 
    ## Make formulas ====
    formulas = []
    leftHand = wSMetric + ' ~ '
    for rightHand in rightHands:
        formula=leftHand + rightHand
        formulas=formulas + [formula]

    ## Make formulas with site/year as fixed effect if desired ====
    if locationPredictor != '':
        formulas=formulas + [formula+' + ' +locationPredictor for formula in formulas]

    return(formulas)




# %%
# example: Test and select the best model to interpolate missing values measured by a weather station.

formulas = make_formula_combinations(
    tWeatherCor = tempWeather.corr(),
    wSMetric = 'DewPointMean', #y
    nTopPredictors = 3 ,       # how many of the most strongly correlated or predictors should we allow?
    rMax = 3,                  # max number of non-intercept non-site/year predictors (n choose r)
    consideredPredictors = [
        'DayLenEst',
        'PrecipEst',
        'SolarRadEst',
        'SnowWaterEqEst',
        'TempMaxEst',
        'TempMinEst',
        'VaporPresEst',
#         'ExpLon', # implicit (albeit catagorical) in YearSite 
#         'ExpLat'
    ],
        # how many of the most strongly correlated or predictors should we allow?
        # these will be tested by addition in the order that they're most strongly correlated:
        # y~1, y~1+a, y~1+a+b ...
    locationPredictor = 'YearSite'
)
formulas

# %%
temp = compare_model_fits(
    weatherModDat=tempWeather,
    modelStatements= formulas
)[0].sort_values('DeltaAICc')

list(temp.loc[temp.DeltaAICc == 0, ].Model)

# %%

# %%
# To make sure that the AICc is working correctly (not just selecting the most complicated model) we'll overwrite one of the y's with an x and make sure it chooses this model.
# tempWeather['TempMin'] = tempWeather['TempMinEst']
# Worked! to demo, we should make a copy and print out some confirmatory statement



# This is the prefered version of set seed
rng = np.random.default_rng(65467)

demoAicc = tempWeather.copy()
demoAicc['DewPointMeanAndNoise'] = demoAicc['DewPointMean']+rng.normal(loc = 0, scale = 0.1, size = len(demoAicc))
demoAicc['JustNoise'] = rng.normal(loc = 0, scale = 10, size = len(demoAicc))

demoOutput =compare_model_fits(
    weatherModDat=demoAicc,
    modelStatements= ['DewPointMean ~ DewPointMeanAndNoise'] + ['DewPointMean ~ JustNoise']+ formulas
)
print("Here's a demo to check that AICc is behaving as expected.")
print("We've added two predictors. One that's the response (and a little noise) and one that's only noise.")
print("These should be the first and last model when sorted by delta aicc.")
demoOutput[0].sort_values('DeltaAICc')


# %%

# %%

# %%

# %%

# %%

# %% code_folding=[]
def model_search_wrapper(
    corDf = tempWeather.corr(),
    metric = 'TempMin',
    consideredPredictors = daymetMetrics,
    nTopPredictors = 3,
    rMax = 3,
    locationPredictor = 'YearSite'
    
):
    formulas = make_formula_combinations(
        tWeatherCor = corDf,
        wSMetric = metric,                #y
        nTopPredictors = nTopPredictors , # how many of the most strongly correlated or predictors should we allow?
        rMax = rMax,                      # max number of non-intercept non-site/year predictors (n choose r)
        consideredPredictors = consideredPredictors,
        locationPredictor = locationPredictor
    )

    temp = compare_model_fits(
        weatherModDat=tempWeather,
        modelStatements= formulas
    )

    aiccTable = temp[0]
    bestFormula = str(list(temp[0].loc[temp[0].DeltaAICc == 0, 'Model'])[0])
    bestModel = temp[1][bestFormula]

    returnDict = {'y': metric,
          'aiccTable': aiccTable,
        'bestFormula': bestFormula,
          'bestModel': bestModel}

    return(returnDict)

# Returns dict with:
#
#           'y': metric,
#   'aiccTable': aiccTable,
# 'bestFormula': bestFormula,
#   'bestModel': bestModel 

# %%

# %%



# %%

# %%

# %%

# %%






# %% code_folding=[2, 32]
from datetime import datetime # just for timing runs while finding the best model for weather interpolation based on aicc
# put into practice for all weatherStationMetrics
import pickle # for pickling dict with tested models for weather interpolation
              # took 11:12.759771 to generate 

useNTopPredictors = 4
useRMax = 4

if findBestWeatherModels == True:
    outputDict = {}
    tic = datetime.now()
    for metric in weatherStationMetrics:
        then = datetime.now()

        res =model_search_wrapper(
                     corDf = tempWeather.corr(),
                    metric = metric,
      consideredPredictors = daymetMetrics,
            nTopPredictors = useNTopPredictors,
                      rMax = useRMax,
         locationPredictor = '' # <------------------------ removed to allow for filling in entries with only daymet data
        )

        outputDict.update({metric: res})

        # not necessary, just wanted to see the number of formulae tested.
        formulas = make_formula_combinations(
            tWeatherCor = tempWeather.corr(),
            wSMetric = metric,                
            nTopPredictors = useNTopPredictors ,
            rMax = useRMax,                      
            consideredPredictors = daymetMetrics,
            locationPredictor = 'YearSite'
        )
        now = datetime.now()
        print(now - then, 'spent finding best of', len(formulas) ,'models for', metric)
    toc = datetime.now()
    print(toc - tic)



    # outputDict
    file = open('../data/interim/weatherModelSearch.pkl', 'wb')
    pickle.dump(outputDict, file)
    file.close()

# %%
if findBestWeatherModels == False:
    file = open('../data/interim/weatherModelSearch.pkl', "rb")
    outputDict = pickle.load(file)
    file.close()

# %% [markdown]
# ### Evaluate and apply

# %%
keys = [key for key in outputDict.keys()]
bestFormulas = [outputDict[key]['bestFormula'] for key in keys]
pd.DataFrame(zip(keys, bestFormulas)).rename(columns = {0:'Response', 1:'BestFormula'})

# %%
## evaluate model effectiveness. ====
# Predict each and then plot y vs yHat 

# example
temp = tempWeather#.loc[(tempWeather.ExperimentCode == 'MOH1'#'ARH1'
                  # ) & (tempWeather.Year == '2018'), :]

# %%
temp= temp.sort_values('Date')
plt.scatter(temp.doy, temp['TempMinEst'], color= 'black', alpha = 0.25)
plt.scatter(temp.doy, temp.TempMin, color= 'skyblue', alpha = 0.75)

# %%
# make a prediciton:
key = 'TempMin'
temp['TempMinP'] = outputDict[key]['bestModel'].predict(temp)

plt.axline((0, 0), (1, 1), linewidth=1, color='r')
plt.scatter(temp.TempMin, temp['TempMinEst'], color= 'black', alpha = 0.25)
plt.scatter(temp.TempMin, temp['TempMinP'], color= 'skyblue', alpha = 0.25)

# %%
temp

# %%
# make a prediciton:
key = 'RainfallTotal'
temp['RainfallTotalP'] = outputDict[key]['bestModel'].predict(temp)

print('Note, the negative values here are corrected later.')
print('There should be the following line at the end of this section.')
print("newWeather.loc[newWeather.RainfallTotal <=0, 'RainfallTotal'] = 0")
plt.axline((0, 0), (1, 1), linewidth=1, color='r')
plt.scatter(temp['RainfallTotal'], temp['RainfallTotalP'], color= 'skyblue', alpha = 0.25)

# %%
temp

# %%
# tempWeather.info()

# %%
# tempWeather.info()

# %%

# backupWeather.info()

# %%
keys = [key for key in outputDict.keys()]

# add placeholder cols & fill
for col in [key+'P' for key in keys]:
    backupWeather[col] = np.nan
for key in keys:
    backupWeather[key+'P'] = outputDict[key]['bestModel'].predict(backupWeather)


# %%
# backupWeather.info()

# %%
# backupWeather

# %%


# %%

# %%
# replacements ====
def replace_if_tolerance_exceeded(
    df=backupWeather,
    y='TempMin',
    yHat='TempMinP',
    tolerance=5
):
    mask = (abs(df.loc[:, y] - df.loc[:, yHat]) > tolerance)
    df.loc[mask, y] = df.loc[mask, yHat]
    return(df)

def replace_if_missing(
    df=backupWeather,
    y='TempMin',
    yHat='TempMinP'
):
    mask = (df.loc[:, y].isna())
    df.loc[mask, y] = df.loc[mask, yHat]
    return(df)


# %%
# Here's an example of exactly what we're trying to get rid of. 
temp = backupWeather.loc[(backupWeather.ExperimentCode == 'ARH1') & (backupWeather.Year == '2018')]

temp= temp.sort_values('Date')

print('Here\'s an exampel of the estimated values alongside the instrumental failure.')

plt.fill_between(temp.Date, temp['TempMinEst'], temp['TempMaxEst'], color='lightgray', alpha=0.5)
plt.plot(temp.Date, temp['TempMinEst'], color= 'black', alpha = 0.25)
plt.plot(temp.Date, temp['TempMaxEst'], color= 'black', alpha = 0.25)

plt.plot(temp.Date, temp.TempMax, color= 'darkorange', alpha = 0.75, label='Max')
plt.plot(temp.Date, temp.TempMin, color= 'skyblue', alpha = 0.75, label='Min')

plt.plot(temp.Date, temp.TempMaxP, '--', color= 'darkorange', alpha = 0.9, label='Predicted Max')
plt.plot(temp.Date, temp.TempMinP, '--', color= 'skyblue', alpha = 0.9, label='Predicted Min')

plt.legend(loc='lower right')
plt.ylim(-20, 60)
plt.savefig('DemoTempError_2.svg',dpi=350)

# %%
# Replace/estimate missings:
# Replace existing values only for temperatures.
iqrMultiplier = 1.5 # for threshold
for key in ['TempMin', 'TempMean', 'TempMax']:
    diffs =backupWeather.loc[:, key] - backupWeather.loc[:, key+'P']
    q25 = np.percentile(diffs[diffs.notna()], [25])[0]
    q75 = np.percentile(diffs[diffs.notna()], [75])[0]
    iqr = (q75-q25)
    
    backupWeather = replace_if_tolerance_exceeded(
        df=backupWeather,
        y=key,
        yHat=key+'P',
        tolerance= iqr*iqrMultiplier)


# %%
# deal with leap days.

mask = ((backupWeather.Date.dt.year % 4 == 0) & (backupWeather.Date.dt.month == 2) & (backupWeather.Date.dt.day == 28)
       ) | ((backupWeather.Date.dt.year % 4 == 0) & (backupWeather.Date.dt.month == 3) & (backupWeather.Date.dt.day == 1))

weatherLeapDays = backupWeather.loc[mask].copy()
weatherLeapDays['Date'] = weatherLeapDays.Date-timedelta(1)

weatherLeapDaysIds = weatherLeapDays.loc[:, ['ExperimentCode', 'Date', 'ExpLon', 'ExpLat']]
weatherLeapDaysIds= weatherLeapDaysIds.groupby(['ExperimentCode', 'ExpLon', 'ExpLat']).agg('max').reset_index()
# mean impute leap day
weatherLeapDays = weatherLeapDays.groupby(['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']).agg('mean')
# assembled leap days ready to go. 
weatherLeapDays = weatherLeapDaysIds.merge(weatherLeapDays.reset_index())


keys = [key for key in outputDict.keys()]

# add placeholder cols & fill
for col in [key+'P' for key in keys]:
    weatherLeapDays[col] = np.nan
for key in keys:
    weatherLeapDays[key+'P'] = outputDict[key]['bestModel'].predict(weatherLeapDays)

backupWeather = pd.concat([backupWeather, weatherLeapDays])

# %%

# %%
# Replace missing values for all.  
for key in keys:    
    backupWeather = replace_if_missing(
        df=backupWeather,
        y=key,
        yHat=key+'P')


# %%

# %%
# Here's an example of exactly what we're trying to get rid of. 
temp = backupWeather.loc[(backupWeather.ExperimentCode == 'ARH1') & (backupWeather.Year == '2018')]

temp= temp.sort_values('Date')

print('Post Replacement.')

plt.fill_between(temp.Date, temp['TempMinEst'], temp['TempMaxEst'], color='lightgray', alpha=0.5)
plt.plot(temp.Date, temp['TempMinEst'], color= 'black', alpha = 0.25)
plt.plot(temp.Date, temp['TempMaxEst'], color= 'black', alpha = 0.25)

plt.plot(temp.Date, temp.TempMax, color= 'darkorange', alpha = 0.75, label='Max')
plt.plot(temp.Date, temp.TempMin, color= 'skyblue', alpha = 0.75, label='Min')

plt.plot(temp.Date, temp.TempMaxP, '--', color= 'darkorange', alpha = 0.9, label='Predicted Max')
plt.plot(temp.Date, temp.TempMinP, '--', color= 'skyblue', alpha = 0.9, label='Predicted Min')

plt.legend(loc='lower right')
plt.ylim(-20, 60)
plt.savefig('DemoTempError_3.svg',dpi=350)

# %%
# overwrite with adjusted measurements.
newWeather = backupWeather.copy()

# %%

# %%

# %%
# project forward and backwards for each

# %%

# %%
# find what additional values we would like from daymet -- each location's previous and subsequent year

availableCombinations = newWeather.loc[:, ['ExperimentCode', 'Year', 'ExpLon', 'ExpLat']].drop_duplicates()
availableCombinations['Year'] = availableCombinations['Year'].astype(int)

availableCombinationsBefore = availableCombinations.copy()
availableCombinationsAfter = availableCombinations.copy()

availableCombinationsBefore['Year'] = availableCombinationsBefore['Year']-1
availableCombinationsAfter['Year']  = availableCombinationsAfter['Year']+1

additionalCombinations = pd.concat([availableCombinationsBefore, 
#                                     availableCombinations, 
                                    availableCombinationsAfter]).drop_duplicates()

# availableCombinations['Available'] = True

# additionalCombinations = availableCombinations.merge(additionalCombinations, how = 'outer').drop_duplicates()
# additionalCombinations = additionalCombinations.loc[additionalCombinations.Available.isna(), ].reset_index().drop(columns = ['Available', 'index'])

# %%
availableCombinations = availableCombinations.reset_index().drop(columns = ['index'])

additionalCombinations = additionalCombinations.reset_index().drop(columns = ['index'])

# %%

# %%
# i = 0
# newWeather.loc[(newWeather.ExperimentCode == availableCombinations.loc[i, 'ExperimentCode']
#                 ) & (newWeather.Year == str(availableCombinations.loc[i, 'Year'])
#                 ) & (newWeather.ExpLon == availableCombinations.loc[i, 'ExpLon']
#                 ) & (newWeather.ExpLat == availableCombinations.loc[i, 'ExpLat']
#                 ), ]

if remakeDaymet == True:

    daymetAdditions = pd.DataFrame()


    for i in tqdm.tqdm(range(len(availableCombinations))):
        addDaymet = retrieve_daymet_data(inputLat = availableCombinations.loc[i, 'ExpLat'],
                                                 inputLon = availableCombinations.loc[i, 'ExpLon'],
                                                 inputYear = (availableCombinations.loc[i, 'Year'] +1),
                                                 saveCsv= False)
        addDaymet['ExperimentCode'] = availableCombinations.loc[i, 'ExperimentCode']
        addDaymet['Year'] = availableCombinations.loc[i, 'Year']

        if daymetAdditions.shape[0] == 0:
            daymetAdditions = addDaymet
        else:
            daymetAdditions = pd.concat([daymetAdditions, addDaymet])


        addDaymet = retrieve_daymet_data(inputLat = availableCombinations.loc[i, 'ExpLat'],
                                                 inputLon = availableCombinations.loc[i, 'ExpLon'],
                                                 inputYear = (availableCombinations.loc[i, 'Year'] -1),
                                                 saveCsv= False)
        addDaymet['ExperimentCode'] = availableCombinations.loc[i, 'ExperimentCode']
        addDaymet['Year'] = availableCombinations.loc[i, 'Year']

        if daymetAdditions.shape[0] == 0:
            daymetAdditions = addDaymet
        else:
            daymetAdditions = pd.concat([daymetAdditions, addDaymet])


    # rename
    for key in newDaymetNames.keys():
        daymetAdditions = daymetAdditions.rename(columns = {key: newDaymetNames[key]})

    #                                           ===================
    #                                           |    Critical!    |
    #                                           ===================
    # get this in the same units as PhotoperiodMean
    daymetAdditions.loc[:, 'DayLenEst'] = (daymetAdditions.loc[:, 'DayLenEst'] / (60*60))
    
    # Deal with leapday
    mask = ((daymetAdditions.Date.dt.year % 4 == 0) & (daymetAdditions.Date.dt.month == 2) & (daymetAdditions.Date.dt.day == 28)
       ) | ((daymetAdditions.Date.dt.year % 4 == 0) & (daymetAdditions.Date.dt.month == 3) & (daymetAdditions.Date.dt.day == 1))

    weatherLeapDays = daymetAdditions.loc[mask].copy()
    weatherLeapDays['Date'] = weatherLeapDays.Date-timedelta(1)

    weatherLeapDaysIds = weatherLeapDays.loc[:, ['Year', 'Date', 'ExpLon', 'ExpLat']]
    weatherLeapDaysIds= weatherLeapDaysIds.groupby(['Year', 'ExpLon', 'ExpLat']).agg('max').reset_index()

    # mean impute leap day
    weatherLeapDays = weatherLeapDays.groupby(['Year', 'ExpLon', 'ExpLat']).agg('mean')
    # assembled leap days ready to go. 
    weatherLeapDays = weatherLeapDaysIds.merge(weatherLeapDays.reset_index())

    daymetAdditions = pd.concat([daymetAdditions, weatherLeapDays])

    # predict

    # add in predictions
    keys = [key for key in outputDict.keys()]
    # add placeholder cols & fill
    for col in [key+'P' for key in keys]:
        daymetAdditions[col] = np.nan
    for key in keys:
        daymetAdditions[key+'P'] = outputDict[key]['bestModel'].predict(daymetAdditions)

    # for merge
    daymetAdditions.loc[:, 'Year'] = daymetAdditions.loc[:, 'Year'].astype(str)

#     newWeather = newWeather.merge(daymetAdditions, how = 'outer')


# %%
if remakeDaymet == True:
    daymetAdditions.to_pickle('../data/interim/daymetAdditions.pkl')
daymetAdditions= pd.read_pickle('../data/interim/daymetAdditions.pkl')

# %%
# add in the newly retrieved daymet data
newWeather = newWeather.merge(daymetAdditions, how = 'outer')


# %%
# Replace missings with predictions in same
for key in keys:    
    newWeather = replace_if_missing(
        df=newWeather,
        y=key,
        yHat=key+'P')

# %%
# Apply any hard limits here

# Rainfall cannot be negative. If it otherwise would be, it is now zero
plt.hist(newWeather['RainfallTotal'])

newWeather.loc[newWeather.RainfallTotal <=0, 'RainfallTotal'] = 0


# %%
expObsTrackerDf = track_newPheno_grainyield_obs(
    df = newPhenotype,
    markerName = '5.4',
    expObsTrackerDf = expObsTrackerDf
)

# %% [markdown]
# ## Management

# %%

# %%
# drop un-needed observations
newManagement = newPhenotype.loc[:, ['ExperimentCode', 'Year']].drop_duplicates().merge(newManagement, how = 'left')

backupManagement = newManagement.copy()

# %%
# Fix non-date dates

# move non-dates to a different col
searchStr = 'Preplant'
newManagement.loc[newManagement.Date == searchStr, 'epoch'] = searchStr
newManagement.loc[newManagement.Date == searchStr, 'Date'] = ''

searchStr = 'Sidedress'
newManagement.loc[newManagement.Date == searchStr, 'epoch'] = searchStr
newManagement.loc[newManagement.Date == searchStr, 'Date'] = ''

searchStr = 'unknown'
newManagement.loc[newManagement.Date == searchStr, 'epoch'] = searchStr
newManagement.loc[newManagement.Date == searchStr, 'Date'] = ''

searchStr = 'V7'
newManagement.loc[newManagement.Date == searchStr, 'epoch'] = searchStr
newManagement.loc[newManagement.Date == searchStr, 'Date'] = ''

searchStr = 'V5'
newManagement.loc[newManagement.Date == searchStr, 'epoch'] = searchStr
newManagement.loc[newManagement.Date == searchStr, 'Date'] = ''

# searchStr = 'fall 2013'
# newManagement.loc[newManagement.Date == searchStr, 'epoch'] = searchStr
# newManagement.loc[newManagement.Date == searchStr, 'Date'] = ''

# searchStr = 'spring 2014'
# newManagement.loc[newManagement.Date == searchStr, 'epoch'] = searchStr
# newManagement.loc[newManagement.Date == searchStr, 'Date'] = ''

# %%
newManagement.Date = pd.to_datetime(newManagement.Date)

# %%
# drop non-informative labels
searchStr = 'unknown'
newManagement.loc[newManagement.epoch == searchStr, 'epoch'] =  np.nan

searchStr = 'Sidedress'
newManagement.loc[newManagement.epoch == searchStr, 'epoch'] =  np.nan


# %% code_folding=[3]
def guess_dates_from_growth_epoch(
    df=newManagement,
    searchStr='Preplant'
):
    # TODO find a more exact reference for these values:    
        # From channel's Growth stages of corn pdf 
        # at https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwi_hYKJ6_LyAhWiVN8KHV7XAV0QFnoECAIQAQ&url=http%3A%2F%2Fchannel.com%2FGrowthStages&usg=AOvVaw1rtAPYql4h0v5sh3dQhnj7
    # VE = 4.5 days post planting
    # V3 = 2-4 weeks post VE
    # V5 = less than 4-6 weeks post VE
    # V7 = somewhere aournd 4-6 weeks post VE

    estimatedDiffForPreplant = -7  # days post plant
    estimatedDiffForV5 = 25  # days post plant
    estimatedDiffForV7 = 35  # days post plant

    strAbsent = df.loc[df.epoch != searchStr, ]
    strPresent = df.loc[df.epoch == searchStr, ]

    # get all the relevant dates, merge in then adjust all of them.
    relevantPlantingDates = strPresent.loc[:, ['ExperimentCode', 'Year']].drop_duplicates(
    ).merge(newPhenotype.loc[:, ['ExperimentCode', 'Year', 'DatePlanted']].drop_duplicates(), how='left')
    relevantPlantingDates = relevantPlantingDates.rename(
        columns={'DatePlanted': 'Date'})
    strPresent = strPresent.drop(columns='Date'
                                 ).merge(relevantPlantingDates, how='left')

    # pd.to_datetime(0)+timedelta()
    strPresent.Date = pd.to_datetime(strPresent.Date)

    if searchStr == 'Preplant':
        strPresent.Date = strPresent.Date + timedelta(estimatedDiffForPreplant)

    elif searchStr == 'V5':
        strPresent.Date = strPresent.Date + timedelta(estimatedDiffForV5)

    elif searchStr == 'V7':
        strPresent.Date = strPresent.Date + timedelta(estimatedDiffForV7)

    strPresent.Date = strPresent.Date.astype(str)

    return(pd.concat([strAbsent, strPresent]))


# %%
newManagement = guess_dates_from_growth_epoch(
    df=newManagement,
    searchStr='Preplant'
)

newManagement = guess_dates_from_growth_epoch(
    df=newManagement,
    searchStr='V5'
)

newManagement = guess_dates_from_growth_epoch(
    df=newManagement,
    searchStr='V7'
)

# can now drop 'epoch' all togehter
newManagement = newManagement.drop(columns = 'epoch')

# %%
# drop data free rows.
mask = (newManagement.Product.notna()) | (
    newManagement.QuantityPerAcre.notna()) | (
    newManagement.ApplicationUnit.notna()) | (
    newManagement.Irrigation.notna()) | (
    newManagement.WeatherStationIrrigation.notna())

newManagement = newManagement.loc[mask, ]

# convert to dates
newManagement.Date = pd.to_datetime(newManagement.Date)



# %%
print('The following entries in "QuantityPerAcre" cannot be ocnverted to a float.')
for entry in list(newManagement['QuantityPerAcre'].drop_duplicates()):
    try:
        float(entry)
    except:
        print(entry)

# %%

# %%
searchStr = 'unknown' # water in texh1 2014, 2014-04-29 & 2014-06-17
newManagement.loc[newManagement.QuantityPerAcre == searchStr, 'QuantityPerAcre'] = np.nan

newManagement.loc[:, 'QuantityPerAcre'] = newManagement.loc[:, 'QuantityPerAcre'].astype(float)

# %%

# %%
# are there multiple values across plots? else drop plot.
temp = newManagement.groupby(['ExperimentCode', 'Year', 'Product', 'Date']).agg(maxQuant=('QuantityPerAcre', max),
                                                                     minQuant=('QuantityPerAcre', min)).reset_index()

temp['rangeQuant'] = (temp['maxQuant'] - temp['minQuant'])
print('There are ', temp.loc[temp.rangeQuant > 0, :].shape[0], 'entries that require the plot column.')

# %%
print('dropping Plot')
newManagement = newManagement.drop(columns = ['Plot']).drop_duplicates()

# %%
# newManagement.loc[newManagement.Date.isna(), 'Application'].drop_duplicates()

# %%
# fix grouping errors:
newManagement.loc[
    (newManagement['Application'] == 'herbicide') &
    (newManagement['Product'] == 'npk28-0-0'), 'Application'] = 'fertilizer'

newManagement.loc[newManagement['Product'] == 'Water', 'Application'] = 'irrigation'

newManagement.loc[newManagement['Application'] == 'irrigation', 'Product'].drop_duplicates()

# %%
newManagement['Application'].drop_duplicates()

# %%
# Split into fertilizer and water ----
mask = (newManagement.Application == 'irrigation'
        ) | (newManagement.Product == 'Water')

colsInBoth = ['ExperimentCode',
    'Year',
    'Location',
    'Date',

    'Application',
    'Product',
    'QuantityPerAcre',
    'ApplicationUnit']

mW = newManagement.loc[mask,  colsInBoth+[#'Irrigation', 
    'WeatherStationIrrigation']]
mF = newManagement.loc[~mask, colsInBoth]

# %% [markdown]
# ### Combine water into weather

# %%
# convert from in/Acre to mm. mm is how it's measured in daymet/weather station.
mW.QuantityPerAcre = 25.4*mW.QuantityPerAcre
mW.ApplicationUnit = 'mm'

# %%

print('Impute missing values as median')
plt.hist(mW.loc[mW.QuantityPerAcre.notna(), 'QuantityPerAcre'], bins = 60)
# plt.hist(mW.loc[(mW.QuantityPerAcre.notna()) & (mW.QuantityPerAcre < 100), 'QuantityPerAcre'], bins = 60)

medianWaterPerAcre = mW.loc[mW.QuantityPerAcre.notna(), 'QuantityPerAcre'].median()


mW.loc[mW.QuantityPerAcre.isna(), 'QuantityPerAcre'] = medianWaterPerAcre

# %%

# %%
plt.hist(newWeather.RainfallTotal)

# %%
temp = mW.merge(newWeather, how = 'outer')
temp

# %%

# %%
print('If a site was supposedly irrigated on the same day that it rained, use the larger number as the true water the plants recieved.')

# prevent doublecounting of irrigation and rainfall
mask = (temp.RainfallTotal.notna()
             ) & (temp.QuantityPerAcre.notna()
             ) & (temp.RainfallTotal != 0
             ) & (temp.QuantityPerAcre != 0
             )
temp.loc[mask & (temp['RainfallTotal'] > temp['QuantityPerAcre']), 'QuantityPerAcre'] = 0
temp.loc[mask & (temp['RainfallTotal'] < temp['QuantityPerAcre']), 'RainfallTotal'] = 0



# %%
temp.loc[temp.RainfallTotal.isna(), 'RainfallTotal'] = 0
temp.loc[temp.QuantityPerAcre.isna(), 'QuantityPerAcre'] = 0

# %%
# Now we can sum the two columns and proceed from there
temp['WaterTotalInmm'] = temp['RainfallTotal'] + temp['QuantityPerAcre']
temp = temp.drop(columns = ['RainfallTotal', 'QuantityPerAcre',
                           'Application', 'Product', 'ApplicationUnit', 'WeatherStationIrrigation'])

# make any missing values 0.



# %%

# %%

# %%
temp= temp.loc[temp.Date.notna()]
newWeather = temp.copy()

# %%

# %% [markdown]
# ### Work with fertilizer

# %%
mF.loc[mF.Application == 'nan', ]


# %%

# %%

# %% code_folding=[0, 9]
def is_npk(entry):    
    #TODO allow for floats 'npk12.72-9.8-23.57'
#     if re.search('npk\d+-\d+-\d+', entry):
#     if re.search('npk\d+[.\d+]*-\d+[.\d+]*-\d+[.\d+]*', str(entry)):
    if re.search('npk\d[.\d]*-\d[.\d]*-\d[.\d]*', str(entry)):
        return(True)
    else:
        return(False)
    
def extract_npk(entry):
#     match= re.search(r'\d+-\d+-\d+', entry)
#     match= re.search(r'\d+[.\d+]*-\d+[.\d+]*-\d+[.\d+]*', str(entry))
    match= re.search(r'\d[.\d]*-\d[.\d]*-\d[.\d]*', str(entry))
    if match:
#         print(match.group(0))
        npk= match.group(0).split('-')
        if len(npk)==3:
            return (npk)
        else: 
            print('Warning! npk of length '+str(len(npk))+'.\nReturning nans.')
            return([np.nan, np.nan, np.nan])
    else:
        return([np.nan, np.nan, np.nan])


# %% code_folding=[0]
def extract_product_components(df):
    ## Convert all amenable weights (oz, ton) into lbs/Acre
    mask= (df['ApplicationUnit'] == 'oz/Acre')
    df.loc[mask, 'QuantityPerAcre']= pd.to_numeric(df.loc[mask, 'QuantityPerAcre'])/16
    df.loc[mask, 'ApplicationUnit']= 'lbs/Acre'

    mask= (df['ApplicationUnit'] == 'ton/Acre')
    df.loc[mask, 'QuantityPerAcre']= pd.to_numeric(df.loc[mask, 'QuantityPerAcre'])*2000
    df.loc[mask, 'ApplicationUnit']= 'lbs/Acre'


    df['Processed']= False
    df['N']= np.nan
    df['P']= np.nan
    df['K']= np.nan

    df= df.reset_index()


    ## Extract readily interpretable npks ==================================================================
    mask= [is_npk(entry) for entry in df['Product']]
    df.loc[mask, ['N', 'P', 'K']]= [extract_npk(entry) for entry in df.loc[mask, 'Product']]
    df.loc[mask, ['Processed']]= True
    # overwrite non lbs/Acre values 
    mask= df['ApplicationUnit'] != 'lbs/Acre'
    df.loc[mask, ['N', 'P', 'K', 'Processed']]= [np.nan, np.nan, np.nan, False]

    # Deprecated version
    # for i in df['index'].unique():
    #     if is_npk(list(df.loc[df['index'] == i, 'Product'])[0]):
    #         if list(df.loc[df['index'] == i, 'ApplicationUnit']) == 'lbs/Acre':
    #             # convert to percent and get lbs of each element
    #             npk= [(pd.to_numeric(entry)/100)* pd.to_numeric(df.loc[df['index'] == i, 'QuantityPerAcre'])
    #                 for entry in extract_npk(list(df.loc[df['index'] == i, 'Product'])[0])]

    #             df.loc[df['index'] == i, 'N']= npk[0]
    #             df.loc[df['index'] == i, 'P']= npk[1]
    #             df.loc[df['index'] == i, 'K']= npk[2]
    #             df.loc[df['index'] == i, 'Processed']= True

    ## Process elements ====================================================================================
    # N P K Mn Mo S Zn 

    ### N ##################################################################################################
    mask= (df['Product'] == 'N') & (df['ApplicationUnit'] == 'lbs/Acre')
    df.loc[mask, 'N']= df.loc[mask, 'QuantityPerAcre']
    df.loc[mask, 'Processed']= True

    ### P ##################################################################################################
    mask= (df['Product'] == 'P') & (df['ApplicationUnit'] == 'lbs/Acre')
    df.loc[mask, 'P']= df.loc[mask, 'QuantityPerAcre']
    df.loc[mask, 'Processed']= True

    ### K ##################################################################################################
    mask= (df['Product'] == 'K') & (df['ApplicationUnit'] == 'lbs/Acre')
    df.loc[mask, 'K']= df.loc[mask, 'QuantityPerAcre']
    df.loc[mask, 'Processed']= True

    ### Mn #################################################################################################
    mask= (df['Product'] == 'KickStand Manganese 4% Xtra') & (df['ApplicationUnit'] == 'lbs/Acre')
    df.loc[mask, 'Mn']= pd.to_numeric(df.loc[mask, 'QuantityPerAcre'])*0.04 #4% of mix
    df.loc[mask, 'Processed']= True

    ### Mo #################################################################################################
    mask= (df['Product'] == 'Mono-Ammonium Phosphate npk11-52-0+Mo0.001') & (df['ApplicationUnit'] == 'lbs/Acre')
    df.loc[mask, 'Mo']= pd.to_numeric(df.loc[mask, 'QuantityPerAcre'])*0.001 #0.001% of mix
    df.loc[mask, 'Processed']= True

    ### S ##################################################################################################
    mask= (df['Product'] == 'S') & (df['ApplicationUnit'] == 'lbs/Acre')
    df.loc[mask, 'S']= df.loc[mask, 'QuantityPerAcre']
    df.loc[mask, 'Processed']= True

    ### Zn #################################################################################################
    mask= (df['Product'] == 'Zn') & (df['ApplicationUnit'] == 'lbs/Acre')
    df.loc[mask, 'Zn']= df.loc[mask, 'QuantityPerAcre']
    df.loc[mask, 'Processed']= True

    mask= (df['Product'] == 'Zn planter fertilizer') & (df['ApplicationUnit'] == 'lbs/Acre')
    df.loc[mask, 'Zn']= df.loc[mask, 'QuantityPerAcre']
    df.loc[mask, 'Processed']= True

    mask= (df['Product'] == 'Zn sulfate') & (df['ApplicationUnit'] == 'lbs/Acre')
    df.loc[mask, 'Zn']= df.loc[mask, 'QuantityPerAcre']
    df.loc[mask, 'Processed']= True

    mask= (df['Product'] == 'npk11-37-0+4Zn') & (df['ApplicationUnit'] == 'lbs/Acre')
    df.loc[mask, 'Zn']= pd.to_numeric(df.loc[mask, 'QuantityPerAcre'])*0.04 #4% of mix
    df.loc[mask, 'Processed']= True

#     ## Write out unprocessed products for review =======================================================================
#     products= list(df.loc[df['Processed'] == False, 'Product'].unique())
# #     products.sort(reverse=False)
#     print('Unprocessed products present:')
#     for product in products:
#         print(product)

    return(df)

# %%
mF= extract_product_components(mF)
mF

# %% code_folding=[]
## NPK Likes ====

product = 'UAN28% npk28-0-0'
# https://www.dtnpf.com/agriculture/web/ag/crops/article/2016/03/21/nitrogen-math-simple-calculations
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'gal/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*2.9
mF.loc[mask, 'ApplicationUnit'] = 'lbs/Acre'
mF.loc[mask, 'Processed'] = 'True'


product = 'UAN30% npk30-0-0'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'gal/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*3.2
mF.loc[mask, 'ApplicationUnit'] = 'lbs/Acre'
mF.loc[mask, 'Processed'] = 'True'


product = 'Starter 16.55-6.59-2.03'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'lbs/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(16.55/100)
mF.loc[mask, 'P'] = mF.loc[mask, 'QuantityPerAcre']*( 6.59/100)
mF.loc[mask, 'K'] = mF.loc[mask, 'QuantityPerAcre']*( 2.03/100)
mF.loc[mask, 'Processed'] = 'True'


product = 'Starter 20-10-0'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'lbs/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(20/100)
mF.loc[mask, 'P'] = mF.loc[mask, 'QuantityPerAcre']*(10/100)
mF.loc[mask, 'Processed'] = 'True'


product = 'Starter 16-16-16'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'lbs/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(16/100)
mF.loc[mask, 'P'] = mF.loc[mask, 'QuantityPerAcre']*(16/100)
mF.loc[mask, 'K'] = mF.loc[mask, 'QuantityPerAcre']*(16/100)
mF.loc[mask, 'Processed'] = 'True'


product = 'nkpLike21-0-0-24' # 24% sulfer
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'lbs/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(21/100)
mF.loc[mask, 'Processed'] = 'True'


product = 'nkpLike28-0-0-0.5' # <- 0.5
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'lbs/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(28/100)
mF.loc[mask, 'Processed'] = 'True'


product = 'nkpLike28-0-0-5' # <- 5.0
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'lbs/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(28/100)
mF.loc[mask, 'Processed'] = 'True'


## N ====
product = 'N'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'units/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']
mF.loc[mask, 'Processed'] = 'True'


product = 'NH3'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'lbs/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*0.82
mF.loc[mask, 'Processed'] = 'True'


product = 'NH3 N-serve'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'lbs/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*0.82
mF.loc[mask, 'Processed'] = 'True'


product = 'UAN' 
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*0.46
mF.loc[mask, 'Processed'] = 'True'


product = 'Urea'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*0.46
mF.loc[mask, 'Processed'] = 'True'


product = 'Super Urea'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*0.46
mF.loc[mask, 'Processed'] = 'True'



# K ====
product = 'Potash'
# https://www.cropnutrition.com/resource-library/potassium-chloride
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ))
mF.loc[mask, 'K'] = mF.loc[mask, 'QuantityPerAcre']*0.60
mF.loc[mask, 'Processed'] = 'True'



## Other ====
product = 'Manure Poultry'
# https://extension.okstate.edu/fact-sheets/using-poultry-litter-as-fertilizer.html
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ) & (mF.ApplicationUnit == 'lbs/Acre'))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(63/2000)
mF.loc[mask, 'P'] = mF.loc[mask, 'QuantityPerAcre']*(61/2000)
mF.loc[mask, 'K'] = mF.loc[mask, 'QuantityPerAcre']*(50/2000)
mF.loc[mask, 'Processed'] = 'True'




# this one has no amount given.
# we'll guestimate it based off this entry for a similar n%
# index                             3584
# ExperimentCode                    ARH1
# Year                              2018
# Location                           NaN
# Date               2018-05-29 00:00:00
# Application                 fertilizer
# Product                      npk46-0-0
# QuantityPerAcre                  350.0
# ApplicationUnit               lbs/Acre
mF.loc[(mF.Product == 'Starter nkpLike50-45-10-10-2') 
      & (mF.QuantityPerAcre.isna()), 'QuantityPerAcre' ] = 350.0


product = 'Starter nkpLike50-45-10-10-2'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(50/100)
mF.loc[mask, 'P'] = mF.loc[mask, 'QuantityPerAcre']*(45/100)
mF.loc[mask, 'K'] = mF.loc[mask, 'QuantityPerAcre']*(10/100)
mF.loc[mask, 'Processed'] = 'True'





product = 'npkLike20-10-0-1'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(20/100)
mF.loc[mask, 'P'] = mF.loc[mask, 'QuantityPerAcre']*(10/100)
mF.loc[mask, 'K'] = mF.loc[mask, 'QuantityPerAcre']*(0/100)
mF.loc[mask, 'Processed'] = 'True'


product = 'npkLike27-0-0-6'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(27/100)
mF.loc[mask, 'P'] = mF.loc[mask, 'QuantityPerAcre']*(0/100)
mF.loc[mask, 'K'] = mF.loc[mask, 'QuantityPerAcre']*(0/100)
mF.loc[mask, 'Processed'] = 'True'


product = 'npkLike24-0-0-3'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(24/100)
mF.loc[mask, 'P'] = mF.loc[mask, 'QuantityPerAcre']*(0/100)
mF.loc[mask, 'K'] = mF.loc[mask, 'QuantityPerAcre']*(0/100)
mF.loc[mask, 'Processed'] = 'True'


product = 'npkLike6.7-13-33.5-4.4'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(6.7/100)
mF.loc[mask, 'P'] = mF.loc[mask, 'QuantityPerAcre']*(13/100)
mF.loc[mask, 'K'] = mF.loc[mask, 'QuantityPerAcre']*(33.5/100)
mF.loc[mask, 'Processed'] = 'True'


product = '30-26-0-6S'
mask = ((mF.Processed == False
    ) & (mF.Product == product
    ))
mF.loc[mask, 'N'] = mF.loc[mask, 'QuantityPerAcre']*(30/100)
mF.loc[mask, 'P'] = mF.loc[mask, 'QuantityPerAcre']*(26/100)
mF.loc[mask, 'K'] = mF.loc[mask, 'QuantityPerAcre']*(0/100)
mF.loc[mask, 'Processed'] = 'True'


# %%

# %%
# Starter nkpLike50-45-10-10-2
ignoredFertilizers = ['B10%', 'Orangeburg Mix', 'Lime', '24S']

print('These fertilizers were knowingly ignored:')
for i in ignoredFertilizers:
    print('  *  ', i)
print('These fertilizers were unknowingly ignored:')
for i in [entry for entry in list(mF.loc[mF.Processed == False, 'Product'].drop_duplicates()
                                 ) if entry not in ignoredFertilizers]:
    print('  *  ', i)


# %%
mF= mF.loc[:, ['ExperimentCode',
           'Year',
           'Location',
           'Date',
#            'Processed',
           'N',
           'P',
           'K']]

# mFN = mF.drop(columns = [     'P', 'K'])
# mFP = mF.drop(columns = ['N',      'K'])
# mFK = mF.drop(columns = ['N', 'P'     ])

# mFN.N = mFN.N.astype(float)
# mFN = mFN.loc[mFN.N > 0, ]

# mFP.P = mFP.P.astype(float)
# mFP = mFP.loc[mFP.P > 0, ]

# mFK.K = mFK.K.astype(float)
# mFK = mFK.loc[mFK.K > 0, ]

# mF = mFN.merge(mFP, how = 'outer').merge(mFK, how = 'outer')

# %%

# %%

# %%

# %%
mF.N = mF.N.astype(float)
mF.P = mF.P.astype(float)
mF.K = mF.K.astype(float)

mF.loc[mF.N.isna(), 'N'] = 0
mF.loc[mF.P.isna(), 'P'] = 0
mF.loc[mF.K.isna(), 'K'] = 0

# this line is dropping all years except for 2014...
# unclear why this behavior is cropping up. observed after date and year were set to string.
# year as the sole grouping variable returns multiple years.
# mFout= mF.groupby(['Year', 'ExperimentCode', 'Location', 'Date']).agg(N = ('N', 'sum'),
#                                                                P = ('P', 'sum'),
#                                                                K = ('K', 'sum')).reset_index()

# This less terse version does not have the same issue.
iterDf = mF.loc[:, ['Year', 'ExperimentCode', #'Location', 
                    'Date']].drop_duplicates().reset_index().drop(columns = ['index'])
mFSum = mF.copy()

for i in range(len(iterDf)):
    mask = (mFSum.Year == iterDf.loc[i, 'Year']) & (
    mFSum.ExperimentCode == iterDf.loc[i, 'ExperimentCode']) & (
    # mFSum.Location == iterDf.loc[i, 'Location']) & (
    mFSum.Date == iterDf.loc[i, 'Date'])

    mFSum.loc[mask, 'N'] = mFSum.loc[mask, 'N'].sum()
    mFSum.loc[mask, 'P'] = mFSum.loc[mask, 'P'].sum()
    mFSum.loc[mask, 'K'] = mFSum.loc[mask, 'K'].sum()


# %%
mF = mFSum.drop_duplicates().copy()

# %%

# %%
# infer dates for those which are not specified

# %%
mF.loc[mF.Date.isna()] #findme

# %%
mF['doy'] = pd.to_datetime(mF.Date).dt.day_of_year

# %%

# %%
plantingDates = newPhenotype.loc[:, ['ExperimentCode', 'Year', 'DatePlanted', 'DateHarvested']].drop_duplicates()

plantingDates.DatePlanted = pd.to_datetime(plantingDates.DatePlanted)
plantingDates['doyPlanted'] = pd.to_datetime(plantingDates.DatePlanted).dt.day_of_year


plantingDates.DateHarvested = pd.to_datetime(plantingDates.DateHarvested)
plantingDates['doyHarvested'] = pd.to_datetime(plantingDates.DateHarvested).dt.day_of_year


temp = plantingDates.merge(mF, how = 'outer')

print("making doy relative to doyPlanted")


temp['Relativedoy'] = temp['doy'] - temp['doyPlanted']
temp['RelativeHarvestdoy'] = temp['doyHarvested'] - temp['doyPlanted']

# %%
fig, axs = plt.subplots(3, 1)  
fig.set_size_inches(18.5, 10.5)

print('This plot shows a potential limitation with how we\'re considering fertilizer.') 
print('Here there are two sites that use fertilized during harvest.')
print('We make the assumption that the previous state of the field (eg did a cover crop get fertilized) is of limited information.')
      
mask = ((temp.doy.isna()) & (temp.N >0))
# Missing values
for i in list(temp.loc[mask, 'N']):
    axs[0].axhline(y=i, xmin=0, xmax=1, color = 'lightgray', alpha = 0.5)
# Harvest Dates
for i in list(temp.RelativeHarvestdoy):
    axs[0].axvline(x=i, ymin=0, ymax=1, color = 'gold', alpha = 0.05)
# Fertilizer Application and Amount
axs[0].scatter(temp.Relativedoy, temp.N, color = 'C0')
axs[0].set_title("Nitrogen", loc='center', pad=None)


mask = ((temp.doy.isna()) & (temp.P >0))
# Missing values
for i in list(temp.loc[mask, 'P']):
    axs[1].axhline(y=i, xmin=0, xmax=1, color = 'lightgray', alpha = 0.5)
# Harvest Dates
for i in list(temp.RelativeHarvestdoy):
    axs[1].axvline(x=i, ymin=0, ymax=1, color = 'gold', alpha = 0.05)
# Fertilizer Application and Amount
axs[1].scatter(temp.Relativedoy, temp.P, color = 'C1')
axs[1].set_title("Phosphorus", loc='center', pad=None)


mask = ((temp.doy.isna()) & (temp.K >0))
# Missing values
for i in list(temp.loc[mask, 'K']):
    axs[2].axhline(y=i, xmin=0, xmax=1, color = 'lightgray', alpha = 0.5)
# Harvest Dates
for i in list(temp.RelativeHarvestdoy):
    axs[2].axvline(x=i, ymin=0, ymax=1, color = 'gold', alpha = 0.05)
# Fertilizer Application and Amount
axs[2].scatter(temp.Relativedoy, temp.K, color = 'C2')
axs[2].set_title("Potassium", loc='center', pad=None)
axs[2].set_xlabel("Days Relative to Planting", loc='center')

fig.savefig('DemoImputeNPKDate_1.svg',dpi=350)

# %%
# we're ignoring out of season fertilizer and will impute with knn to get relative doy then convert that into a date. 

imputeRelativedoy = temp.loc[:, ['ExperimentCode', 'Year', 'N', 'P', 'K', 'doyPlanted', 'Relativedoy']]
imputeRelativedoy = imputeRelativedoy.loc[imputeRelativedoy.N.notna()]
justMissings = imputeRelativedoy.loc[imputeRelativedoy.Relativedoy.isna(), :]

# %%
imputeRelativedoyIds = imputeRelativedoy.loc[:,         ['ExperimentCode', 'Year']]
imputeRelativedoyData = imputeRelativedoy.drop(columns= ['ExperimentCode', 'Year'])
imputeRelativedoyDataNames = list(imputeRelativedoyData)
# knn impute
# imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
imputer = KNNImputer(n_neighbors=5, weights='distance', metric='nan_euclidean')

imputer.fit(imputeRelativedoyData)
imputeRelativedoyData = imputer.transform(imputeRelativedoyData)
# convert back to df and add back col names
imputeRelativedoyData = pd.DataFrame(imputeRelativedoyData)
imputeRelativedoyData.columns = imputeRelativedoyDataNames
#reset indexes to prevent issues in binding cols
imputeRelativedoyIds = imputeRelativedoyIds.reset_index().drop(columns = ['index'])
imputeRelativedoyData = imputeRelativedoyData.reset_index().drop(columns = ['index'])
# replace
imputeRelativedoy = pd.concat([imputeRelativedoyIds, imputeRelativedoyData], axis=1)

# %%

# %%
justMissings = justMissings.drop(columns = 'Relativedoy').merge(imputeRelativedoy, how = 'left')

# %%
justMissings['Imputedoy'] = justMissings['doyPlanted'] - justMissings['Relativedoy']
justMissings

# plt.scatter(justMissings.Relativedoy)

# %%
fig, axs = plt.subplots(3, 1)  
fig.set_size_inches(18.5, 10.5)
      
mask = ((temp.doy.isna()) & (temp.N >0))
# Missing values
for i in list(temp.loc[mask, 'N']):
    axs[0].axhline(y=i, xmin=0, xmax=1, color = 'lightgray', alpha = 0.5)
# Harvest Dates
for i in list(temp.RelativeHarvestdoy):
    axs[0].axvline(x=i, ymin=0, ymax=1, color = 'gold', alpha = 0.05)
# Fertilizer Application and Amount
axs[0].scatter(temp.Relativedoy, temp.N, color = 'C0')
axs[0].scatter(justMissings.Relativedoy, justMissings.N, color = 'Red')
axs[0].set_title("Nitrogen", loc='center', pad=None)


mask = ((temp.doy.isna()) & (temp.P >0))
# Missing values
for i in list(temp.loc[mask, 'P']):
    axs[1].axhline(y=i, xmin=0, xmax=1, color = 'lightgray', alpha = 0.5)
# Harvest Dates
for i in list(temp.RelativeHarvestdoy):
    axs[1].axvline(x=i, ymin=0, ymax=1, color = 'gold', alpha = 0.05)
# Fertilizer Application and Amount
axs[1].scatter(temp.Relativedoy, temp.P, color = 'C1')
axs[1].scatter(justMissings.Relativedoy, justMissings.P, color = 'Red')
axs[1].set_title("Phosphorus", loc='center', pad=None)


mask = ((temp.doy.isna()) & (temp.K >0))
# Missing values
for i in list(temp.loc[mask, 'K']):
    axs[2].axhline(y=i, xmin=0, xmax=1, color = 'lightgray', alpha = 0.5)
# Harvest Dates
for i in list(temp.RelativeHarvestdoy):
    axs[2].axvline(x=i, ymin=0, ymax=1, color = 'gold', alpha = 0.05)
# Fertilizer Application and Amount
axs[2].scatter(temp.Relativedoy, temp.K, color = 'C2')
axs[2].scatter(justMissings.Relativedoy, justMissings.K, color = 'Red')
axs[2].set_title("Potassium", loc='center', pad=None)
axs[2].set_xlabel("Days Relative to Planting", loc='center')

fig.savefig('DemoImputeNPKDate_2.svg',dpi=350)

# %%
print('The maximum relative harvest day is', temp.RelativeHarvestdoy.max(),',', round(212/30, 2), 'months')


# %%
# add imputed values back into the df

justMissings

# %%
# get rid of harvest info post plottin
temp = temp.drop(columns = ['DateHarvested', 'doyHarvested', 'RelativeHarvestdoy']).drop_duplicates()

# %%

# %%
# use missing value df to add in relative days to temp.
# then we'll convert those into real days.

for i in range(len(justMissings)):
    mask = (temp['ExperimentCode'] == justMissings.loc[i, 'ExperimentCode']) & (
    temp['Year'] == justMissings.loc[i, 'Year']) & (
    temp['N'] == justMissings.loc[i, 'N']) & (
    temp['P'] == justMissings.loc[i, 'P']) & (
    temp['K'] == justMissings.loc[i, 'K']) & (
    temp['doyPlanted'] == justMissings.loc[i, 'doyPlanted'])

    if temp.loc[mask, ].shape[0] > 1:
        print("Warning selection for relative doy replacement contains", temp.loc[mask, ].shape, 'for', i)
    else:
        # print(i, temp.loc[mask,].shape)
        temp.loc[mask, 'Relativedoy'] = justMissings.loc[i, 'Relativedoy']

# %%

# %%
temp= temp.loc[temp.N.notna()]

temp['EstimatedDate'] = np.nan
temp = temp.reset_index().drop(columns =['index'])

for i in range(temp.shape[0]):
    # if there is a planting date
    if type(temp.loc[i, 'DatePlanted']) != pd._libs.tslibs.nattype.NaTType:
        temp.loc[i, 'EstimatedDate'] = temp.loc[i, 'DatePlanted']+timedelta(temp.loc[i, 'Relativedoy'])


# %%
print('The following lack an estimated date:')
temp.loc[temp.EstimatedDate.isna()]

# %%
# fill in estimates
temp.loc[temp.Date.isna(), 'Date'] = temp.loc[temp.Date.isna(), 'EstimatedDate']

# round to nearest day
temp.Date = pd.to_datetime(temp.Date)
temp.Date = temp.Date.dt.round('D')

# %%
# drop cols and replace mF

mF = temp.loc[:, ['ExperimentCode', 'Year', 'Location', 'Date', 'N', 'P', 'K']].copy()

# %%
print('How far back should we select weather and management data?')
print('Vertical lines and numbers above denote the deciles.')
print('Going back 75 days -- ~2.5 months will be needed to get _all_ pre planting ferilizer applications for that year' )




plt.figure(figsize=(12, 3))
plt.scatter(temp.Relativedoy, y = [np.random.uniform() for i in range(len(temp.Relativedoy))])

j = 0.1
jCounter = 0
for i in list(np.percentile(temp.Relativedoy, [i for i in range(0, 110, 10)])):
    print(i)   
    plt.text(x = round(i), y = 1 + j, s = str(round(i)))
    plt.axvline(x=i, ymin=0, ymax=1, color = 'gold')
    

    if jCounter < 5:
        j += 0.1
    else:
        j -= 0.1
    jCounter += 1


# %%

# %%

# %%

# %%
nW = newWeather
nW['Weather'] = True
mF['Management'] = True

# required for merge to line up all 2014 entries
mF = mF.drop(columns = [entry for entry in list(mF) if entry in ['Location'] ] )



# %%
# Drop  any managment where the data for the application is outside of the relavant year (fertilizer that's going on in the fall)
# This is specifically for:
# ExperimentCode   Year   Location   Date
# IAH3             2014   NaN        2014-11-20
mF['Flag']= pd.to_datetime(mF.loc[:, 'Year']) > mF.loc[:, 'Date']

print('These will be dropped:')
print(mF.loc[mF.Flag, ])

mF = mF.loc[~mF.Flag].drop(columns = ['Flag'])

# %%
print(nW.shape, mF.shape)
res = mF.merge(nW, how = 'outer')
print(res.shape)
print('Additional row is a duplicate.')
print(res.drop_duplicates().shape)
res = res.drop_duplicates().drop(columns = ['Weather', 'Management'])
newWeather = res.copy()

# %%
newWeather.head()
newWeather.loc[newWeather.N.isna(), 'N'] = 0
newWeather.loc[newWeather.P.isna(), 'P'] = 0
newWeather.loc[newWeather.K.isna(), 'K'] = 0

# %% [markdown]
# ## Quick look at TASSEL hybrid genotype results

# %%
# These predate successful merger of genomic data.

# We've done two successful runs. 
# The first which uses the 
#                           210915_ prefix had the following filtering criteria:
#            name prefix | 210915_ | 210916_ |
# taxa min% site present | 0.1     | 0.1     |
# table min allele freq  | 0.4     | 0.225   |
# table max allele freq  | 0.6     | 0.775   |
#      allele freq range | **0.2** | **.55** | <- greatly relaxed, took ~120gb to pca transform
#              impute by | mean    | mean    |

# %%

# %%
# pca210915 = pd.read_table('../data/interim/210915_Eigenvalues_Imputed_g2fMerged_Hybrids_with_Probability_Filter_Filter.txt')
# pca210916 = pd.read_table('../data/interim/210916_Eigenvalues_Imputed_g2fMerged_Hybrids_Filter_Filter_with_Probability.txt')

# fig, axs = plt.subplots(3, 1)  
# fig.set_size_inches(18.5, 10.5)

# # axs[0].scatter(newPhenotype.DatePlanted.dt.day_of_year, newPhenotype.ExperimentCode)


# # axs[1].scatter(newPhenotype.Year, newPhenotype.ExperimentCode)
# # temp = newPhenotype.loc[newPhenotype.DatePlanted.isna(), ['ExperimentCode', 'Year']].drop_duplicates()
# # axs[1].scatter(temp.Year, temp.ExperimentCode, color = 'red')


# axs[0].plot(pca210915.PC, pca210915['cumulative proportion'], label = 'minor allele freq. 40.0%-60.0%')
# axs[0].plot(pca210916.PC, pca210916['cumulative proportion'], label = 'minor allele freq. 22.5%-77.5%')

# axs[1].plot(pca210915.loc[pca210915.PC < 30, 'PC'
#          ], pca210915.loc[pca210915.PC < 30, 'cumulative proportion'], label = 'minor allele freq. 40.0%-60.0%')
# axs[1].plot(pca210916.loc[pca210916.PC < 30, 'PC'
#          ], pca210916.loc[pca210916.PC < 30, 'cumulative proportion'], label = 'minor allele freq. 22.5%-77.5%')

# axs[2].plot(pca210915.loc[pca210915.PC < 30, 'PC'
#                        ], pca210915.loc[pca210915.PC < 30, 'cumulative proportion'] - pca210916.loc[pca210916.PC < 30, 'cumulative proportion'],
#            label = 'Diff. Variance explained (40%-60% - 22%-77%)',
#            color = 'black')

# axs[0].legend(loc='lower right')
# axs[1].legend(loc='lower right')
# axs[2].legend(loc='lower right')

# # axs[2].subtitle('Normal Range')

# %%
# pca210915 = pd.read_table('../data/interim/210915_PC_Imputed_g2fMerged_Hybrids_with_Probability_Filter_Filter.txt', skiprows=2)
# pca210916 = pd.read_table('../data/interim/210916_Imputed_g2fMerged_Hybrids_Filter_Filter_with_Probability.txt', skiprows=2)



# %%
# fig, axs = plt.subplots(1, 2)  
# fig.set_size_inches(16, 8)

# axs[0].scatter(pca210915.PC1, pca210915.PC2)
# axs[1].scatter(pca210916.PC1, pca210916.PC2)


# %%
# import plotly.express as px

# fig = px.scatter_3d(pca210915, x='PC1', y='PC2', z='PC3'
#                    )
# fig.show()

# %%
# fig = px.scatter_3d(pca210916, x='PC1', y='PC2', z='PC3'
#                    )
# fig.show()

# %%

# %%
# read in PCA info from TASSEL, look for hybrids that got filtered out due to min % site present requirement
# genoPCA = pca210916

# %%
expObsTrackerDf = track_newPheno_grainyield_obs(
    df = newPhenotype,
    markerName = '5.6',
    expObsTrackerDf = expObsTrackerDf
)

# %% [markdown]
# # Check that each phenotype has the data it needs

# %%
# Site level properties
# newPhenotype

# %%
nP = newPhenotype.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()
nS = newSoil.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()
nS['Indicator'] = True

temp = nP.merge(nS, how = 'outer')

print('If this is empty then there are no experiment/year pairs in phenotype that aren\'t in soil.')
temp.loc[temp.Indicator != True]


# %%
nP = newPhenotype.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()
nW = newWeather.loc[:, ['ExperimentCode', 'Year']].drop_duplicates()
nW['Indicator'] = True


for key in groupingDict.keys():
    nP.loc[:, 'ExperimentCode'] = [entry if entry not in groupingDict[key] else key for entry in list(nP.loc[:, 'ExperimentCode']) ]
    nW.loc[:, 'ExperimentCode'] = [entry if entry not in groupingDict[key] else key for entry in list(nW.loc[:, 'ExperimentCode']) ]
    
temp = nP.merge(nW, how = 'outer')

print('If this is empty then there are no experimentGroup/year pairs in phenotype that aren\'t in weather.')
print('Refer to "groupingDict" for groupings.')
temp.loc[temp.Indicator != True, 'ExperimentCode'
        ].drop_duplicates(
        ).reset_index(
        ).drop(columns = 'index'
        ).rename(columns = {'ExperimentCode':'Group'})

# %%
if True == False:
    print('If this is empty then there are no missing GBS entries in Phenotype.')
    newPhenotype.loc[newPhenotype.GBS0.isna() | newPhenotype.GBS1.isna(), ]

    genoPCA.head(3)

    gbsInNewP = newPhenotype.GBS0 +'/'+ newPhenotype.GBS1
    gbsInNewP = gbsInNewP.drop_duplicates()


    inBoth = [[gbsPair, True] if gbsPair in list(gbsInNewP) else [gbsPair, False] for gbsPair in list(genoPCA.Taxa)]

    inBoth = pd.DataFrame(inBoth).rename(columns = {0: 'GBS', 1: 'InPheno'})
    print('If this is empty there are no GBS in the PCA which are not in the pheotypic data')
    inBoth.loc[inBoth.InPheno != True]


    inBoth = [[gbsPair, True] if gbsPair in list(genoPCA.Taxa) else [gbsPair, False] for gbsPair in list(gbsInNewP)]

    inBoth = pd.DataFrame(inBoth).rename(columns = {0: 'GBS', 1: 'InGeno'})
    print('If this is empty there are no GBS in the phenotypic data which are not in the PCA')
    inBoth.loc[inBoth.InGeno != True]


    # pd.DataFrame([i.split(sep = '/') for i in inBoth.GBS]).to_clipboard()


    newPhenotype['GBS'] = newPhenotype.GBS0 +'/'+ newPhenotype.GBS1


    rmTheseGenotypes = [i for i in list(newPhenotype.GBS.drop_duplicates()) if i not in list(genoPCA.Taxa.drop_duplicates())]

    newPhenotype['NoGenotype'] = False

    for i in rmTheseGenotypes:
        newPhenotype.loc[newPhenotype.GBS == i, 'NoGenotype'] = True

    nPRemoved = newPhenotype.loc[newPhenotype.NoGenotype, ].shape[0]
    nPKept    = newPhenotype.loc[~newPhenotype.NoGenotype, ].shape[0]
    nPTot     = newPhenotype.shape[0]

    print('After removing hybrid genotypes filtered in TASSEL...')
    print('Removed Obs.:', nPRemoved, ' ', round(nPRemoved/nPTot, 2), '%')
    print('   Kept Obs.:', nPKept,    '', round(nPKept/nPTot, 2), '%')
    print('  Total Obs.:', nPTot)

    # Drop Non-Matches
    newPhenotype = newPhenotype.loc[newPhenotype.NoGenotype == False, 
                               ].reset_index().drop(columns = ['NoGenotype', 'index'])

    # Change name to match newPhenotype
    genoPCA = genoPCA.rename(columns = {'Taxa':'GBS'})
    


# %%
expObsTrackerDf = track_newPheno_grainyield_obs(
    df = newPhenotype,
    markerName = '6',
    expObsTrackerDf = expObsTrackerDf
)

# %% [markdown]
# ## Reduce columns in each dataframe

# %%

# %%
# what columns do we expect in each dataframe?
# Phenotype
reducedPhenotypeCols = [
# selection and matching
'ExperimentCode',
'Year',
'DatePlanted',
'DateHarvested',
'DaysToPollen', 
'DaysToSilk', # ---- Weather/Management
'Pedigree',
'GBS0',
'GBS1',
'GBS', # ----------- Geno
'F',   # ----------- Geno
'M',   # ----------- Geno
    
    
# possible ys    
'Height',
'EarHeight',
'StandCount',
'PercentStand',
'RootLodging',
'StalkLodging',
'PercentGrainMoisture',
'TestWeight',
'PlotWeight',
'GrainYield',
]


# # metadata
# reducedMetadataCols = [
# 'ExperimentCode',
# 'Year',
# #  'PreviousCrop',
# #  'PreplantTillage',
# #  'InseasonTillage',
# 'ExpLon',
# 'ExpLat', # ----- deprecated?
# #  'Location'
# ]



# soil
reducedSoilCols = [
'ExperimentCode',
'Year',
# 'ExpLon',
# 'ExpLat',
'SoilpH',
'WDRFpH',
'SSalts',
# 'TexturePSA',
'PercentOrganic',
'ppmNitrateN',
'NitrogenPerAcre',
'ppmK',
'ppmSulfateS',
'ppmCa',
'ppmMg',
'ppmNa',
'CationExchangeCapacity',
'PercentH',
'PercentK',
'PercentCa',
'PercentMg',
'PercentNa',
'ppmP',
'PercentSand',
'PercentSilt',
'PercentClay'
]

# weather
# all the measured parameters and any that daymet provides which we don't have represented in the dataset
reducedWeatherCols = [
'ExperimentCode',
'Year',
'Date',
    
'N',
'P',
'K',
'WaterTotalInmm',
# 'Management',
# 'Location',
# 'ExpLon',
# 'ExpLat',
# 'StationId',
# 'Month',
# 'Day',
'TempMin',
'TempMean',
'TempMax',
'DewPointMean',
'RelativeHumidityMean',
'SolarRadiationMean',
'WindSpeedMax',
'WindDirectionMean',
'WindGustMax',
'SoilTempMean',
'SoilMoistureMean',
'UVLMean',
'PARMean',
'PhotoperiodMean',

# 'DayLenEst',
# 'PrecipEst',
# 'SolarRadEst',
# 'SnowWaterEqEst',
# 'TempMaxEst',
# 'TempMinEst',
'VaporPresEst',
# 'doy',
# 'YearSite',
# 'TempMinP',
# 'TempMeanP',
# 'TempMaxP',
# 'DewPointMeanP',
# 'RelativeHumidityMeanP',
# 'SolarRadiationMeanP',
# 'RainfallTotalP',
# 'WindSpeedMaxP',
# 'WindDirectionMeanP',
# 'WindGustMaxP',
# 'SoilTempMeanP',
# 'SoilMoistureMeanP',
# 'UVLMeanP',
# 'PARMeanP',
# 'PhotoperiodMeanP',

# 'Weather'
]
# 

# %%

# %%
# apply reducitons
newPhenotype = newPhenotype.loc[:, ifColIn(newPhenotype, reducedPhenotypeCols)]
newWeather = newWeather.loc[:,  ifColIn(newWeather, reducedWeatherCols)]
newSoil = newSoil.loc[:, ifColIn(newSoil, reducedSoilCols)]
# genoPCA = genoPCA.loc[:, ]

# %%

# %%

#TODO move this to the appropriate location
newPhenotype = newPhenotype.loc[newPhenotype.GrainYield.notna(), ]
newPhenotype = newPhenotype.loc[newPhenotype.GrainYield != 'nan', ]

# reset all the indices
newMetadata = newMetadata.reset_index().drop(columns = ['index'])
newPhenotype = newPhenotype.reset_index().drop(columns = ['index'])
newWeather = newWeather.reset_index().drop(columns = ['index'])
newSoil = newSoil.reset_index().drop(columns = ['index'])
# genoPCA = genoPCA.reset_index().drop(columns = ['index'])

# %%
expObsTrackerDf = track_newPheno_grainyield_obs(
    df = newPhenotype,
    markerName = '6.1 post dropna',
    expObsTrackerDf = expObsTrackerDf
)

# %%
# write out:
newMetadata.to_pickle('../data/processed/metadata.pkl')
newPhenotype.to_pickle('../data/processed/phenotype.pkl')
newWeather.to_pickle('../data/processed/weather.pkl')
newSoil.to_pickle('../data/processed/soil.pkl')
# genoPCA.to_pickle('../data/processed/genoPCA.pkl')



# %% [markdown]
# # General Workspace

# %%
newPhenotype.loc[newPhenotype.ExperimentCode == 'NYH1', 'DatePlanted'].drop_duplicates()

# %%
# demo getting all the data needed for one observation. ----
i=1
print(newPhenotype.loc[i, ])
experimentCode = newPhenotype.loc[i, 'ExperimentCode']
year = newPhenotype.loc[i, 'Year']

# DatePlanted	DateHarvested	Anthesis	Silking
datePlanted = newPhenotype.loc[i, 'DatePlanted']
dateHarvested = newPhenotype.loc[i, 'DateHarvested']

gbs = newPhenotype.loc[i, 'GBS']

# %%

# %%
newSoil.loc[(newSoil.ExperimentCode == experimentCode
           ) & (newSoil.Year == year
           ),]

# %%
temp = newWeather.loc[(newWeather.ExperimentCode == experimentCode
           ) & (newWeather.Year == year
           ),]

temp = temp.sort_values('Date')

# two weeks before planting throught to harvest.
temp.loc[(temp.Date >= pd.to_datetime(datePlanted)-timedelta(75)) &
         (temp.Date <= pd.to_datetime(datePlanted)+timedelta(212)),]

# %%
genoPCA.loc[genoPCA.GBS == gbs, ]

# %%

# %%
# get 75 days prior to planing for each
# get 212 days after planting for each

# %%

# %%

# %%
