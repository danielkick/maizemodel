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
import numpy as np
import pandas as pd

# %%
# read in production datasets
# metadata  = pd.read_pickle('../data/processed/metadata.pkl') # for double checking expeirment code independence
phenotype = pd.read_pickle('../data/processed/phenotype.pkl')
# weather   = pd.read_pickle('../data/processed/weather.pkl')
# soil      = pd.read_pickle('../data/processed/soil.pkl')
# genoPCA   = pd.read_pickle('../data/processed/genoPCA.pkl')

# %%
phenotype

# %%
decoder2018 = pd.read_table('../data/raw/G2FGenomes/G2F_sample_decoder.txt') 
decoder2018old = pd.read_table('../data/raw/GenomesToFields_G2F_Data_2018/d._2018_genotypic_data/G2F_PHG_minreads1_Mo44_PHW65_MoG_assemblies_14112019_filtered_plusParents_sampleDecoder.txt',
              sep = ' ',
              names= ['GBS','F']
             )

decoder2017 = pd.read_excel('../data/raw/G2F_Planting_Season_2017_v1/d._2017_genotypic_data/g2f_2017_gbs_hybrid_codes.xlsx') 

# %%

# %%
# standardize column names
decoder2018 = decoder2018.rename(columns = {'inbred':'F', 'ID':'GBS'})

temp1 = decoder2017.loc[:, ['Female Pedigree', 'Female GBS']].drop_duplicates().rename(columns = {'Female Pedigree':'F', 'Female GBS':'GBS'})
temp2 = decoder2017.loc[:, ['Male Pedigree', 'Male GBS']].drop_duplicates().rename(columns = {'Male Pedigree':'F', 'Male GBS':'GBS'})
decoder2017 = temp1.merge(temp2, how = 'outer')

# %%
# what does the update get us?
decoder2018old['Old'] = True
decoder2018['New'] = True

temp = decoder2018.merge(decoder2018old, how = 'outer')
decoder2018 = decoder2018.drop(columns = 'New')

mask = (temp.New & temp.Old)
print(temp.loc[(~mask & temp.Old == True), ].shape[0], 'entries present in the old dataset but absent in the new one')


# %%

# %%

def fix_numeric_parent_names(entry):
    if type(entry) == str:
        return(entry)

    if type(entry) == int:
        return(str(entry))

    if type(entry) == float:
        return(str(int(entry)))
    
decoder2017['F'] = [fix_numeric_parent_names(entry) for entry in list(decoder2017['F'])]
decoder2018['F'] = [fix_numeric_parent_names(entry) for entry in list(decoder2018['F'])]

# %%
p = phenotype.loc[:, ['Pedigree', 'F', 'M', 'Year']]
p['Year'] = p.Year.astype(int)
p['UseDecoder'] = 0
p.loc[p.Year >= 2018, 'UseDecoder'] = 2018
p.loc[p.Year <  2018, 'UseDecoder'] = 2017
print(str(p.loc[p.UseDecoder == 0, ].shape[0]), 'rows contain years not defined.')
print('^ if that\'s 0, then we can just worry about the new one. ')

# %%
print(p.shape)
p = p.drop(columns = 'Year').drop_duplicates()
print(p.shape)

# %%
p1 = p.drop(columns = ['Pedigree', 'M'])
p1 = p1.drop_duplicates()

p2 = p.drop(columns = ['Pedigree', 'F']).rename(columns = {'M':'F'})
p2 = p2.drop_duplicates()

pMatch = p1.merge(p2, how = 'outer')
pMatch = pMatch.reset_index().drop(columns = 'index')
print(pMatch.shape)

# %%
pMatch['F'] = [fix_numeric_parent_names(entry) for entry in list(pMatch['F'])]

# %%
pMatch['Match2018'] = ''
pMatch['Match2017'] = ''

# %%
i=0
for i in range(pMatch.shape[0]):
    findF = pMatch.loc[i, 'F']

    matches2018 = list(decoder2018.loc[decoder2018.F == findF, 'GBS'])
    matches2017 = list(decoder2017.loc[decoder2017.F == findF, 'GBS'])

    if len(matches2018) == len(matches2017) == 0:
        print('"', findF, '"', ' returned no matches!\n', sep ='')
        
    else:
        if len(matches2018) > 1:
            print('"', findF, '"', ' returned ', len(matches2018) ,' matches!', sep ='')
            print('No replacement made.')
            print(matches2018)
#             print()
        elif len(matches2018) == 1:
            pMatch.loc[i, 'Match2018'] = matches2018[0]

        if len(matches2017) > 1:
            print('"', findF, '"', ' returned ', len(matches2017) ,' matches!', sep ='')
            print('No replacement made.')
            print(matches2017)
#             print()
        elif len(matches2017) == 1:
            pMatch.loc[i, 'Match2017'] = matches2017[0]


# %%
pMatch['FoundExpected'] = 'No'

# %%
mask = (pMatch.UseDecoder == 2018) & (pMatch.Match2018 != '') 
pMatch.loc[mask, 'FoundExpected'] = 'Yes'
mask = (pMatch.UseDecoder == 2017) & (pMatch.Match2017 != '') 
pMatch.loc[mask, 'FoundExpected'] = 'Yes'

# %%
pMatch

# %%
# okay, so now we have a set of matches we can start connecting it back to phenotype. 
# That'll let us get the most bang for our buck of sludging through names to find matches. 

pMatch['Either'] = False
mask = (pMatch.Match2018 != '') | (pMatch.Match2017 != '')
pMatch.loc[mask, 'Either'] = True

fMatch = pMatch.loc[pMatch.Either, 
          ].drop(columns = ['FoundExpected', 'Either']
                ).rename(columns = {
    'F':'F', 
    'Match2018':'F_Match2018', 
    'Match2017':'F_Match2017'    
})

mMatch = pMatch.loc[pMatch.Either, 
          ].drop(columns = ['FoundExpected', 'Either']
                ).rename(columns = {
    'F':'M', 
    'Match2018':'M_Match2018', 
    'Match2017':'M_Match2017'    
})



# %%
pObsMatched = p.merge(fMatch, how = 'left').merge(mMatch, how = 'left')
pObsMatched

# %%
# merge introduces NaNs (for str blank we used '') wo we can then use isna()to find those we still need to match. 

# mask = (pObsMatched.F_Match2017.isna()) | (pObsMatched.F_Match2018.isna())
mask = (pObsMatched.F_Match2017.isna()) | (pObsMatched.F_Match2018.isna(
    ) | pObsMatched.M_Match2017.isna()) | (pObsMatched.M_Match2018.isna())


nTotal = pObsMatched.shape[0]
nNoMatch = pObsMatched.loc[mask, ].shape[0]

print(round((nNoMatch / nTotal)*100, 2), '% unmatched observations')
print(nNoMatch, 'unmatched.')

# %%

# %%
temp = pObsMatched.loc[mask, ['F', 'Pedigree']
               ].merge(pObsMatched.loc[mask, ['M', 'Pedigree']].rename(columns = {'M':'F'}), how = 'outer'
                ).groupby(['F']
                ).count(
                ).reset_index(
                ).rename(columns = {'Pedigree':'ObsUnmatched'}
                ).sort_values('ObsUnmatched', ascending = False)
# temp.to_clipboard()
temp


# %%
# Read in manually matched lookup table
manualComparison = pd.read_excel('../data/raw/G2FGenomes/ManualComparison.xlsx')

# %%
# Get matches
manualMatches = manualComparison.loc[manualComparison.FObsUnmbatched.notna(), ['FObsUnmbatched', 'F', 'GBS', 'Year']]

manualMatches

# %%
manualMatches.loc[manualMatches.FObsUnmbatched == 'LH195', ]

# %%
# Instead of pivoting the data we're going to repeat on pairs of columns
colPairs = [
    {'col':'F_Match2017', # <-  contains GBS code
     'colF':'F'},         # <-  contains parent name
    
    {'col':'F_Match2018',
     'colF':'F'},
    
    {'col':'M_Match2017',
     'colF':'M'},
    
    {'col':'M_Match2018',
     'colF':'M'}]

verbose = False

# for each pair of parent/gbs columns 
for colPair in colPairs:
    # col = 'F_Match2017'
    # colF = 'F'
    col = colPair['col']
    colF = colPair['colF']

    #look only at those with missing values to speed up the process
    mask = pObsMatched[col].isna()
    findFObs = list(pObsMatched.loc[mask, colF].drop_duplicates())

    for i in range(len(findFObs)):
        FObsMatches = manualMatches.loc[manualMatches.FObsUnmbatched == findFObs[i], ]
        # FObsMatches
        if verbose:
            print(i, ':', FObsMatches.shape[0], 'entries found for', findFObs[i])

        # if there is a match
        if FObsMatches.shape[0] >= 1:
            maskCurrentMissing = (mask & (pObsMatched[colF] == findFObs[i]))
            maskCurrentMissingIdxs = pObsMatched.loc[maskCurrentMissing, ].index
            # maskCurrentMissingIdxs

            # within a matching set go row by row:
            # If there's a replacement which matches by year use it otherwise if there's any use that.    
            for j in maskCurrentMissingIdxs:
                # j = maskCurrentMissingIdxs[0]    
                matchesAndYearMatches = list(FObsMatches.loc[FObsMatches.Year == float(pObsMatched.loc[j, 'UseDecoder']), 'GBS'])
                matchesButYearDoesNot = list(FObsMatches.loc[FObsMatches.Year != float(pObsMatched.loc[j, 'UseDecoder']), 'GBS'])

                if len(matchesAndYearMatches) == 1:
                    pObsMatched.loc[j, col] = matchesAndYearMatches
                elif ((len(matchesAndYearMatches) == 0) & (len(matchesButYearDoesNot) == 1)):
                    pObsMatched.loc[j, col] = matchesButYearDoesNot
                elif len(matchesAndYearMatches) > 1:
                    print('For ', findFObs[i], 'multiple matches were found for the right year:', matchesAndYearMatches)
                elif len(matchesButYearDoesNot) > 1:
                    print('For ', findFObs[i], 'no matches were found for the right year \n but multiple matches were found for the wrong year:', matchesButYearDoesNot)
                else:
                    print('For ', findFObs[i], 'no matches were found.')


# %%
pObsMatched

# %%
# look for a difference

mask = (pObsMatched.F_Match2017.isna()) | (pObsMatched.F_Match2018.isna(
    ) | pObsMatched.M_Match2017.isna()) | (pObsMatched.M_Match2018.isna())

nTotal = pObsMatched.shape[0]
nNoMatch = pObsMatched.loc[mask, ].shape[0]

print(round((nNoMatch / nTotal)*100, 2), '% unmatched observations')
print(nNoMatch, 'unmatched.')

# %%
# make the best gbs pairs that we can with what we have(i.e. use within right timeframe if we can)

pObsMatched['F_MatchBest'] = ''
pObsMatched['M_MatchBest'] = ''

# Preferentially match within the same year set
## F
mask = ((pObsMatched.UseDecoder == 2018) & (pObsMatched.F_MatchBest == ''))
pObsMatched.loc[mask, 'F_MatchBest'] = pObsMatched.loc[mask, 'F_Match2018']

mask = ((pObsMatched.UseDecoder == 2017) & (pObsMatched.F_MatchBest == ''))
pObsMatched.loc[mask, 'F_MatchBest'] = pObsMatched.loc[mask, 'F_Match2017']
## M
mask = ((pObsMatched.UseDecoder == 2018) & (pObsMatched.M_MatchBest == ''))
pObsMatched.loc[mask, 'M_MatchBest'] = pObsMatched.loc[mask, 'M_Match2018']

mask = ((pObsMatched.UseDecoder == 2017) & (pObsMatched.M_MatchBest == ''))
pObsMatched.loc[mask, 'M_MatchBest'] = pObsMatched.loc[mask, 'M_Match2017']

# %%
# Fill in any missing that we can with the other year set.
## F
mask = ((pObsMatched.UseDecoder == 2018) & (pObsMatched.F_MatchBest == ''))
pObsMatched.loc[mask, 'F_MatchBest'] = pObsMatched.loc[mask, 'F_Match2017']

mask = ((pObsMatched.UseDecoder == 2017) & (pObsMatched.F_MatchBest == ''))
pObsMatched.loc[mask, 'F_MatchBest'] = pObsMatched.loc[mask, 'F_Match2018']
## M
mask = ((pObsMatched.UseDecoder == 2018) & (pObsMatched.M_MatchBest == ''))
pObsMatched.loc[mask, 'M_MatchBest'] = pObsMatched.loc[mask, 'M_Match2017']

mask = ((pObsMatched.UseDecoder == 2017) & (pObsMatched.M_MatchBest == ''))
pObsMatched.loc[mask, 'M_MatchBest'] = pObsMatched.loc[mask, 'M_Match2018']

# %%
# are there any left over that have a match outside their expected year?

mask = (pObsMatched.F_MatchBest == '') | (pObsMatched.M_MatchBest == ''
   ) | (pObsMatched.F_MatchBest.isna()) | (pObsMatched.M_MatchBest.isna())
obsStillUnmatched = pObsMatched.loc[mask, ]

# reduce down to just the matches
pObsMatched = pObsMatched.loc[~mask]

print(obsStillUnmatched.shape[0], 'unique pedigrees still contain missing gbs codes')
obsStillUnmatched

# %%
parentsStillMissing = set(
    list(obsStillUnmatched.loc[obsStillUnmatched.F_MatchBest.isna(), 'F']
  )+list(obsStillUnmatched.loc[obsStillUnmatched.M_MatchBest.isna(), 'M'])
)

print('This corresponds to ', len(parentsStillMissing), 'unique parents.')

# %%
# Merge back to convert # of pedigrees into # of observations:
pObsMatched = pObsMatched.loc[:, ['Pedigree', 'UseDecoder', 'F_MatchBest', 'M_MatchBest']].drop_duplicates()


# %%
# Remake -- modified from above.
p = phenotype#.loc[:, ['Pedigree', 'F', 'M', 'Year']]
p['Year'] = p.Year.astype(int)
p['UseDecoder'] = 0
p.loc[p.Year >= 2018, 'UseDecoder'] = 2018
p.loc[p.Year <  2018, 'UseDecoder'] = 2017


# indicator to make sure we're not dropping anything unexpected

# %%
pObsMatched['Indicator'] = 'Updated'

# %%
temp = pObsMatched.merge(p, how = 'outer')

print(temp.loc[(temp.Indicator == 'Updated') & (temp.GrainYield.isna()),  ].shape[0], 
      'of the pedigrees manually checked fail to match in phenotype.',
      'i.e. were in `pObsMatched` but lack GrainYield.')

# %%
# if none failed to match then we can get rid of the indicator.
# and overwrite `p`
p = temp.drop(columns = ['Indicator'])
p

# %%
mask = (p.F_MatchBest.notna() & p.M_MatchBest.notna())


print(p.loc[~mask, ].shape[0], 
      ' observations (', 
      round(100* (p.loc[~mask, ].shape[0] / p.shape[0]), 2), 
      '%) remain unmatched.', sep = '')
print(p.loc[mask, ].shape[0], ' observations are good to go.')

# %%
# instead of keeping only the desired entries, we'll use an indicator column. 
# This will allow us to make a lookup of index in full dataset to index of data conditioned on having genomic info present.

# p = p.loc[mask, ].copy()
p['HasGenomes'] = False
p.loc[mask, 'HasGenomes'] = True

# %%
# Prep out the file to use in TASSEL
uniqGBSPairs = p.loc[:, ['F_MatchBest', 'M_MatchBest']].drop_duplicates()

# %%

# %%
# Check that the codes all match.
# For making this file apply a _very_ aggressive table filter to get the taxa plus a few columns (only taxa will be kept).
# inTablePath = '../data/raw/G2FGenomes/pca_in_out/mH_with_pr_filteredHeavily_for_taxa_alone.txt'
# outPCAPath = '../data/raw/G2FGenomes/pca_in_out/PC_Imputed_mH_with_pr_Filter_Filter.txt'
inTablePath = '../data/raw/G2FGenomes/pca_in_out3/g2017m2019_rmID_merged_pr_filt_imp_filter_maf0.5_for_taxa_alone.txt'
outPCAPath = '../data/raw/G2FGenomes/pca_in_out3/PCs.txt'

print("""
This block requires a table with taxa going into tassel and a pca file coming out of tassel.
These are currently drawn from

1. Data into Tassel:
   """+inTablePath+"""\n
2. PCA out of Tassel:
   """+outPCAPath+"""
   
If you would like to proceed, type "yes".
""")


response = input()
if response == 'yes':


    # read in files
    inTable = pd.read_table(inTablePath, skiprows=0)
    outPCA = pd.read_table(outPCAPath, skiprows=2)


    # get a deduplicated list of the 
    phenoMatchBest = p.loc[p.HasGenomes, ["F_MatchBest", "M_MatchBest"]].copy()
    # phenoMatchBest = pd.DataFrame(phenoMatchBest.stack()).rename(columns = {0:'Taxa'})
    # distinctMatchBestList = list(set(distinctMatchBest.Taxa))
    # distinctMatchBestList


    # inelegant and slow but functional.
    phenoMatchBest['F_inTable'] = [True if entry in list(inTable.Taxa) else False 
                                   for entry in list(phenoMatchBest.F_MatchBest)]
    phenoMatchBest['M_inTable'] = [True if entry in list(inTable.Taxa) else False 
                                   for entry in list(phenoMatchBest.M_MatchBest)]
    phenoMatchBest['F_outPCA' ] = [True if entry in list(outPCA.Taxa) else False 
                                   for entry in list(phenoMatchBest.F_MatchBest)]
    phenoMatchBest['M_outPCA' ] = [True if entry in list(outPCA.Taxa) else False 
                                   for entry in list(phenoMatchBest.M_MatchBest)]
    
    # In table supplied to TASSEL
    mask = ((phenoMatchBest.F_inTable) & (phenoMatchBest.M_inTable))
    # In PCA recieved from TASSEL
    mask2 = ((phenoMatchBest.F_outPCA) & (phenoMatchBest.M_outPCA))
    print(phenoMatchBest.loc[mask2, ].shape[0])


    print("""
    Initial samples--------------- """+str(p.shape[0])+"""
    Expected to match input table- """+str(phenoMatchBest.shape[0])+"""
    Actually Match input table---- """+str(phenoMatchBest.loc[mask, ].shape[0])+"""
    Match output PCA-------------- """+str(phenoMatchBest.loc[mask2, ].shape[0])+"""
    """)

    # distinctMatchBest.loc[:, ["F_MatchBest"]].rename(columns = {"F_MatchBest": "Taxa"})
    # distinctMatchBest.loc[:, ["M_MatchBest"]].rename(columns = {"M_MatchBest": "Taxa"})


    # inTable.Taxa
    # outPCA.Taxa

# %%
# Output with pca_in_out
# yes
# 54272

#     Initial samples--------------- 98996
#     Expected to match input table- 86757
#     Actually Match input table---- 86757
#     Match output PCA-------------- 54272
    
# Output with pca_in_out3
# yes
# 86757

#     Initial samples--------------- 98996
#     Expected to match input table- 86757
#     Actually Match input table---- 86757
#     Match output PCA-------------- 86757

# %%
pd.DataFrame(inTable.Taxa).assign(Input = True).merge(pd.DataFrame(outPCA.Taxa).assign(Output = True), how = 'outer').to_clipboard()

# %%
8996-6137

6757-4135

4272-2224

# %%

# %%
# Write out the file to use in TASSEL
uniqGBSPairs.to_csv(r'../data/raw/G2FGenomes/desiredHybrids.txt', 
                    header=None, index=None, sep='\t', mode='a')

# %%

# %%

# %%
# write out phenotype with only those that match on GBS codes.
p.to_pickle('../data/processed/phenotype_with_GBS.pkl')


# %%
# indexComparison = p.reset_index().rename(columns = {'index':'OriginalIndex'}
#                   ).loc[mask, 
#                   ].reset_index().rename(columns = {'index':'NewIndex'}
#                   ).loc[:, ['OriginalIndex', 'NewIndex']]

# mask = indexComparison.OriginalIndex != indexComparison.NewIndex

# indexComparison.loc[mask, ]




# p['HasGenome'] = False
# p.loc[mask, 'HasGenome'] = True

# indexHasGenome = list(p.loc[p.HasGenome].reset_index().rename(columns = {'index':'HasGenome'}).HasGenome)
# indexHasNoGenome = list(p.loc[p.HasGenome == False].reset_index().rename(columns = {'index':'HasNoGenome'}).HasNoGenome)
