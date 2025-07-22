# -*- coding: utf-8 -*-
"""

This module contains the Standard-Sample-Bracketing

@author: Andreas Wittke
"""

import pandas as pd
import os
from pathlib import Path
from statistics import mean
import sys
import numpy as np
import pickle
pd.options.mode.chained_assignment = None 
sys.path.append('')

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from modules.config import outfile_results_Cu, outfile_results_Sn, outfile_results_Ag, outfile_results_Li, outfile_results_Sb
from modules.config import WorkspaceVariableInput
fullnameCu = os.path.join(outfile_results_Cu, 'Cu_export.csv')
resultnameCu = os.path.join(outfile_results_Cu, 'Cu_delta.csv') # moved to IsoPy.py
resultnameCuExcel = os.path.join(outfile_results_Cu, 'Cu_delta.xlsx')
fullnameSn = os.path.join(outfile_results_Sn, 'Sn_export.csv')
resultnameSn = os.path.join(outfile_results_Sn, 'Sn_delta.csv') # moved to IsoPy.py
resultnameSnExcel = os.path.join(outfile_results_Sn, 'Sn_delta.xlsx')
plotSn = os.path.join(outfile_results_Sn, 'Sn_plots.csv')
fullnameAg = os.path.join(outfile_results_Ag, 'Ag_export.csv')
resultnameAg = os.path.join(outfile_results_Ag, 'Ag_delta.csv') # moved to IsoPy.py
fullnameLi = os.path.join(outfile_results_Li, 'Li_export.csv')
resultnameLi = os.path.join(outfile_results_Li, 'Li_delta.csv') # moved to IsoPy.py
resultnameLiExcel = os.path.join(outfile_results_Li, 'Li_delta.xlsx')
resultnameSb = os.path.join(outfile_results_Sb, 'Sb_delta.csv') # moved to IsoPy.py
resultnameSbExcel = os.path.join(outfile_results_Sb, 'Sb_delta.xlsx')
fullnameSb = os.path.join(outfile_results_Sb, 'Sb_export.csv')

file_path2 = os.path.join("modules", "variable_data_input.pkl")
# Variable mit pickle laden
with open(file_path2, 'rb') as file2:
    standardinputpfad = pickle.load(file2)

infile = standardinputpfad



def ssbCu():
    global fullnameCu
    global resultnameCu
    global infile

    df = pd.read_csv(fullnameCu,  delimiter='\t')
    # create now column holding the booleans
    df.drop(['61Ni','64Ni', '66Zn'], axis=1, inplace=True)
    df['blank'] = df['Filename'].str.contains('_Blk|_BLK')
    df['standard'] = df['Filename'].str.contains('_s0|s1')
    df['sample'] = ~df['blank'] & ~df['standard']

    # drop _1, _2 and _3 in the filenames
    df['oldFilename'] = df['Filename']
    df['Filename'] = df['Filename'].map(lambda x: x.rstrip('_123'))

    # choose only samples
    dfSamples=df[df['sample'] == True]

# detect the standard before and after the sample
    i = 0
    for sampleIndex in dfSamples.index:
        # find standards before the current index
        # sub dataframe that holds only the df before the curretn sample
        dfBefore=df.iloc[0:sampleIndex]
        # invert in order to earch backward
        dfBeforeInverse=dfBefore[::-1]
        # get the rows that hold a standard
        dfStandardBefore = dfBeforeInverse[dfBeforeInverse['standard'] == True]
        # index of the standard closest to the sample 
        indexBefore = dfStandardBefore.index[0]
        # safe index in dataframe at position of the sample
        df.at[sampleIndex, 'standardBefore'] = indexBefore
        # find standards after the current index
        lastIndex = df.index[-1]
        dfAfter=df.iloc[sampleIndex:lastIndex]
        dfStandardAfter = dfAfter[dfAfter['standard'] == True]
        indexAfter = dfStandardAfter.index[0]
        df.at[sampleIndex, 'standardAfter'] = indexAfter
    
        # calculate the delta and the 2sd
        stdBeforeCu_ratio=df['65Cu/63Cu_corrected'].loc[df['standardBefore'].loc[sampleIndex]]
        stdAfterCu_ratio=df['65Cu/63Cu_corrected'].loc[df['standardAfter'].loc[sampleIndex]]
        stdBeforeCu_ratio_uncorr=df['65Cu/63Cu'].loc[df['standardBefore'].loc[sampleIndex]]
        stdAfterCu_ratio_uncorr=df['65Cu/63Cu'].loc[df['standardAfter'].loc[sampleIndex]]
        Filename = df['Filename']
        meanStdCu_ratio = (stdBeforeCu_ratio + stdAfterCu_ratio)/2
        meanStdCu_ratio_uncorr= (stdBeforeCu_ratio_uncorr + stdAfterCu_ratio_uncorr) / 2
        
        
        sampleCu_ratiodelta_uncorr = ((dfSamples['65Cu/63Cu'].loc[sampleIndex]/meanStdCu_ratio_uncorr)-1)*1000
        dfSamples.loc[sampleIndex, 'd65Cu/63Cu uncorr'] = sampleCu_ratiodelta_uncorr
        sampleCu_ratiodelta_corrected = ((dfSamples['65Cu/63Cu_corrected'].loc[sampleIndex]/meanStdCu_ratio)-1)*1000
        dfSamples.loc[sampleIndex, 'd65Cu/63Cu corrected'] = sampleCu_ratiodelta_corrected

    deltaresults = dfSamples.groupby('Filename')['d65Cu/63Cu corrected'].mean(['delta65Cu/63Cu corrected'])
    deltaresults_uncorr = dfSamples.groupby('Filename')['d65Cu/63Cu uncorr'].mean(['delta65Cu/63Cu uncorr'])
    deltaresults_round = np.round(deltaresults, 3)
    deltaresults_uncorr_round = np.round(deltaresults_uncorr, 3)

    delta2sd = dfSamples.groupby('Filename')['d65Cu/63Cu corrected'].std(ddof = 0) * 2
    delta2sd_uncorr = dfSamples.groupby('Filename')['d65Cu/63Cu uncorr'].std(ddof = 0) * 2
    delta2sd_round = np.round(delta2sd, 3)
    delta2sd_uncorr_round = np.round(delta2sd_uncorr, 3)
    resultuncorr = pd.concat([deltaresults_uncorr_round, delta2sd_uncorr_round, deltaresults_round, delta2sd_round], axis = 1) 
    resultuncorr.to_csv(resultnameCu, sep='\t', mode="w", header=False, index_label='Index_name')
    reread = pd.read_csv(resultnameCu, sep='\t', names = ['Lab Nummer', 'd65Cu/63Cu_uncorr', '2SD_uncorr', 'd65Cu/63Cu_corrected', '2SD_corrected'], index_col=False)
    
    ##get date
    infiles = infile + '/'
    filehead = [x for x in os.listdir(infiles) if x .endswith('exp') and 's01' in x]
    filehead = infiles + str(filehead)[2:-2]
    df3 = pd.read_csv(filehead, skiprows = 1, nrows = 1, sep=' |\t', engine='python', index_col = False)
    df3.columns = range(df3.shape[1])
    df3.drop(df3.columns[0], axis = 1, inplace = True)
    df3.drop(df3.columns[1:3], axis = 1, inplace = True)

    reread.columns = reread.columns.str.replace('2SD', '2SD')
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('_').str.get(-1)  #make Filename nice
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('.').str.get(0)
    
    reread.insert(1,'Messdatum', '')
    reread.insert(0, 'Objekt', '')
    reread.insert(0, 'Orig. Nummer', '')
    reread.insert(0, 'Ma Nummer', '')
    
    reread['Messdatum'] = df3.values[0][0]
    
    Nist = [x for x in os.listdir(infiles) if x .endswith('exp') and 's1' or 's01'in x]
    if len(Nist) > 0:
        reread['Referenz'] = 'NIST SRM 976'
        reread['Verwenden'] = ''
        reread['Bemerkung'] ='Bracketing, Sprühkammer, Ni-Cone'
    else:
        pass
    
    
    reread.to_csv(resultnameCu, sep = '\t', header=True, index=False)
    reread.drop(reread.columns[5:7], axis=1, inplace = True)
    reread.columns = ['\u03B465/63Cu' if x=='d65Cu/63Cu_corrected' else x for x in reread.columns]
    reread.columns = ['2SD' if x=='2SD_corrected' else x for x in reread.columns]
    reread.to_excel(resultnameCuExcel, index=False)
    
#############################################################    
def ssbSn():
    global fullnameSn
    global resultnameSn
    
    # read data into dataframe and set headers
    df = pd.read_csv(fullnameSn,  delimiter='\t')
    
    df.drop(df.columns[0], axis = 1, inplace = True)
    df.drop(df.columns[1:27], axis = 1, inplace = True)
    cols = list(df.columns)
    del cols[0]
    
    # create now column holding the booleans
    df['blank'] = df['Filename'].str.contains('_Blk|_BLK')
    df['standard'] = df['Filename'].str.contains('_Nis|_Nis1|_Ni0')
    df['sample'] = ~df['blank'] & ~df['standard']

    # drop _1, _2 and _3 in the filenames
    df['oldFilename'] = df['Filename']
    df['Filename'] = df['Filename'].map(lambda x: x.rstrip('_123'))

    # choose only samples
    dfSamples=df[df['sample'] == True]

    colsSamples = list(dfSamples.columns)
    del colsSamples[0]
    del colsSamples[-4:-1]
    del colsSamples[-1]

    delta = pd.DataFrame(columns = colsSamples)
    new_names = [(i,'d_' +i) for i in delta.iloc[:, 0:].columns.values]
    delta.rename(columns = dict(new_names), inplace = True)
    delta.columns = delta.columns.str.rstrip('_corrected')

    dfSamples = pd.concat([dfSamples, delta])
    
    # detect the standard before and after the sample
    i = 0
    for sampleIndex in dfSamples.index:

        dfBefore=df.iloc[0:sampleIndex]
        dfBeforeInverse=dfBefore[::-1]
        dfStandardBefore = dfBeforeInverse[dfBeforeInverse['standard'] == True]
        indexBefore = dfStandardBefore.index[0]
        df.at[sampleIndex, 'standardBefore'] = indexBefore
        lastIndex = df.index[-1]
        dfAfter=df.iloc[sampleIndex:lastIndex]
        dfStandardAfter = dfAfter[dfAfter['standard'] == True]
        indexAfter = dfStandardAfter.index[0]
        df.at[sampleIndex, 'standardAfter'] = indexAfter
    
    
    # calculate the StdBefore and StdAfter for each column
        stdBeforeCu_ratio=df[cols].loc[df['standardBefore'].loc[sampleIndex]]
        stdAfterCu_ratio=df[cols].loc[df['standardAfter'].loc[sampleIndex]]
    
        meanStdCu_ratio = (stdBeforeCu_ratio + stdAfterCu_ratio)/2
    
    # calculate delta and save them into new dataframe
        sampleSn_ratiodelta_corrected = (((dfSamples[colsSamples].loc[sampleIndex]/meanStdCu_ratio)-1)*1000)
        sampleSn_ratiodelta_corrected.index = list(delta)
        dfSamples.loc[sampleIndex] = sampleSn_ratiodelta_corrected
    
        dfSamples['Filename'] = df['Filename']
    ## this part is for inserting 120Sn, 121Sb and 120Sn/121Sb into result file
    
    df_2 = pd.read_csv(fullnameSn,  delimiter='\t', index_col = False)
    df_2['blank'] = df_2['Filename'].str.contains('_Blk|_BLK')
    df_2['standard'] = df_2['Filename'].str.contains('_Nis|Ni1')
    df_2['sample'] = ~df_2['blank'] & ~df_2['standard']
    df_2['oldFilename'] = df_2['Filename']
    df_2['Filename'] = df_2['Filename'].map(lambda x: x.rstrip('_123'))
    
    df_2Samples=df_2[df_2['sample'] == True]
    df_2Samples = df_2Samples.drop(columns=['oldFilename'])


    df_2Samples = df_2Samples.groupby('Filename').mean()
    
    
    delta_cols = [col for col in dfSamples if 'd_' in col]
    deltaresults = dfSamples.groupby('Filename')[delta_cols].mean()
    deltaresults_round = np.round(deltaresults, 3)
    
    delta2sd = dfSamples.groupby('Filename')[delta_cols].std(ddof = 0) * 2
    new_names_sd = [(i,'2SD+' + i) for i in delta2sd.iloc[:, 0:].columns.values]
    delta2sd.rename(columns = dict(new_names_sd), inplace = True)
    delta2sd_round = np.round(delta2sd, 3)
    results = pd.concat([deltaresults_round, delta2sd_round], axis = 1) 
    results = results[[item for items in zip(deltaresults_round.columns, delta2sd_round.columns) for item in items]]

    #insert needed stuff for export file
    results['120Sn'] = df_2Samples['120Sn'].round(1)
    results['121Sb'] = df_2Samples['121Sb'].round(1)
    results['120Sn/121Sb'] = (df_2Samples['120Sn'] / df_2Samples['121Sb']).round(2)
    
    #dSn
    df_ratio_delta = pd.DataFrame([{'116Sn/120Sn': -4, '117Sn/120Sn': -3, '118Sn/120Sn': -2, '119Sn/120Sn': -1, '122Sn/120Sn': 2, '124Sn/120Sn': 4, '124Sn/116Sn': 8, '122Sn/116Sn': 6, '117Sn/119Sn': -2}])
    df_4 = dfSamples.copy()
    df_4.drop(df_4.columns[1:14], axis = 1, inplace = True)
    col4_result = list(df_4.columns)
    col4_result = [t.replace('d_','') for t in col4_result]
    df_4.columns =col4_result
    df_4 = df_4.set_index('Filename')
    df_4 = df_4.astype(float)
    
    def cal(row):
        reg = np.polyfit(df_ratio_delta.values[0], row.values, 1)
        predict = np.poly1d(reg) # Slope and intercept
        trend = np.polyval(reg, df_ratio_delta)
        std = row.std() # Standard deviation
        r2 = np.round(r2_score(row.values, predict(df_ratio_delta.T)), 5) #R-squared
        df5 = pd.DataFrame([predict])
        df5.columns = ['dSn', 'Intercept']
        Slope = df5['dSn'].to_string(dtype=False)
        Slope = Slope[3:]
        Intercept = df5['Intercept'].to_string(dtype=False)
        Intercept = Intercept[3:]
        return Slope, Intercept, r2
    
    df_4[['dSn', 'intercept', 'r2']] = df_4.apply(cal, axis=1, result_type='expand')
    df_4[['dSn', 'intercept']] = df_4[['dSn', 'intercept']].astype(float)
    df_4.to_csv(plotSn, sep = '\t', header = True)
    
    df_6 = df_4[['dSn', 'intercept', 'r2']]
    df_6 = df_6.reset_index(drop = True)
    dfSamples = pd.concat([dfSamples, df_6], axis = 1)
    dSn_mean = np.round(df_4.groupby('Filename')['dSn'].mean(), 3)
    dSn_2sd = np.round(df_4.groupby('Filename')['dSn'].std(ddof = 0) * 2, 3)
    dSn_2sd = dSn_2sd.rename('2SD')

    results = pd.concat([results, dSn_mean], axis = 1)
    results = pd.concat([results, dSn_2sd], axis = 1)
   
    
    ## Insert correct header
    col_result = list(results.columns)
    col_result.insert(0, 'Lab Nummer')
    col_result = [l.replace('_','') for l in col_result]
    col_result = [i.split('+', 1)[0] for i in col_result]
    
    results.to_csv(resultnameSn, sep='\t', mode="w", header=True, index_label='Index_name')
    
    ## Insert Messdatum in results and other needed columns
    
    infiles = infile + '/'
    filehead = [x for x in os.listdir(infiles) if x .endswith('exp') and 'Nis01' in x]
    filehead = infiles + str(filehead)[2:-2]
        
    df3 = pd.read_csv(filehead, skiprows = 1, nrows = 1, sep=' |\t', engine='python', index_col = False)
    df3.columns = range(df3.shape[1])
    df3.drop(df3.columns[0], axis = 1, inplace = True)
    df3.drop(df3.columns[1:3], axis = 1, inplace = True)
    
        
    reread = pd.read_csv(resultnameSn, sep='\t', index_col=False)
    reread.columns = col_result
    reread.columns = reread.columns.str.replace('2SD', '2SD')
    
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('_').str.get(-1) 
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('.').str.get(0)
    reread.insert(1,'Messdatum', '')
    reread.insert(0, 'Objekt', '')
    reread.insert(0, 'Orig. Nummer', '')
    reread.insert(0, 'Ma Nummer', '')
    
    reread['Messdatum'] = df3.values[0][0]
    
    Nist = [x for x in os.listdir(infiles) if x .endswith('exp') and 'Nis1' or 'Nis01' or 'Nis08' or 'Nis06' in x]
    if len(Nist) > 0:
        reread['Referenz'] = 'NIST SRM 3161a'
        reread['Verwenden'] = ''
        reread['Bemerkung'] ='Bracketing, Sprühkammer, Ni-Cone'
    else:
        pass
    
    
    reread.to_csv(resultnameSn, sep = '\t', header=True, index=False)
    reread.to_excel(resultnameSnExcel, index=False)

    
#############################################################    
def ssbSn_short():
    global fullnameSn
    global resultnameSn
    
    # read data into dataframe and set headers
    df = pd.read_csv(fullnameSn,  delimiter='\t')
    
    df.drop(df.columns[0], axis = 1, inplace = True)
    df.drop(df.columns[1:21], axis = 1, inplace = True)
    cols = list(df.columns)
    del cols[0]
    
    # create now column holding the booleans
    df['blank'] = df['Filename'].str.contains('_Blk|_BLK')
    df['standard'] = df['Filename'].str.contains('_Nis|_Nis1|_Ni0')
    df['sample'] = ~df['blank'] & ~df['standard']

    # drop _1, _2 and _3 in the filenames
    df['oldFilename'] = df['Filename']
    df['Filename'] = df['Filename'].map(lambda x: x.rstrip('_123'))

    # choose only samples
    dfSamples=df[df['sample'] == True]

    colsSamples = list(dfSamples.columns)
    del colsSamples[0]
    del colsSamples[-4:-1]
    del colsSamples[-1]

    delta = pd.DataFrame(columns = colsSamples)
    new_names = [(i,'d_' +i) for i in delta.iloc[:, 0:].columns.values]
    delta.rename(columns = dict(new_names), inplace = True)
    delta.columns = delta.columns.str.rstrip('_corrected')

    dfSamples = pd.concat([dfSamples, delta])
    
    # detect the standard before and after the sample
    i = 0
    for sampleIndex in dfSamples.index:

        dfBefore=df.iloc[0:sampleIndex]
        dfBeforeInverse=dfBefore[::-1]
        dfStandardBefore = dfBeforeInverse[dfBeforeInverse['standard'] == True]
        indexBefore = dfStandardBefore.index[0]
        df.at[sampleIndex, 'standardBefore'] = indexBefore
        lastIndex = df.index[-1]
        dfAfter=df.iloc[sampleIndex:lastIndex]
        dfStandardAfter = dfAfter[dfAfter['standard'] == True]
        indexAfter = dfStandardAfter.index[0]
        df.at[sampleIndex, 'standardAfter'] = indexAfter
    
    
    # calculate the StdBefore and StdAfter for each column
        stdBefore_ratio=df[cols].loc[df['standardBefore'].loc[sampleIndex]]
        stdAfter_ratio=df[cols].loc[df['standardAfter'].loc[sampleIndex]]
    
        meanStd_ratio = (stdBefore_ratio + stdAfter_ratio)/2
    
    # calculate delta and save them into new dataframe
        sampleSn_ratiodelta_corrected = (((dfSamples[colsSamples].loc[sampleIndex]/meanStd_ratio)-1)*1000)
        sampleSn_ratiodelta_corrected.index = list(delta)
        dfSamples.loc[sampleIndex] = sampleSn_ratiodelta_corrected
    
        dfSamples['Filename'] = df['Filename']

    ## this part is for inserting 120Sn, 121Sb and 120Sn/121Sb into result file
    
    df_2 = pd.read_csv(fullnameSn,  delimiter='\t', index_col = False)
    df_2['blank'] = df_2['Filename'].str.contains('_Blk|_BLK')
    df_2['standard'] = df_2['Filename'].str.contains('_Nis|Ni1')
    df_2['sample'] = ~df_2['blank'] & ~df_2['standard']
    df_2['oldFilename'] = df_2['Filename']
    df_2['Filename'] = df_2['Filename'].map(lambda x: x.rstrip('_123'))
    df_2Samples=df_2[df_2['sample'] == True]
    df_2Samples = df_2Samples.groupby('Filename').mean()
        
    delta_cols = [col for col in dfSamples if 'd_' in col]
    deltaresults = dfSamples.groupby('Filename')[delta_cols].mean()
    deltaresults_round = np.round(deltaresults, 3)
    
    ## create stuff for export file
    delta2sd = dfSamples.groupby('Filename')[delta_cols].std(ddof = 0) * 2
    new_names_sd = [(i,'2SD+' + i) for i in delta2sd.iloc[:, 0:].columns.values]
    delta2sd.rename(columns = dict(new_names_sd), inplace = True)
    delta2sd_round = np.round(delta2sd, 3)
    results = pd.concat([deltaresults_round, delta2sd_round], axis = 1) 
    results = results[[item for items in zip(deltaresults_round.columns, delta2sd_round.columns) for item in items]]
        
    #insert needed stuff for export file
    results['120Sn'] = df_2Samples['120Sn'].round(1)
    results['121Sb'] = df_2Samples['121Sb'].round(1)
    results['120Sn/121Sb'] = (df_2Samples['120Sn'] / df_2Samples['121Sb']).round(2)
    
    #dSn
    df_ratio_delta = pd.DataFrame([{'116Sn/120Sn': -4,'118Sn/120Sn': -2, '122Sn/120Sn': 2, '124Sn/120Sn': 4, '124Sn/116Sn': 8, '122Sn/116Sn': 6}])
    df_4 = dfSamples.copy()
    df_4.drop(df_4.columns[1:11], axis = 1, inplace = True)
    col4_result = list(df_4.columns)
    col4_result = [t.replace('d_','') for t in col4_result]
    df_4.columns =col4_result
    df_4 = df_4.set_index('Filename')
    df_4 = df_4.astype(float)
    
    def cal(row):
        reg = np.polyfit(df_ratio_delta.values[0], row.values, 1)
        predict = np.poly1d(reg) # Slope and intercept
        trend = np.polyval(reg, df_ratio_delta)
        std = row.std() # Standard deviation
        r2 = np.round(r2_score(row.values, predict(df_ratio_delta.T)), 5) #R-squared
        df5 = pd.DataFrame([predict])
        df5.columns = ['dSn', 'Intercept']
        Slope = df5['dSn'].to_string(dtype=False)
        Slope = Slope[3:]
        Intercept = df5['Intercept'].to_string(dtype=False)
        Intercept = Intercept[3:]
        return Slope, Intercept, r2
    
    df_4[['dSn', 'intercept', 'r2']] = df_4.apply(cal, axis=1, result_type='expand')
    df_4[['dSn', 'intercept']] = df_4[['dSn', 'intercept']].astype(float)
    df_4.to_csv(plotSn, sep = '\t', header = True)
    
    df_6 = df_4[['dSn', 'intercept', 'r2']]
    df_6 = df_6.reset_index(drop = True)
    dfSamples = pd.concat([dfSamples, df_6], axis = 1)
    dSn_mean = np.round(df_4.groupby('Filename')['dSn'].mean(), 3)
    dSn_2sd = np.round(df_4.groupby('Filename')['dSn'].std(ddof = 0) * 2, 3)
    dSn_2sd = dSn_2sd.rename('2SD')

    results = pd.concat([results, dSn_mean], axis = 1)
    results = pd.concat([results, dSn_2sd], axis = 1)
        
    
    ## Insert correct header
    col_result = list(results.columns)
    col_result.insert(0, 'Lab Nummer')
    col_result = [l.replace('_','') for l in col_result]
    col_result = [i.split('+', 1)[0] for i in col_result]
    
    results.to_csv(resultnameSn, sep='\t', mode="w", header=True, index_label='Index_name')
    
    ## Insert Messdatum in results and other needed columns
    
    infiles = infile + '/'
    filehead = [x for x in os.listdir(infiles) if x .endswith('exp') and 'Nis01' in x]
    filehead = infiles + str(filehead)[2:-2]
        
    df3 = pd.read_csv(filehead, skiprows = 1, nrows = 1, sep=' |\t', engine='python', index_col = False)
    df3.columns = range(df3.shape[1])
    df3.drop(df3.columns[0], axis = 1, inplace = True)
    df3.drop(df3.columns[1:3], axis = 1, inplace = True)
            
    reread = pd.read_csv(resultnameSn, sep='\t', index_col=False) # names = col_result,
    reread.columns = col_result
    reread.columns = reread.columns.str.replace('2SD', '2SD')
    
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('_').str.get(-1)  #make Filename nice
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('.').str.get(0)
    reread.insert(1,'Messdatum', '')
    reread.insert(0, 'Objekt', '')
    reread.insert(0, 'Orig. Nummer', '')
    reread.insert(0, 'Ma Nummer', '')
    
    reread['Messdatum'] = df3.values[0][0]
    
    Nist = [x for x in os.listdir(infiles) if x .endswith('exp') and 'Nis1' or 'Nis01' in x]
    if len(Nist) > 0:
        reread['Referenz'] = 'NIST SRM 3161a'
        reread['Verwenden'] = ''
        reread['Bemerkung'] =''
    else:
        pass
    
    
    reread.to_csv(resultnameSn, sep = '\t', header=True, index=False)
    
###############################################################    
def ssbPSn():
    global fullnameSn
    global resultnameSn
    
    # read data into dataframe and set headers
    df = pd.read_csv(fullnameSn,  delimiter='\t')
    
    df.drop(df.columns[0], axis = 1, inplace = True)
    df.drop(df.columns[1:27], axis = 1, inplace = True)
    cols = list(df.columns)
    del cols[0]
    
    # create now column holding the booleans
    df['blank'] = df['Filename'].str.contains('_Blk|_BLK|BLK|Blk')
    
    df['standard'] = df['Filename'].str.contains('JM|jm')
    df['sample'] = ~df['blank'] & ~df['standard']

    # drop _1, _2 and _3 in the filenames
    df['oldFilename'] = df['Filename']
    df['Filename'] = df['Filename'].map(lambda x: x.rstrip('_123'))

    # choose only samples
    dfSamples=df[df['sample'] == True]

    colsSamples = list(dfSamples.columns)
    del colsSamples[0]
    del colsSamples[-4:-1]
    del colsSamples[-1]

    delta = pd.DataFrame(columns = colsSamples)
    new_names = [(i,'d_' +i) for i in delta.iloc[:, 0:].columns.values]
    delta.rename(columns = dict(new_names), inplace = True)
    delta.columns = delta.columns.str.rstrip('_corrected')

    dfSamples = pd.concat([dfSamples, delta])
    
    # detect the standard before and after the sample
    i = 0
    for sampleIndex in dfSamples.index:
        dfBefore=df.iloc[0:sampleIndex]
        dfBeforeInverse=dfBefore[::-1]
        dfStandardBefore = dfBeforeInverse[dfBeforeInverse['standard'] == True]
        indexBefore = dfStandardBefore.index[0]
        df.at[sampleIndex, 'standardBefore'] = indexBefore
        lastIndex = df.index[-1]
        dfAfter=df.iloc[sampleIndex:lastIndex]
        dfStandardAfter = dfAfter[dfAfter['standard'] == True]
        indexAfter = dfStandardAfter.index[0]
        df.at[sampleIndex, 'standardAfter'] = indexAfter
    
    
    # calculate the StdBefore and StdAfter for each column
        stdBeforeCu_ratio=df[cols].loc[df['standardBefore'].loc[sampleIndex]]
        stdAfterCu_ratio=df[cols].loc[df['standardAfter'].loc[sampleIndex]]
    
        meanStdCu_ratio = (stdBeforeCu_ratio + stdAfterCu_ratio)/2
    
    # calculate delta and save them into new dataframe
        sampleSn_ratiodelta_corrected = (((dfSamples[colsSamples].loc[sampleIndex]/meanStdCu_ratio)-1)*1000)
        sampleSn_ratiodelta_corrected.index = list(delta)
        dfSamples.loc[sampleIndex] = sampleSn_ratiodelta_corrected
    
        dfSamples['Filename'] = df['Filename']

    ## this part is for inserting 120Sn, 121Sb and 120Sn/121Sb into result file
    df_2 = pd.read_csv(fullnameSn,  delimiter='\t', index_col = False)
    df_2['blank'] = df_2['Filename'].str.contains('_Blk|_BLK')
    df_2['standard'] = df_2['Filename'].str.contains('JM|jm')
    df_2['sample'] = ~df_2['blank'] & ~df_2['standard']
    df_2['oldFilename'] = df_2['Filename']
    df_2['Filename'] = df_2['Filename'].map(lambda x: x.rstrip('_123'))
    df_2Samples=df_2[df_2['sample'] == True]
    df_2Samples = df_2Samples.groupby('Filename').mean()
    
    delta_cols = [col for col in dfSamples if 'd_' in col]
    deltaresults = dfSamples.groupby('Filename')[delta_cols].mean()
    deltaresults_round = np.round(deltaresults, 3)
    
    ## create stuff for export file
    delta2sd = dfSamples.groupby('Filename')[delta_cols].std(ddof = 0) * 2
    new_names_sd = [(i,'2SD+' + i) for i in delta2sd.iloc[:, 0:].columns.values]
    delta2sd.rename(columns = dict(new_names_sd), inplace = True)
    delta2sd_round = np.round(delta2sd, 3)
    results = pd.concat([deltaresults_round, delta2sd_round], axis = 1) 
    results = results[[item for items in zip(deltaresults_round.columns, delta2sd_round.columns) for item in items]]

    #insert needed stuff for export file
    results['120Sn'] = df_2Samples['120Sn'].round(1)
    results['121Sb'] = df_2Samples['121Sb'].round(1)
    results['120Sn/121Sb'] = (df_2Samples['120Sn'] / df_2Samples['121Sb']).round(2)
    
    #dSn
    df_ratio_delta = pd.DataFrame([{'116Sn/120Sn': -4, '117Sn/120Sn': -3, '118Sn/120Sn': -2, '119Sn/120Sn': -1, '122Sn/120Sn': 2, '124Sn/120Sn': 4, '124Sn/116Sn': 8, '122Sn/116Sn': 6, '117Sn/119Sn': -2}])
    df_4 = dfSamples.copy()
    df_4.drop(df_4.columns[1:14], axis = 1, inplace = True)
    col4_result = list(df_4.columns)
    col4_result = [t.replace('d_','') for t in col4_result]
    df_4.columns =col4_result
    df_4 = df_4.set_index('Filename')
    df_4 = df_4.astype(float)
    
    def cal(row):
        reg = np.polyfit(df_ratio_delta.values[0], row.values, 1)
        predict = np.poly1d(reg) # Slope and intercept
        trend = np.polyval(reg, df_ratio_delta)
        std = row.std() # Standard deviation
        r2 = np.round(r2_score(row.values, predict(df_ratio_delta.T)), 5) #R-squared
        df5 = pd.DataFrame([predict])
        df5.columns = ['dSn', 'Intercept']
        Slope = df5['dSn'].to_string(dtype=False)
        Slope = Slope[3:]
        Intercept = df5['Intercept'].to_string(dtype=False)
        Intercept = Intercept[3:]
        return Slope, Intercept, r2
    
    df_4[['dSn', 'intercept', 'r2']] = df_4.apply(cal, axis=1, result_type='expand')
    df_4[['dSn', 'intercept']] = df_4[['dSn', 'intercept']].astype(float)
    df_4.to_csv(plotSn, sep = '\t', header = True)
    
    df_6 = df_4[['dSn', 'intercept', 'r2']]
    df_6 = df_6.reset_index(drop = True)
    dfSamples = pd.concat([dfSamples, df_6], axis = 1)
    dSn_mean = np.round(df_4.groupby('Filename')['dSn'].mean(), 3)
    dSn_2sd = np.round(df_4.groupby('Filename')['dSn'].std(ddof = 0) * 2, 3)
    dSn_2sd = dSn_2sd.rename('2SD')

    results = pd.concat([results, dSn_mean], axis = 1)
    results = pd.concat([results, dSn_2sd], axis = 1)
    
    
    
    ## Insert correct header
    col_result = list(results.columns)
    col_result.insert(0, 'Lab Nummer')
    col_result = [l.replace('_','') for l in col_result]
    col_result = [i.split('+', 1)[0] for i in col_result]
    
    results.to_csv(resultnameSn, sep='\t', mode="w", header=True, index_label='Index_name')

    ## Insert Messdatum in results and other needed columns
    
    infiles = infile + '/'
    filehead = [x for x in os.listdir(infiles) if x .endswith('exp') and 'JM01' in x]
    filehead = infiles + str(filehead)[2:-2]
        
    df3 = pd.read_csv(filehead, skiprows = 1, nrows = 1, sep=' |\t', engine='python', index_col = False)
    df3.columns = range(df3.shape[1])
    df3.drop(df3.columns[0], axis = 1, inplace = True)
    df3.drop(df3.columns[1:3], axis = 1, inplace = True)
             

        
    reread = pd.read_csv(resultnameSn, sep='\t', index_col=False)
    reread.columns = col_result
    reread.columns = reread.columns.str.replace('2SD', '2SD')
    
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('_').str.get(-1)  #make Filename nice
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('.').str.get(0)
    reread['Lab Nummer'] = reread['Lab Nummer'].str.replace('22', '22-')
    reread.insert(1,'Messdatum', '')
    reread.insert(0, 'Objekt', '')
    reread.insert(0, 'Orig. Nummer', '')
    reread.insert(0, 'Ma Nummer', '')
    
    reread['Messdatum'] = df3.values[0][0]
    
    PSn = [x for x in os.listdir(infiles) if x .endswith('exp') and 'JM' in x]
    if len(PSn) > 0:
        reread['Referenz'] = 'PSn'
        reread['Verwenden'] = ''
        reread['Bemerkung'] =''
    else:
        pass
    
    
    reread.to_csv(resultnameSn, sep = '\t', header=True, index=False)
    
    
#################################################
def ssbAg():
    global fullnameAg
    global resultnameAg
    
    # read data into dataframe and set headers
    df = pd.read_csv(fullnameAg,  delimiter='\t')
    
    df.drop(df.columns[0], axis = 1, inplace = True)
    df.drop(df.columns[1:27], axis = 1, inplace = True)
    cols = list(df.columns)
    del cols[0]
    
    # create now column holding the booleans
    df['blank'] = df['Filename'].str.contains('_Blk|_BLK|BLK|Blk')
    
    df['standard'] = df['Filename'].str.contains('JM|jm')
    df['sample'] = ~df['blank'] & ~df['standard']

    # drop _1, _2 and _3 in the filenames
    df['oldFilename'] = df['Filename']
    df['Filename'] = df['Filename'].map(lambda x: x.rstrip('_123'))

    # choose only samples
    dfSamples=df[df['sample'] == True]

    colsSamples = list(dfSamples.columns)
    del colsSamples[0]
    del colsSamples[-4:-1]
    del colsSamples[-1]

    delta = pd.DataFrame(columns = colsSamples)
    new_names = [(i,'d_' +i) for i in delta.iloc[:, 0:].columns.values]
    delta.rename(columns = dict(new_names), inplace = True)
    delta.columns = delta.columns.str.rstrip('_corrected')

    dfSamples = pd.concat([dfSamples, delta])
    
# detect the standard before and after the sample
    i = 0
    for sampleIndex in dfSamples.index:
        dfBefore=df.iloc[0:sampleIndex]
        dfBeforeInverse=dfBefore[::-1]
        dfStandardBefore = dfBeforeInverse[dfBeforeInverse['standard'] == True]
        indexBefore = dfStandardBefore.index[0]
        df.at[sampleIndex, 'standardBefore'] = indexBefore
        lastIndex = df.index[-1]
        dfAfter=df.iloc[sampleIndex:lastIndex]
        dfStandardAfter = dfAfter[dfAfter['standard'] == True]
        indexAfter = dfStandardAfter.index[0]
        df.at[sampleIndex, 'standardAfter'] = indexAfter
    
    
    # calculate the StdBefore and StdAfter for each column
        stdBeforeCu_ratio=df[cols].loc[df['standardBefore'].loc[sampleIndex]]
        stdAfterCu_ratio=df[cols].loc[df['standardAfter'].loc[sampleIndex]]
    
        meanStdCu_ratio = (stdBeforeCu_ratio + stdAfterCu_ratio)/2
    
    # calculate delta and save them into new dataframe
        sampleSn_ratiodelta_corrected = (((dfSamples[colsSamples].loc[sampleIndex]/meanStdCu_ratio)-1)*1000)
        sampleSn_ratiodelta_corrected.index = list(delta)
        dfSamples.loc[sampleIndex] = sampleSn_ratiodelta_corrected
    
        dfSamples['Filename'] = df['Filename']

    ## this part is for inserting 120Sn, 121Sb and 120Sn/121Sb into result file
    
    df_2 = pd.read_csv(fullnameSn,  delimiter='\t', index_col = False)
    df_2['blank'] = df_2['Filename'].str.contains('_Blk|_BLK')
    df_2['standard'] = df_2['Filename'].str.contains('JM|jm')
    df_2['sample'] = ~df_2['blank'] & ~df_2['standard']
    df_2['oldFilename'] = df_2['Filename']
    df_2['Filename'] = df_2['Filename'].map(lambda x: x.rstrip('_123'))
    df_2Samples=df_2[df_2['sample'] == True]
    df_2Samples = df_2Samples.groupby('Filename').mean()
    
    
   # print(type(sampleSn_ratiodelta_corrected))
    delta_cols = [col for col in dfSamples if 'd_' in col]
    deltaresults = dfSamples.groupby('Filename')[delta_cols].mean()
    deltaresults_round = np.round(deltaresults, 3)
    
    ## create stuff for export file
    delta2sd = dfSamples.groupby('Filename')[delta_cols].std(ddof = 0) * 2
    new_names_sd = [(i,'2SD+' + i) for i in delta2sd.iloc[:, 0:].columns.values]
    delta2sd.rename(columns = dict(new_names_sd), inplace = True)
    delta2sd_round = np.round(delta2sd, 3)
    results = pd.concat([deltaresults_round, delta2sd_round], axis = 1) 
    results = results[[item for items in zip(deltaresults_round.columns, delta2sd_round.columns) for item in items]]

    #insert needed stuff for export file
    results['120Sn'] = df_2Samples['120Sn'].round(1)
    results['121Sb'] = df_2Samples['121Sb'].round(1)
    results['120Sn/121Sb'] = (df_2Samples['120Sn'] / df_2Samples['121Sb']).round(2)
    
    #dSn
    df_ratio_delta = pd.DataFrame([{'116Sn/120Sn': -4, '117Sn/120Sn': -3, '118Sn/120Sn': -2, '119Sn/120Sn': -1, '122Sn/120Sn': 2, '124Sn/120Sn': 4, '124Sn/116Sn': 8, '122Sn/116Sn': 6, '117Sn/119Sn': -2}])
    df_4 = dfSamples.copy()
    df_4.drop(df_4.columns[1:14], axis = 1, inplace = True)
    col4_result = list(df_4.columns)
    col4_result = [t.replace('d_','') for t in col4_result]
    df_4.columns =col4_result
    df_4 = df_4.set_index('Filename')
    df_4 = df_4.astype(float)
    
    def cal(row):
        reg = np.polyfit(df_ratio_delta.values[0], row.values, 1)
        predict = np.poly1d(reg) # Slope and intercept
        trend = np.polyval(reg, df_ratio_delta)
        std = row.std() # Standard deviation
        r2 = np.round(r2_score(row.values, predict(df_ratio_delta.T)), 5) #R-squared
        df5 = pd.DataFrame([predict])
        df5.columns = ['dSn', 'Intercept']
        Slope = df5['dSn'].to_string(dtype=False)
        Slope = Slope[3:]
        Intercept = df5['Intercept'].to_string(dtype=False)
        Intercept = Intercept[3:]
        return Slope, Intercept, r2
    
    df_4[['dSn', 'intercept', 'r2']] = df_4.apply(cal, axis=1, result_type='expand')
    df_4[['dSn', 'intercept']] = df_4[['dSn', 'intercept']].astype(float)
    df_4.to_csv(plotSn, sep = '\t', header = True)
    
    df_6 = df_4[['dSn', 'intercept', 'r2']]
    df_6 = df_6.reset_index(drop = True)
    dfSamples = pd.concat([dfSamples, df_6], axis = 1)
    dSn_mean = np.round(df_4.groupby('Filename')['dSn'].mean(), 3)
    dSn_2sd = np.round(df_4.groupby('Filename')['dSn'].std(ddof = 0) * 2, 3)
    dSn_2sd = dSn_2sd.rename('2SD')

    results = pd.concat([results, dSn_mean], axis = 1)
    results = pd.concat([results, dSn_2sd], axis = 1)
    
    
    
    ## Insert correct header
    col_result = list(results.columns)
    col_result.insert(0, 'Lab Nummer')
    col_result = [l.replace('_','') for l in col_result]
    col_result = [i.split('+', 1)[0] for i in col_result]
    
    results.to_csv(resultnameSn, sep='\t', mode="w", header=True, index_label='Index_name')

    ## Insert Messdatum in results and other needed columns
    
    infiles = infile + '/'
    filehead = [x for x in os.listdir(infiles) if x .endswith('exp') and 'JM01' in x]
    filehead = infiles + str(filehead)[2:-2]
        
    df3 = pd.read_csv(filehead, skiprows = 1, nrows = 1, sep=' |\t', engine='python', index_col = False)
    df3.columns = range(df3.shape[1])
    df3.drop(df3.columns[0], axis = 1, inplace = True)
    df3.drop(df3.columns[1:3], axis = 1, inplace = True)
             

        
    reread = pd.read_csv(resultnameSn, sep='\t', index_col=False) 
    reread.columns = col_result
    reread.columns = reread.columns.str.replace('2SD', '2SD')
    
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('_').str.get(-1)  #make Filename nice
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('.').str.get(0)
    reread['Lab Nummer'] = reread['Lab Nummer'].str.replace('22', '22-')
    reread.insert(1,'Messdatum', '')
    reread.insert(0, 'Objekt', '')
    reread.insert(0, 'Orig. Nummer', '')
    reread.insert(0, 'Ma Nummer', '')
    
    reread['Messdatum'] = df3.values[0][0]
    
    PSn = [x for x in os.listdir(infiles) if x .endswith('exp') and 'JM' in x]
    if len(PSn) > 0:
        reread['Referenz'] = 'PSn'
        reread['Verwenden'] = ''
        reread['Bemerkung'] =''
    else:
        pass
    
    
    reread.to_csv(resultnameSn, sep = '\t', header=True, index=False)
    
def ssbLi():
    global fullnameLi
    global resultnameLi
    global infile

    df = pd.read_csv(fullnameLi,  delimiter='\t')

    df['blank'] = df['Filename'].str.contains('_Blk|_BLK')
    df['standard'] = df['Filename'].str.contains('_s0|s1')
    df['sample'] = ~df['blank'] & ~df['standard']

    # drop _1, _2 and _3 in the filenames
    df['oldFilename'] = df['Filename']
    df['Filename'] = df['Filename'].map(lambda x: x.rstrip('_123'))

    # choose only samples
    dfSamples=df[df['sample'] == True]

# detect the standard before and after the sample
    i = 0
    for sampleIndex in dfSamples.index:
        # find standards before the Lirrent index
        # sub dataframe that holds only the df before the Lirretn sample
        dfBefore=df.iloc[0:sampleIndex]
        # invert in order to earch backward
        dfBeforeInverse=dfBefore[::-1]
        # get the rows that hold a standard
        dfStandardBefore = dfBeforeInverse[dfBeforeInverse['standard'] == True]
        # index of the standard closest to the sample 
        indexBefore = dfStandardBefore.index[0]
        # safe index in dataframe at position of the sample
        df.at[sampleIndex, 'standardBefore'] = indexBefore
        # find standards after the Lirrent index
        lastIndex = df.index[-1]
        dfAfter=df.iloc[sampleIndex:lastIndex]
        dfStandardAfter = dfAfter[dfAfter['standard'] == True]
        indexAfter = dfStandardAfter.index[0]
        df.at[sampleIndex, 'standardAfter'] = indexAfter
    
        # calLilate the delta and the 2sd
        stdBeforeLi_ratio=df['7Li/6Li'].loc[df['standardBefore'].loc[sampleIndex]]
        stdAfterLi_ratio=df['7Li/6Li'].loc[df['standardAfter'].loc[sampleIndex]]
        Filename = df['Filename']
        meanStdLi_ratio = (stdBeforeLi_ratio + stdAfterLi_ratio)/2
        
        sampleLi_ratiodelta_ = ((dfSamples['7Li/6Li'].loc[sampleIndex]/meanStdLi_ratio)-1)*1000
        dfSamples.loc[sampleIndex, 'd7Li/6Li '] = sampleLi_ratiodelta_

    deltaresults = dfSamples.groupby('Filename')['d7Li/6Li '].mean(['delta7Li/6Li '])
    deltaresults_round = np.round(deltaresults, 2)

    delta2sd = dfSamples.groupby('Filename')['d7Li/6Li '].std(ddof = 0) * 2
    delta2sd_round = np.round(delta2sd, 2)
    resultuncorr = pd.concat([deltaresults_round, delta2sd_round], axis = 1) 
    resultuncorr.to_csv(resultnameLi, sep='\t', mode="w", header=False, index_label='Index_name')
    reread = pd.read_csv(resultnameLi, sep='\t', names = ['Lab Nummer', 'd7Li/6Li', '2SD_'], index_col=False)
    
    ##get date
    infiles = infile + '/'
    filehead = [x for x in os.listdir(infiles) if x .endswith('exp') and 's01' in x]
    filehead = infiles + str(filehead)[2:-2]
    df3 = pd.read_csv(filehead, skiprows = 1, nrows = 1, sep=' |\t', engine='python', index_col = False)
    df3.columns = range(df3.shape[1])
    df3.drop(df3.columns[0], axis = 1, inplace = True)
    df3.drop(df3.columns[1:3], axis = 1, inplace = True)

    reread.columns = reread.columns.str.replace('2SD', '2SD')
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('_').str.get(-1)  #make Filename nice
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('.').str.get(0)
    
    reread.insert(1,'Messdatum', '')
    reread.insert(0, 'Objekt', '')
    reread.insert(0, 'Orig. Nummer', '')
    reread.insert(0, 'Ma Nummer', '')
    
    reread['Messdatum'] = df3.values[0][0]
    
    Nist = [x for x in os.listdir(infiles) if x .endswith('exp') and 's1' or 's01'in x]
    if len(Nist) > 0:
        reread['Referenz'] = 'L-SVEC'
        reread['Verwenden'] = ''
        reread['Bemerkung'] ='Bracketing, Aridus, Ni-Cone'
    else:
        pass
    
    
    reread.to_csv(resultnameLi, sep = '\t', header=True, index=False)
    reread.columns = ['\u03B47/6Li' if x=='d7Li/6Li' else x for x in reread.columns]
    reread.columns = ['2SD' if x=='2SD_' else x for x in reread.columns]
    reread.to_excel(resultnameLiExcel, index=False)   

def ssbSb():
    global fullnameSb
    global resultnameSb
    global infile

    df = pd.read_csv(fullnameSb,  delimiter='\t')

    df['blank'] = df['Filename'].str.contains('_Blk|_BLK')
    df['standard'] = df['Filename'].str.contains('Nis|_Nis1|_Ni0')
    df['sample'] = ~df['blank'] & ~df['standard']

    # drop _1, _2 and _3 in the filenames
    df['oldFilename'] = df['Filename']
    df['Filename'] = df['Filename'].map(lambda x: x.rstrip('_123'))

    # choose only samples
    dfSamples=df[df['sample'] == True]

# detect the standard before and after the sample
    i = 0
    for sampleIndex in dfSamples.index:
        # find standards before the Sbrrent index
        # sub dataframe that holds only the df before the Sbrretn sample
        dfBefore=df.iloc[0:sampleIndex]
        # invert in order to earch backward
        dfBeforeInverse=dfBefore[::-1]
        # get the rows that hold a standard
        dfStandardBefore = dfBeforeInverse[dfBeforeInverse['standard'] == True]
        # index of the standard closest to the sample 
        indexBefore = dfStandardBefore.index[0]
        # safe index in dataframe at position of the sample
        df.at[sampleIndex, 'standardBefore'] = indexBefore
        # find standards after the Sbrrent index
        lastIndex = df.index[-1]
        dfAfter=df.iloc[sampleIndex:lastIndex]
        dfStandardAfter = dfAfter[dfAfter['standard'] == True]
        indexAfter = dfStandardAfter.index[0]
        df.at[sampleIndex, 'standardAfter'] = indexAfter
    
        # calSblate the delta and the 2sd
        stdBeforeSb_ratio=df['123Sb/121Sb_corrected'].loc[df['standardBefore'].loc[sampleIndex]]
        stdAfterSb_ratio=df['123Sb/121Sb_corrected'].loc[df['standardAfter'].loc[sampleIndex]]
        stdBeforeSb_ratio_uncorr=df['123Sb/121Sb'].loc[df['standardBefore'].loc[sampleIndex]]
        stdAfterSb_ratio_uncorr=df['123Sb/121Sb'].loc[df['standardAfter'].loc[sampleIndex]]
        Filename = df['Filename']
        meanStdSb_ratio = (stdBeforeSb_ratio + stdAfterSb_ratio)/2
        meanStdSb_ratio_uncorr= (stdBeforeSb_ratio_uncorr + stdAfterSb_ratio_uncorr) / 2
        
        sampleSb_ratiodelta_uncorr = ((dfSamples['123Sb/121Sb'].loc[sampleIndex]/meanStdSb_ratio_uncorr)-1)*1000
        dfSamples.loc[sampleIndex, 'd123Sb/121Sb uncorr'] = sampleSb_ratiodelta_uncorr
        sampleSb_ratiodelta_corrected = ((dfSamples['123Sb/121Sb_corrected'].loc[sampleIndex]/meanStdSb_ratio)-1)*1000
        dfSamples.loc[sampleIndex, 'd123Sb/121Sb corrected'] = sampleSb_ratiodelta_corrected

    deltaresults = dfSamples.groupby('Filename')['d123Sb/121Sb corrected'].mean(['delta123Sb/121Sb corrected'])
    deltaresults_uncorr = dfSamples.groupby('Filename')['d123Sb/121Sb uncorr'].mean(['delta123Sb/121Sb uncorr'])
    deltaresults_round = np.round(deltaresults, 3)
    deltaresults_uncorr_round = np.round(deltaresults_uncorr, 3)

    delta2sd = dfSamples.groupby('Filename')['d123Sb/121Sb corrected'].std(ddof = 0) * 2
    delta2sd_uncorr = dfSamples.groupby('Filename')['d123Sb/121Sb uncorr'].std(ddof = 0) * 2
    delta2sd_round = np.round(delta2sd, 3)
    delta2sd_uncorr_round = np.round(delta2sd_uncorr, 3)
    resultuncorr = pd.concat([deltaresults_uncorr_round, delta2sd_uncorr_round, deltaresults_round, delta2sd_round], axis = 1) 
    resultuncorr.to_csv(resultnameSb, sep='\t', mode="w", header=False, index_label='Index_name')
    reread = pd.read_csv(resultnameSb, sep='\t', names = ['Lab Nummer', 'd123Sb/121Sb_uncorr', '2SD_uncorr', 'd123Sb/121Sb_corrected', '2SD_corrected'], index_col=False)
    
    ##get date
    infiles = infile + '/'
    filehead = [x for x in os.listdir(infiles) if x .endswith('exp') and 's01' in x]
    filehead = infiles + str(filehead)[2:-2]
    df3 = pd.read_csv(filehead, skiprows = 1, nrows = 1, sep=' |\t', engine='python', index_col = False)
    df3.columns = range(df3.shape[1])
    df3.drop(df3.columns[0], axis = 1, inplace = True)
    df3.drop(df3.columns[1:3], axis = 1, inplace = True)

    reread.columns = reread.columns.str.replace('2SD', '2SD')
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('_').str.get(-1)  #make Filename nice
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('.').str.get(0)
    
    reread.insert(1,'Messdatum', '')
    reread.insert(0, 'Objekt', '')
    reread.insert(0, 'Orig. Nummer', '')
    reread.insert(0, 'Ma Nummer', '')
    
    reread['Messdatum'] = df3.values[0][0]
    
    Nist = [x for x in os.listdir(infiles) if x .endswith('exp') and 's1' or 's01'in x]
    if len(Nist) > 0:
        reread['Referenz'] = 'NIST SRM 976'
        reread['Verwenden'] = ''
        reread['Bemerkung'] ='Bracketing, Sprühkammer, Ni-Cone'
    else:
        pass
    
    
    reread.to_csv(resultnameSb, sep = '\t', header=True, index=False)
    reread.drop(reread.columns[5:7], axis=1, inplace = True)
    reread.columns = ['\u03B465/121Sb' if x=='d123Sb/121Sb_corrected' else x for x in reread.columns]
    reread.columns = ['2SD' if x=='2SD_corrected' else x for x in reread.columns]
    reread.to_excel(resultnameSbExcel, index=False)
    