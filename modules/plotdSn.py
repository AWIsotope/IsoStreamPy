# -*- coding: utf-8 -*-
"""
@author: Andreas Wittke
"""



import pandas as pd
import os
from pathlib import Path
from statistics import mean
from scipy import stats
import sys
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None 
sys.path.append('')
import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from modules.config import outfile_results_Sn, outfile_plt_Sn
from modules.config import WorkspaceVariableInput
infile = WorkspaceVariableInput
fullnameSn = os.path.join(outfile_results_Sn, 'Sn_export.csv')
resultnameSn = os.path.join(outfile_results_Sn, 'Sn_delta.csv')
plotSn = os.path.join(outfile_results_Sn, 'Sn_plots.csv')



def plotdSn():
    global fullnameSn
    global resultnameSn
    
    df = pd.read_csv(fullnameSn,  delimiter='\t', index_col = False)
    df.drop(df.columns[0], axis = 1, inplace = True)
    df.drop(df.columns[1:27], axis = 1, inplace = True)
    cols = list(df.columns)
    del cols[0]
    # create booleans
    df['blank'] = df['Filename'].str.contains('_Blk|_BLK')
    df['standard'] = df['Filename'].str.contains('_Nis|_Nis1|_Ni0')
    df['sample'] = ~df['blank'] & ~df['standard']
    df['oldFilename'] = df['Filename']
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
        
    df['oldFilename'] = df['Filename']
    df['Filename'] = df['Filename'].map(lambda x: x.rstrip('_123'))
    df_4 = dfSamples.copy()

    df_4.drop(df_4.columns[1:14], axis = 1, inplace = True)
    # a = 1
    col4_result = list(df_4.columns)
    col4_result = [t.replace('d_','') for t in col4_result]
    df_4.columns =col4_result
    
    df_ratio_delta = pd.DataFrame([{'116Sn/120Sn': -4, '117Sn/120Sn': -3, '118Sn/120Sn': -2, '119Sn/120Sn': -1, '122Sn/120Sn': 2, '124Sn/120Sn': 4, '124Sn/116Sn': 8, '122Sn/116Sn': 6, '117Sn/119Sn': -2}])
    df_4 = df_4.set_index('Filename')
    df_4 = df_4.astype(float)
    
    for index, row in df_4.iterrows():
        start_time = time.time()
        slope, intercept, r, p, se = stats.linregress(df_ratio_delta.values[0], row.values)
        rsquared = r**2   
        end_timea = time.time()
        
        name = row.name

        fig, ax = plt.subplots(figsize=(15, 10))
        plt.xticks(np.arange(min(df_ratio_delta.values[0]), max(df_ratio_delta.values[0])+1, 1.0))
        ax.scatter(df_ratio_delta.values[0], row.values, zorder = 2)
        ax.margins(0.02, 0.3)
        ax.set_title(name)

        ax.plot(df_ratio_delta.values[0], intercept + slope*df_ratio_delta.values[0], linestyle = '-', color = 'r',label = 'Linear Regression', zorder =1)
        ax.plot([], [], ' ', label='R$^{2}$ = '+str(rsquared))
        ax.legend()
        basename = os.path.basename(str(row))
        basename = basename[:-16]
        fig.savefig(outfile_plt_Sn + basename +"_dSn.png", dpi = 75)
        print("Finished plotting dSn of " + name)
