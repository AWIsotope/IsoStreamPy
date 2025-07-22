"""
@author: Andreas Wittke
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import time
import seaborn as sb
from csv import DictReader
sb.set()
from sklearn.metrics import mean_squared_error, r2_score
plt.ioff()
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 16})
import sys
sys.path.append('')
from modules.config import WorkspaceVariableInput, outfile_plt_Cu, outfile_corrected_raw_data_Cu, outfile_results_Cu, outfile_plt_Cu_Std 
from modules import outlierSn 
infile = WorkspaceVariableInput
save_dStandards = os.path.join(outfile_results_Cu, 'Delta of Cu Standards.csv') 
save_dStandardsExcel = os.path.join(outfile_results_Cu, 'Delta of Cu Standards.xlsx')

def dCu_Standards():

    global outfile_plt_Cu_Std
    global outfile_results_Cu
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu
    save = outfile_plt_Cu + '/Standards'
    print(outfile_corrected_raw_data_Cu)
        
    fullname = os.path.join(outfile_results_Cu, 'Cu_export.csv')
    pd.options.mode.chained_assignment = None
    df = pd.read_csv(fullname, sep='\t')
    
    
    True_Nickel = 0.036345 / 0.262231
    
    df = df[df['Filename'].str.contains('Blk')==False]
    df = df[df['Filename'].str.contains('s0|s1')]
    for index in df.index:
        df['60Ni/61Ni'] = (df['60Ni'] / df['61Ni'])
        
    True_60Ni_61Ni_mean = np.mean(df['60Ni/61Ni'])
    True_60Ni_61Ni_mean_2SD = np.std(df['60Ni/61Ni']) * 2
    
    True_65Cu_63Cu_mean = np.mean(df['65Cu/63Cu'])
    True_65Cu_63Cu_mean_2SD = np.std(df['65Cu/63Cu']) * 2
    
    for index in df.index:
        df['d60Ni'] = ((df['60Ni/61Ni'] / True_60Ni_61Ni_mean) - 1) * 1000 
        df['d65Cu'] = ((df['65Cu/63Cu'] / True_65Cu_63Cu_mean) - 1) * 1000 
    
    
    df['d60Ni_Std_mean'] = np.mean(df['d60Ni'])
    df['d65Cu_Std_mean'] = np.mean(df['d65Cu'])

    df['d60Ni_Std_mean_2SD'] = np.std(df['d60Ni']) * 2
    df['d65Cu_Std_mean_2SD'] = np.std(df['d65Cu']) * 2
    
    
    reg = np.polyfit(df['63Cu'], df['60Ni'], 1)
    predict = np.poly1d(reg) # Slope and intercept
    trend = np.polyval(reg, df['63Cu'])
    std = df['60Ni'].std() # Standard deviation
    r2 = np.round(r2_score(df['60Ni'], predict(df['63Cu'])), 5) #R-squared
    r2string = str(r2)
    
    df.insert(1, 'ID', range(1, 1+len(df)))
    df['ID'] = df['ID'].round(decimals=0).astype(object)
    
    df.to_csv(save_dStandards, sep = '\t', index = False, header = True)
    df.to_excel(save_dStandardsExcel, index = False)