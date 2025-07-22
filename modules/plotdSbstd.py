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
from modules.config import WorkspaceVariableInput, outfile_plt_Sb, outfile_corrected_raw_data_Sb, outfile_results_Sb, outfile_plt_Sb_Std
from modules import outlierSn 
infile = WorkspaceVariableInput
save_dStandards = os.path.join(outfile_results_Sb, 'Delta of Sb Standards.csv') 
save_dStandardsExcel = os.path.join(outfile_results_Sb, 'Delta of Sb Standards.xlsx')

def dSb_Standards():

    global outfile_plt_Sb_Std
    global outfile_results_Sb
    global outfile_corrected_raw_data_Sb
    global outfile_plt_Sb
    save = outfile_plt_Sb + '/Standards'
    print(outfile_corrected_raw_data_Sb)
        
    fullname = os.path.join(outfile_results_Sb, 'Sb_export.csv')
    pd.options.mode.chained_assignment = None
    df = pd.read_csv(fullname, sep='\t')
    
    
    True_Nickel = 0.036345 / 0.262231
    
    df = df[df['Filename'].str.contains('Blk')==False]
    df = df[df['Filename'].str.contains('s0|s1')]
    for index in df.index:
        df['118Sn/120Sn'] = (df['118Sn'] / df['120Sn'])
        
    True_118Sn_120Sn_mean = np.mean(df['118Sn/120Sn'])
    True_118Sn_120Sn_mean_2SD = np.std(df['118Sn/120Sn']) * 2
    
    True_123Sb_121Sb_mean = np.mean(df['123Sb/121Sb'])
    True_123Sb_121Sb_mean_2SD = np.std(df['123Sb/121Sb']) * 2
    
    for index in df.index:
        df['d118Sn'] = ((df['118Sn/120Sn'] / True_118Sn_120Sn_mean) - 1) * 1000 
        df['d123Sb'] = ((df['123Sb/121Sb'] / True_123Sb_121Sb_mean) - 1) * 1000 
    
    
    df['d118Sn_Std_mean'] = np.mean(df['d118Sn'])
    df['d123Sb_Std_mean'] = np.mean(df['d123Sb'])

    df['d118Sn_Std_mean_2SD'] = np.std(df['d118Sn']) * 2
    df['d123Sb_Std_mean_2SD'] = np.std(df['d123Sb']) * 2
    
    
    reg = np.polyfit(df['121Sb'], df['118Sn'], 1)
    predict = np.poly1d(reg) # Slope and intercept
    trend = np.polyval(reg, df['121Sb'])
    std = df['118Sn'].std() # Standard deviation
    r2 = np.round(r2_score(df['118Sn'], predict(df['121Sb'])), 5) #R-squared
    r2string = str(r2)
    
    df.insert(1, 'ID', range(1, 1+len(df)))
    df['ID'] = df['ID'].round(decimals=0).astype(object)
    
    df.to_csv(save_dStandards, sep = '\t', index = False, header = True)
    df.to_excel(save_dStandardsExcel, index = False)