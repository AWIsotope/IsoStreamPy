# -*- coding: utf-8 -*-
"""
This module containts the Outlier correction for Sn

@author: Andreas Wittke 
"""

"""Calculates the outliers. """

#import modules.config as conf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import time
import seaborn as sb
from csv import DictReader
sb.set_theme()
from sklearn.metrics import mean_squared_error, r2_score
plt.ioff()
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 16})

from modules.config import WorkspaceVariableInput, outfile_plt_Sn, outfile_corrected_raw_data_Sn, outfile_results_Sn, Baxter_Sn
import modules.plotSn as plotSn
infile = WorkspaceVariableInput
import pickle
from statistics import mean

file_path = os.path.join("modules", "variable_data.pkl")

# Variable mit pickle laden
with open(file_path, 'rb') as file:
    standardoutputpfad = pickle.load(file)
    

def outlier_Sn_blk(file, append=True): #False):
    global fullname
    #global outfile
    global outfile_results_Sn
    global outfile_corrected_raw_data_Sn
    global outfile_plt_Sn
    global Baxter_Sn
    fullname = os.path.join(outfile_results_Sn, 'Sn_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    pd.options.display.float_format = '{:.8}'.format
    
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=30, index_col=False, dtype=float)#  names=['Time', '116Sn', '117Sn', '118Sn', '119Sn', '120Sn', '121Sb', '122Sn', '123Sb', '124Sn'], skiprows=6, nrows=30, index_col=False, dtype=float)
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})

    data = pd.concat([data, data[cols].div(data['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    data = pd.concat([data, data[cols].div(data['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    data['117Sn/119Sn'] = data['117Sn'] / data['119Sn']
    data['123Sb/121Sb'] = data['123Sb'] / data['121Sb']
    data.drop(['120Sn/120Sn', '116Sn/116Sn', '117Sn/116Sn'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    datao[cols1] = data[cols1]
    del cols1[0:10]
    
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)
    
    
    datao.to_csv(outfile_corrected_raw_data_Sn + basename + '.csv', sep='\t', header = True, index_label='Index_name')
    mean_data = datao.mean()
    mean_filtered_transposed = pd.DataFrame(mean_data).T
    
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file)

    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')
     

def outlier_Sn_std(file, append=True):

    global fullname
    global outfile_results_Sn
    global outfile_corrected_raw_data_Sn
    global outfile_plt_Sn
    global Baxter_Sn
    fullname = os.path.join(outfile_results_Sn, 'Sn_export.csv')
    
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
  
    pd.options.display.float_format = '{:.8f}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=90, index_col=False, dtype=float)
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    datao[cols] = data[cols]
    
    datao = pd.concat([datao, datao[cols].div(datao['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    datao = pd.concat([datao, datao[cols].div(datao['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    datao['117Sn/119Sn'] = datao['117Sn'] / datao['119Sn']
    datao['123Sb/121Sb'] = datao['123Sb'] / datao['121Sb']
    datao.drop(['120Sn/120Sn', '116Sn/116Sn', '117Sn/116Sn'], axis = 1, inplace = True)
    
    cols1 = list(datao.drop(columns='Time').columns)
    del cols1[0:9]
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 1.5)

       
    TrueStdSb = 0.747946
    fSbstd = (np.log(TrueStdSb / ((datao['123Sb/121Sb']))) / (np.log(122.90421 / 120.90381)))
    datao['116Sn/120Sn_corrected'] = datao['116Sn/120Sn'] * ((115.901743 / 119.902202) ** fSbstd)
    datao['117Sn/120Sn_corrected'] = datao['117Sn/120Sn'] * ((116.902954 / 119.902202) ** fSbstd)
    datao['118Sn/120Sn_corrected'] = datao['118Sn/120Sn'] * ((117.901607 / 119.902202) ** fSbstd)
    datao['119Sn/120Sn_corrected'] = datao['119Sn/120Sn'] * ((118.903311 / 119.902202) ** fSbstd)
    datao['122Sn/120Sn_corrected'] = datao['122Sn/120Sn'] * ((121.90344 / 119.902202) ** fSbstd)
    datao['124Sn/120Sn_corrected'] = datao['124Sn/120Sn'] * ((123.905277 / 119.902202) ** fSbstd)
    datao['124Sn/116Sn_corrected'] = datao['124Sn/116Sn'] * ((123.905277 / 115.901743) ** fSbstd)
    datao['122Sn/116Sn_corrected'] = datao['122Sn/116Sn'] * ((121.90344 / 115.901743) ** fSbstd)
    datao['117Sn/119Sn_corrected'] = datao['117Sn/119Sn'] * ((116.902954 / 118.90331117) ** fSbstd)
        
    datao.to_csv(outfile_corrected_raw_data_Sn + basename + '.csv', sep='\t', header = True, index_label='Index_name')
    
    mean_data = datao.mean()
    mean_filtered_transposed = pd.DataFrame(mean_data).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file)

    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')
    
    
    databaxter_1 = clean.copy()
    databaxter_1['116Sn/120Sn_1SE'] = datao['116Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['116Sn/120Sn']))
    databaxter_1['117Sn/120Sn_1SE'] = datao['117Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['117Sn/120Sn']))
    databaxter_1['118Sn/120Sn_1SE'] = datao['118Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['118Sn/120Sn']))
    databaxter_1['119Sn/120Sn_1SE'] = datao['119Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['119Sn/120Sn']))
    databaxter_1['122Sn/120Sn_1SE'] = datao['122Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['122Sn/120Sn']))
    databaxter_1['124Sn/120Sn_1SE'] = datao['124Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['124Sn/120Sn']))
    databaxter_1['124Sn/116Sn_1SE'] = datao['124Sn/116Sn'].std(ddof = 1) / np.sqrt(np.size(datao['124Sn/116Sn']))
    databaxter_1['122Sn/116Sn_1SE'] = datao['122Sn/116Sn'].std(ddof = 1) / np.sqrt(np.size(datao['122Sn/116Sn']))
    databaxter_1['117Sn/119Sn_1SE'] = datao['117Sn/119Sn'].std(ddof = 1) / np.sqrt(np.size(datao['117Sn/119Sn']))
    databaxter_1['123Sb/121Sb_1SE'] = datao['123Sb/121Sb'].std(ddof = 1) / np.sqrt(np.size(datao['123Sb/121Sb']))
    
    databaxter_1.to_csv(Baxter_Sn + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    
    return f"datao  {datao}"

    
def outlier_Sn_sample(file, append=True): 

    global fullname
    global outfile
    global outfile_results_Sn
    global outfile_corrected_raw_data_Sn
    global outfile_plt_Sn
    global Baxter_Sn
    fullname = os.path.join(outfile_results_Sn, 'Sn_export.csv')
    
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    corrected_fullname = os.path.join(outfile_results_Sn, 'Sn_corrected.csv')
    pd.options.display.float_format = '{:.8f}'.format

    length = len(pd.read_csv(file))
    split = int(length / 3) - 1
    double_split = split * 2
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=split, index_col=False, dtype=float)
    data2 = pd.read_csv(file, sep='\t', names=col_names, skiprows=6+split, nrows=split, index_col=False, dtype=float)
    data3 = pd.read_csv(file, sep='\t', names=col_names, skiprows=6+double_split, index_col=False, dtype=float)
    data_all = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=split, index_col=False, dtype=float)
    
    cols = list(data.drop(columns='Time').columns)
    cols2 = list(data2.drop(columns='Time').columns)
    cols3 = list(data3.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data2o = pd.DataFrame({'Time':data2['Time']})
    data3o = pd.DataFrame({'Time':data3['Time']})

    data = pd.concat([data, data[cols].div(data['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    data = pd.concat([data, data[cols].div(data['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    data['117Sn/119Sn'] = data['117Sn'] / data['119Sn']
    data['123Sb/121Sb'] = data['123Sb'] / data['121Sb']
    data.drop(['120Sn/120Sn', '116Sn/116Sn', '117Sn/116Sn'], axis = 1, inplace = True)
    
    data2 = pd.concat([data2, data2[cols2].div(data2['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    data2 = pd.concat([data2, data2[cols2].div(data2['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    data2['117Sn/119Sn'] = data2['117Sn'] / data2['119Sn']
    data2['123Sb/121Sb'] = data2['123Sb'] / data2['121Sb']
    data2.drop(['120Sn/120Sn', '116Sn/116Sn', '117Sn/116Sn'], axis = 1, inplace = True)
    
    data3 = pd.concat([data3, data3[cols3].div(data3['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    data3 = pd.concat([data3, data3[cols3].div(data3['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    data3['117Sn/119Sn'] = data3['117Sn'] / data3['119Sn']
    data3['123Sb/121Sb'] = data3['123Sb'] / data3['121Sb']
    data3.drop(['120Sn/120Sn', '116Sn/116Sn', '117Sn/116Sn'], axis = 1, inplace = True)
    
    colsa1 = list(data.columns)
    colsa2 = list(data2.columns)
    colsa3 = list(data3.columns)

    datao[colsa1] = data[colsa1]
    data2o[colsa2] = data2[colsa2]
    data3o[colsa3] = data3[colsa3]
    
    del colsa1[0:10]
    del colsa2[0:10]
    del colsa3[0:10]
    
    #Outlier correction
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 1.5)
    data2o[colsa2] = data2o[colsa2].where(np.abs(stats.zscore(data2o[colsa2])) < 1.5)
    data3o[colsa3] = data3o[colsa3].where(np.abs(stats.zscore(data3o[colsa3])) < 1.5)
    

    # Mass-bias correction
    TrueStdSb = 0.747946    
    fSbstd = (np.log(TrueStdSb / ((datao['123Sb/121Sb']))) / (np.log(122.90421 / 120.90381)))
    datao['116Sn/120Sn_corrected'] = datao['116Sn/120Sn'] * ((115.901743 / 119.902202) ** fSbstd)
    datao['117Sn/120Sn_corrected'] = datao['117Sn/120Sn'] * ((116.902954 / 119.902202) ** fSbstd)
    datao['118Sn/120Sn_corrected'] = datao['118Sn/120Sn'] * ((117.901607 / 119.902202) ** fSbstd)
    datao['119Sn/120Sn_corrected'] = datao['119Sn/120Sn'] * ((118.903311 / 119.902202) ** fSbstd)
    datao['122Sn/120Sn_corrected'] = datao['122Sn/120Sn'] * ((121.90344 / 119.902202) ** fSbstd)
    datao['124Sn/120Sn_corrected'] = datao['124Sn/120Sn'] * ((123.905277 / 119.902202) ** fSbstd)
    datao['124Sn/116Sn_corrected'] = datao['124Sn/116Sn'] * ((123.905277 / 115.901743) ** fSbstd)
    datao['122Sn/116Sn_corrected'] = datao['122Sn/116Sn'] * ((121.90344 / 115.901743) ** fSbstd)
    datao['117Sn/119Sn_corrected'] = datao['117Sn/119Sn'] * ((116.902954 / 118.90331117) ** fSbstd)
    
    
    fSbstd2 = (np.log(TrueStdSb / ((data2o['123Sb/121Sb']))) / (np.log(122.90421 / 120.90381)))
    data2o['116Sn/120Sn_corrected'] = data2o['116Sn/120Sn'] * ((115.901743 / 119.902202) ** fSbstd2)
    data2o['117Sn/120Sn_corrected'] = data2o['117Sn/120Sn'] * ((116.902954 / 119.902202) ** fSbstd2)
    data2o['118Sn/120Sn_corrected'] = data2o['118Sn/120Sn'] * ((117.901607 / 119.902202) ** fSbstd2)
    data2o['119Sn/120Sn_corrected'] = data2o['119Sn/120Sn'] * ((118.903311 / 119.902202) ** fSbstd2)
    data2o['122Sn/120Sn_corrected'] = data2o['122Sn/120Sn'] * ((121.90344 / 119.902202) ** fSbstd2)
    data2o['124Sn/120Sn_corrected'] = data2o['124Sn/120Sn'] * ((123.905277 / 119.902202) ** fSbstd2)
    data2o['124Sn/116Sn_corrected'] = data2o['124Sn/116Sn'] * ((123.905277 / 115.901743) ** fSbstd2)
    data2o['122Sn/116Sn_corrected'] = data2o['122Sn/116Sn'] * ((121.90344 / 115.901743) ** fSbstd2)
    data2o['117Sn/119Sn_corrected'] = data2o['117Sn/119Sn'] * ((116.902954 / 118.90331117) ** fSbstd2)
    
    
    fSbstd3 = (np.log(TrueStdSb / ((data3o['123Sb/121Sb']))) / (np.log(122.90421 / 120.90381)))
    data3o['116Sn/120Sn_corrected'] = data3o['116Sn/120Sn'] * ((115.901743 / 119.902202) ** fSbstd3)
    data3o['117Sn/120Sn_corrected'] = data3o['117Sn/120Sn'] * ((116.902954 / 119.902202) ** fSbstd3)
    data3o['118Sn/120Sn_corrected'] = data3o['118Sn/120Sn'] * ((117.901607 / 119.902202) ** fSbstd3)
    data3o['119Sn/120Sn_corrected'] = data3o['119Sn/120Sn'] * ((118.903311 / 119.902202) ** fSbstd3)
    data3o['122Sn/120Sn_corrected'] = data3o['122Sn/120Sn'] * ((121.90344 / 119.902202) ** fSbstd3)
    data3o['124Sn/120Sn_corrected'] = data3o['124Sn/120Sn'] * ((123.905277 / 119.902202) ** fSbstd3)
    data3o['124Sn/116Sn_corrected'] = data3o['124Sn/116Sn'] * ((123.905277 / 115.901743) ** fSbstd3)
    data3o['122Sn/116Sn_corrected'] = data3o['122Sn/116Sn'] * ((121.90344 / 115.901743) ** fSbstd3)
    data3o['117Sn/119Sn_corrected'] = data3o['117Sn/119Sn'] * ((116.902954 / 118.90331117) ** fSbstd3)
    
    

    #''''''
    datao.to_csv(outfile_corrected_raw_data_Sn + basename + '_1.csv', sep='\t', header = True, index_label='Index_name')
    data2o.to_csv(outfile_corrected_raw_data_Sn + basename + '_2.csv', sep='\t', header = True, index_label='Index_name')
    data3o.to_csv(outfile_corrected_raw_data_Sn + basename + '_3.csv', sep='\t', header = True, index_label='Index_name')
    
    mean_data = datao.mean()
    mean_filtered_transposed = pd.DataFrame(mean_data).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file + '_1')
    clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    
    
    mean_data2 = data2o.mean()
    
    mean_filtered_transposed2 = pd.DataFrame(mean_data2).T
    mean_filtered_transposed2['Time'] = pd.to_datetime(mean_filtered_transposed2["Time"], unit='s')
    clean2 = mean_filtered_transposed2.drop(mean_filtered_transposed2.columns[[0]], axis=1) 
    clean2.insert(0, 'Inputfile', file + '_2')
    clean2.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')

    mean_data3 = data3o.mean()
    mean_filtered_transposed3 = pd.DataFrame(mean_data3).T
    
    mean_filtered_transposed3['Time'] = pd.to_datetime(mean_filtered_transposed3["Time"], unit='s')

    clean3 = mean_filtered_transposed3.drop(mean_filtered_transposed3.columns[[0]], axis=1) 
    clean3.insert(0, 'Inputfile', file + '_3')
    clean3.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
     
    databaxter_1 = clean.copy()
    databaxter_2 = clean2.copy()
    databaxter_3 = clean3.copy()
    databaxter_1['116Sn/120Sn_1SE'] = datao['116Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['116Sn/120Sn']))
    databaxter_1['117Sn/120Sn_1SE'] = datao['117Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['117Sn/120Sn']))
    databaxter_1['118Sn/120Sn_1SE'] = datao['118Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['118Sn/120Sn']))
    databaxter_1['119Sn/120Sn_1SE'] = datao['119Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['119Sn/120Sn']))
    databaxter_1['122Sn/120Sn_1SE'] = datao['122Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['122Sn/120Sn']))
    databaxter_1['124Sn/120Sn_1SE'] = datao['124Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(datao['124Sn/120Sn']))
    databaxter_1['124Sn/116Sn_1SE'] = datao['124Sn/116Sn'].std(ddof = 1) / np.sqrt(np.size(datao['124Sn/116Sn']))
    databaxter_1['122Sn/116Sn_1SE'] = datao['122Sn/116Sn'].std(ddof = 1) / np.sqrt(np.size(datao['122Sn/116Sn']))
    databaxter_1['117Sn/119Sn_1SE'] = datao['117Sn/119Sn'].std(ddof = 1) / np.sqrt(np.size(datao['117Sn/119Sn']))
    databaxter_1['123Sb/121Sb_1SE'] = datao['123Sb/121Sb'].std(ddof = 1) / np.sqrt(np.size(datao['123Sb/121Sb']))
    
    baxter_header = databaxter_1.copy()
    baxter_header.to_csv(Baxter_Sn + 'Baxter_header.csv', sep='\t')
    
    databaxter_2['116Sn/120Sn_1SE'] = data2o['116Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data2o['116Sn/120Sn']))
    databaxter_2['117Sn/120Sn_1SE'] = data2o['117Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data2o['117Sn/120Sn']))
    databaxter_2['118Sn/120Sn_1SE'] = data2o['118Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data2o['118Sn/120Sn']))
    databaxter_2['119Sn/120Sn_1SE'] = data2o['119Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data2o['119Sn/120Sn']))
    databaxter_2['122Sn/120Sn_1SE'] = data2o['122Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data2o['122Sn/120Sn']))
    databaxter_2['124Sn/120Sn_1SE'] = data2o['124Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data2o['124Sn/120Sn']))
    databaxter_2['124Sn/116Sn_1SE'] = data2o['124Sn/116Sn'].std(ddof = 1) / np.sqrt(np.size(data2o['124Sn/116Sn']))
    databaxter_2['122Sn/116Sn_1SE'] = data2o['122Sn/116Sn'].std(ddof = 1) / np.sqrt(np.size(data2o['122Sn/116Sn']))
    databaxter_2['117Sn/119Sn_1SE'] = data2o['117Sn/119Sn'].std(ddof = 1) / np.sqrt(np.size(data2o['117Sn/119Sn']))
    databaxter_2['123Sb/121Sb_1SE'] = data2o['123Sb/121Sb'].std(ddof = 1) / np.sqrt(np.size(data2o['123Sb/121Sb']))
    
    databaxter_3['116Sn/120Sn_1SE'] = data3o['116Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data3o['116Sn/120Sn']))
    databaxter_3['117Sn/120Sn_1SE'] = data3o['117Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data3o['117Sn/120Sn']))
    databaxter_3['118Sn/120Sn_1SE'] = data3o['118Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data3o['118Sn/120Sn']))
    databaxter_3['119Sn/120Sn_1SE'] = data3o['119Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data3o['119Sn/120Sn']))
    databaxter_3['122Sn/120Sn_1SE'] = data3o['122Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data3o['122Sn/120Sn']))
    databaxter_3['124Sn/120Sn_1SE'] = data3o['124Sn/120Sn'].std(ddof = 1) / np.sqrt(np.size(data3o['124Sn/120Sn']))
    databaxter_3['124Sn/116Sn_1SE'] = data3o['124Sn/116Sn'].std(ddof = 1) / np.sqrt(np.size(data3o['124Sn/116Sn']))
    databaxter_3['122Sn/116Sn_1SE'] = data3o['122Sn/116Sn'].std(ddof = 1) / np.sqrt(np.size(data3o['122Sn/116Sn']))
    databaxter_3['117Sn/119Sn_1SE'] = data3o['117Sn/119Sn'].std(ddof = 1) / np.sqrt(np.size(data3o['117Sn/119Sn']))
    databaxter_3['123Sb/121Sb_1SE'] = data3o['123Sb/121Sb'].std(ddof = 1) / np.sqrt(np.size(data3o['123Sb/121Sb']))
    
    databaxter_1.to_csv(Baxter_Sn + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    databaxter_2.to_csv(Baxter_Sn + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    databaxter_3.to_csv(Baxter_Sn + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    
    return baxter_header