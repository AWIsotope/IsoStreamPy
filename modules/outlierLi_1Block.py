# -*- coding: utf-8 -*-
"""
This module containts the Outlier correction for Li

ToDo: insert Lithium in IsoStreamPy

@author: Andreas Wittke
"""

"""CalLilates the outliers. """

#import modules.config as conf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import time
from csv import DictReader
import seaborn as sb
sb.set()

plt.ioff()
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 16})

from modules.config import WorkspaceVariableInput, outfile_plt_Li, outfile_corrected_raw_data_Li, outfile_results_Li, Baxter_Li #outfile
infile = WorkspaceVariableInput



def outlier_Li_blk(file, append=True): 
    global fullname
    global outfile
    global outfile_results_Li
    global outfile_corrected_raw_data_Li
    global outfile_plt_Li
    global Baxter_Li
    
    fullname = os.path.join(outfile_results_Li, 'Li_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8f}'.format

    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
    
    data.drop(['6.53'], axis = 1, inplace = True)

    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data = pd.concat([data, data[cols].div(data['6Li'], axis=0).add_suffix('/6Li')], axis=1)
    data.drop(['6Li/6Li'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    
    datao[cols1] = data[cols1]
    del cols1[0:8] 
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2) 
    
    
    datao.to_csv(outfile_corrected_raw_data_Li + basename + '.csv', sep='\t', header = True, index_label='Index_name')

    mean_filtered_transposed = pd.DataFrame(data=np.mean(datao)).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file)

    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')
     


    
def outlier_Li_std(file, append=True): #False):
    global outfile
    global outfile_results_Li
    global outfile_corrected_raw_data_Li
    global outfile_plt_Li
    global Baxter_Li

    fullname = os.path.join(outfile_results_Li, 'Li_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
  
    pd.options.display.float_format = '{:.8f}'.format
 
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
     
    data.drop(['6.53'], axis = 1, inplace = True)

    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data = pd.concat([data, data[cols].div(data['6Li'], axis=0).add_suffix('/6Li')], axis=1)
    data.drop(['6Li/6Li'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    
    datao[cols1] = data[cols1]
    del cols1[0:8] 
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)

    datao.to_csv(outfile_corrected_raw_data_Li + basename + '.csv', sep='\t', header = True, index_label='Index_name')
  
    mean_filtered_transposed = pd.DataFrame(data=np.mean(datao)).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file)

    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')
        

    databaxter_1 = clean.copy()
    databaxter_1['7Li/6Li_1SE'] = datao['7Li/6Li'].std(ddof = 1) / np.sqrt(np.size(datao['7Li/6Li']))
    baxter_header = databaxter_1.copy()
    baxter_header.to_csv(Baxter_Li + 'Baxter_header.csv', sep='\t')#, header = baxter_header, index_label='Index_name')
    databaxter_1.to_csv(Baxter_Li + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    
def outlier_Li_sample(file, append=True): #False):
    global outfile
    global outfile_results_Li 
    global outfile_corrected_raw_data_Li
    global outfile_plt_Li
    global Baxter_Li
    
    fullname = os.path.join(outfile_results_Li, 'Li_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
 
    pd.options.display.float_format = '{:.8f}'.format
  
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
    data_all = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)

    data.drop(['6.53'], axis = 1, inplace = True)
    data_all.drop(['6.53'], axis = 1, inplace = True)

    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    datao[cols] = data[cols]
    data = pd.concat([data, data[cols].div(data['6Li'], axis=0).add_suffix('/6Li')], axis=1)
    data.drop(['6Li/6Li'], axis = 1, inplace = True)
    colsa1 = list(data.columns)

    
    datao[colsa1] = data[colsa1]
    del colsa1[0:8] 
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 1.5)
    datao.to_csv(outfile_corrected_raw_data_Li + basename + '_1.csv', sep='\t', header = True, index_label='Index_name')

    mean_filtered_transposed = pd.DataFrame(data=np.mean(datao)).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file + '_1')
    clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')

     
    #ln 
    datalogLi = np.log(datao['7Li'] / datao['6Li'])

    databaxter_1 = clean.copy()

    databaxter_1['7Li/6Li_1SE'] = datao['7Li/6Li'].std(ddof = 1) / np.sqrt(np.size(datao['7Li/6Li']))

    
    baxter_header = databaxter_1.copy()
    baxter_header.to_csv(Baxter_Li + 'Baxter_header.csv', sep='\t')

    
    databaxter_1.to_csv(Baxter_Li + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')

    return baxter_header
