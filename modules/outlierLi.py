# -*- coding: utf-8 -*-
"""
This module containts the Outlier correction for Li

@author: Andreas Wittke and Ronny Friedrich
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
#from sklearn.linear_model import LinearRegression
from csv import DictReader
import seaborn as sb
sb.set()
#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score
plt.ioff()
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 16})

# import seaborn as sns 
from modules.config import WorkspaceVariableInput, outfile_plt_Li, outfile_corrected_raw_data_Li, outfile_results_Li, Baxter_Li #outfile
#outfile = (conf.outfile) 
infile = WorkspaceVariableInput
#outfile_plt = conf.outfile_plt
#outfile_corrected_raw_data = conf.outfile_corrected_raw_data 
#outfile_results = conf.outfile_results



def outlier_Li_blk(file, append=True): #False):
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
    #data.drop(['60Ni/60Ni', '61Ni/60Ni', '6Li/60Ni', '64Ni/60Ni', '7Li/60Ni', '66Zn/60Ni', '60Ni/6Li', '61Ni/6Li', '62Ni/6Li', '6Li/6Li', '64Ni/6Li', '66Zn/6Li'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    
    datao[cols1] = data[cols1]#.where(np.abs(stats.zscore(data[cols])) < 2)
    del cols1[0:3] 
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
 
    #data = pd.read_csv(file, sep='\t', names=['Time', '60Ni', '61Ni', '62Ni', '6Li', '64Ni', '7Li', '66Zn'], skiprows=6, nrows=80, index_col=False, dtype=float)
    #cols = list(data.drop(columns='Time').columns)
    #datao = pd.DataFrame({'Time':data['Time']})
    #datao[cols] = data[cols].where(np.abs(stats.zscore(data[cols])) < 2)
    
    #datao['62/60Ni'] = datao['62Ni'] / datao['60Ni']
    #datao['65/6Li'] = datao['7Li'] / datao['6Li']
    
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
     
    data.drop(['6.53'], axis = 1, inplace = True)
    
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
#    data = pd.concat([data, data[cols].div(data['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data = pd.concat([data, data[cols].div(data['6Li'], axis=0).add_suffix('/6Li')], axis=1)
    data.drop(['6Li/6Li'], axis = 1, inplace = True)
   # data.drop(['60Ni/60Ni', '61Ni/60Ni', '6Li/60Ni', '64Ni/60Ni', '7Li/60Ni', '66Zn/60Ni', '60Ni/6Li', '61Ni/6Li', '62Ni/6Li', '6Li/6Li', '64Ni/6Li', '66Zn/6Li'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    
    datao[cols1] = data[cols1]#.where(np.abs(stats.zscore(data[cols])) < 2)
    del cols1[0:3] 
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)
    
    
    
#    TrueStdNi = 0.13860
#    fnistd = (np.log(TrueStdNi / ((datao['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
#    datao['7Li/6Li_corrected'] = datao['7Li/6Li'] * ((64.927790 / 62.929598) ** fnistd)

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
#    databaxter_1 = databaxter_1.drop('7Li/6Li_corrected', axis = 1)
    databaxter_1['7Li/6Li_1SE'] = datao['7Li/6Li'].std(ddof = 1) / np.sqrt(np.size(datao['7Li/6Li']))
#    databaxter_1['62Ni/60Ni_1SE'] = datao['62Ni/60Ni'].std(ddof = 1) / np.sqrt(np.size(datao['62Ni/60Ni']))
    
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

    length = len(pd.read_csv(file))
    split = int(length / 2) - 1
    #double_split = split * 2
    
  
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=split, index_col=False, dtype=float)
    data2 = pd.read_csv(file, sep='\t', names=col_names, skiprows=6+split, nrows=split, index_col=False, dtype=float)
    #data3 = pd.read_csv(file, sep='\t', names=col_names, skiprows=6+double_split, index_col=False, dtype=float)

    data_all = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)

    data.drop(['6.53'], axis = 1, inplace = True)
    data2.drop(['6.53'], axis = 1, inplace = True)
    #data3.drop(['6.53'], axis = 1, inplace = True)
    data_all.drop(['6.53'], axis = 1, inplace = True)

    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    datao[cols] = data[cols]
#    data = pd.concat([data, data[cols].div(data['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data = pd.concat([data, data[cols].div(data['6Li'], axis=0).add_suffix('/6Li')], axis=1)
    data.drop(['6Li/6Li'], axis = 1, inplace = True)
    
#    data.drop(['60Ni/60Ni', '61Ni/60Ni', '6Li/60Ni', '64Ni/60Ni', '7Li/60Ni', '66Zn/60Ni', '60Ni/6Li', '61Ni/6Li', '62Ni/6Li', '6Li/6Li', '64Ni/6Li', '66Zn/6Li'], axis = 1, inplace = True)
    colsa1 = list(data.columns)

    cols2 = list(data2.drop(columns='Time').columns)
    data2o = pd.DataFrame({'Time':data2['Time']})
    data2o[cols] = data2[cols]
#    data2 = pd.concat([data2, data2[cols2].div(data2['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data2 = pd.concat([data2, data2[cols2].div(data2['6Li'], axis=0).add_suffix('/6Li')], axis=1)
    data2.drop(['6Li/6Li'], axis = 1, inplace = True)
#    data2.drop(['60Ni/60Ni', '61Ni/60Ni', '6Li/60Ni', '64Ni/60Ni', '7Li/60Ni', '66Zn/60Ni', '60Ni/6Li', '61Ni/6Li', '62Ni/6Li', '6Li/6Li', '64Ni/6Li', '66Zn/6Li'], axis = 1, inplace = True)
    colsa2 = list(data2.columns)

#    cols3 = list(data3.drop(columns='Time').columns)
#    data3o = pd.DataFrame({'Time':data3['Time']})
#    data3o[cols] = data3[cols]
#    data3 = pd.concat([data3, data3[cols3].div(data3['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
#    data3 = pd.concat([data3, data3[cols3].div(data3['6Li'], axis=0).add_suffix('/6Li')], axis=1)
#    data3.drop(['6Li/6Li'], axis = 1, inplace = True)
#    data3.drop(['60Ni/60Ni', '61Ni/60Ni', '6Li/60Ni', '64Ni/60Ni', '7Li/60Ni', '66Zn/60Ni', '60Ni/6Li', '61Ni/6Li', '62Ni/6Li', '6Li/6Li', '64Ni/6Li', '66Zn/6Li'], axis = 1, inplace = True)
#    colsa3 = list(data3.columns)
    
   #TrueStdNi = 0.13860
    

    
    datao[colsa1] = data[colsa1]#.where(np.abs(stats.zscore(data[cols])) < 2)
    del colsa1[0:3] 
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 2)
    
    data2o[colsa2] = data2[colsa2]#.where(np.abs(stats.zscore(data[cols])) < 2)
    del colsa2[0:3] 
    data2o[colsa2] = data2o[colsa2].where(np.abs(stats.zscore(data2o[colsa2])) < 2)
    
    # data3o[colsa3] = data3[colsa3]#.where(np.abs(stats.zscore(data[cols])) < 2)
    # del colsa3[0:3] 
    # data3o[colsa3] = data3o[colsa3].where(np.abs(stats.zscore(data3o[colsa3])) < 2)
    
    
#    fnistd = (np.log(TrueStdNi / ((datao['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
#    datao['7Li/6Li_corrected'] = datao['7Li/6Li'] * ((64.927790 / 62.929598) ** fnistd)
    
#    fnistd2 = (np.log(TrueStdNi / ((data2o['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
#    data2o['7Li/6Li_corrected'] = data2o['7Li/6Li'] * ((64.927790 / 62.929598) ** fnistd2)

#    fnistd3 = (np.log(TrueStdNi / ((data3o['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
#    data3o['7Li/6Li_corrected'] = data3o['7Li/6Li'] * ((64.927790 / 62.929598) ** fnistd3)
    #''''''
    datao.to_csv(outfile_corrected_raw_data_Li + basename + '_1.csv', sep='\t', header = True, index_label='Index_name')
    data2o.to_csv(outfile_corrected_raw_data_Li + basename + '_2.csv', sep='\t', header = True, index_label='Index_name')
    # data3o.to_csv(outfile_corrected_raw_data_Li + basename + '_3.csv', sep='\t', header = True, index_label='Index_name')
    
    mean_filtered_transposed = pd.DataFrame(data=np.mean(datao)).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file + '_1')
    clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    
    mean_filtered_transposed2 = pd.DataFrame(data=np.mean(data2o)).T
    mean_filtered_transposed2['Time'] = pd.to_datetime(mean_filtered_transposed2["Time"], unit='s')
    clean2 = mean_filtered_transposed2.drop(mean_filtered_transposed2.columns[[0]], axis=1) 
    clean2.insert(0, 'Inputfile', file + '_2')
    clean2.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')

    # mean_filtered_transposed3 = pd.DataFrame(data=np.mean(data3o)).T
    # mean_filtered_transposed3['Time'] = pd.to_datetime(mean_filtered_transposed3["Time"], unit='s')
    mean_all = pd.concat([mean_filtered_transposed, mean_filtered_transposed2])#, mean_filtered_transposed3])
    mean = pd.DataFrame(data=np.mean(mean_all)).T
    # clean3 = mean_filtered_transposed3.drop(mean_filtered_transposed3.columns[[0]], axis=1) 
    # clean3.insert(0, 'Inputfile', file + '_3')
    # clean3.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
     
    #ln 
#    datalogNi = np.log(datao['62Ni'] / datao['60Ni'])
    datalogLi = np.log(datao['7Li'] / datao['6Li'])
#    datalogNi2 = np.log(data2o['62Ni'] / data2o['60Ni'])
    datalogLi2 = np.log(data2o['7Li'] / data2o['6Li'])
#    datalogNi3 = np.log(data3o['62Ni'] / data3o['60Ni'])
#    datalogLi3 = np.log(data3o['7Li'] / data3o['6Li'])
    

    databaxter_1 = clean.copy()
    databaxter_2 = clean2.copy()
 #   databaxter_3 = clean3.copy()
#    databaxter_1 = databaxter_1.drop('7Li/6Li_corrected', axis = 1)
#    databaxter_2 = databaxter_2.drop('7Li/6Li_corrected', axis = 1)
#    databaxter_3 = databaxter_3.drop('7Li/6Li_corrected', axis = 1)
    
    databaxter_1['7Li/6Li_1SE'] = datao['7Li/6Li'].std(ddof = 1) / np.sqrt(np.size(datao['7Li/6Li']))
#    databaxter_1['62Ni/60Ni_1SE'] = datao['62Ni/60Ni'].std(ddof = 1) / np.sqrt(np.size(datao['62Ni/60Ni']))

    
    baxter_header = databaxter_1.copy()
    #baxter_header = databaxter_1.columns.values.tolist()
    baxter_header.to_csv(Baxter_Li + 'Baxter_header.csv', sep='\t')#, header = baxter_header, index_label='Index_name')
    
    databaxter_2['7Li/6Li_1SE'] = data2o['7Li/6Li'].std(ddof = 1) / np.sqrt(np.size(data2o['7Li/6Li']))
#    databaxter_2['62Ni/60Ni_1SE'] = data2o['62Ni/60Ni'].std(ddof = 1) / np.sqrt(np.size(data2o['62Ni/60Ni']))
    
#    databaxter_3['7Li/6Li_1SE'] = data3o['7Li/6Li'].std(ddof = 1) / np.sqrt(np.size(data3o['7Li/6Li']))
#    databaxter_3['62Ni/60Ni_1SE'] = data3o['62Ni/60Ni'].std(ddof = 1) / np.sqrt(np.size(data3o['62Ni/60Ni']))
    
    databaxter_1.to_csv(Baxter_Li + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    databaxter_2.to_csv(Baxter_Li + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
#    databaxter_3.to_csv(Baxter_Li + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    
    return baxter_header
