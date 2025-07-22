# -*- coding: utf-8 -*-
"""
This module containts the Outlier correction for Cu

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
from csv import DictReader
import seaborn as sb
import pickle

sb.set_theme()
plt.ioff()
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 16})

# import seaborn as sns 
from modules.config import WorkspaceVariableInput, outfile_plt_Cu, outfile_corrected_raw_data_Cu, outfile_results_Cu, Baxter_Cu, exportCu #outfile
file_path = os.path.join("modules", "variable_data.pkl")

# Variable mit pickle laden
with open(file_path, 'rb') as file:
    standardoutputpfad = pickle.load(file)
    
file_path2 = os.path.join("modules", "variable_data_input.pkl")
# Variable mit pickle laden
with open(file_path2, 'rb') as file2:
    standardinputpfad = pickle.load(file2)

infile = standardinputpfad

def outlier_Cu_blk(file, append=True): #False):
    global fullname
    global outfile
    global outfile_results_Cu
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu
    global Baxter_Cu
    global infile
    global exportCu
   
    fullname = os.path.join(outfile_results_Cu, 'Cu_export.csv')

    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8f}'.format

    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
    #print('DATA', data)    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data = pd.concat([data, data[cols].div(data['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data = pd.concat([data, data[cols].div(data['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    
    datao[cols1] = data[cols1]#.where(np.abs(stats.zscore(data[cols])) < 2)
    del cols1[0:8] 
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2) 

    datao.to_csv(exportCu + basename + '.csv', sep='\t', header = True, index_label='Index_name')

    cols_datao = list(datao.columns)
    datao.columns = cols_datao

    mean_data = datao.mean()
    # print(mean_data)
    mean_filtered_transposed = pd.DataFrame(mean_data).T
    
    print("Fullname", fullname)
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file)
    # print(clean)
    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')
     

def outlier_Cu_std(file, append=True): #False):
    global outfile
    global outfile_results_Cu
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu
    global Baxter_Cu
    global infile

    fullname = os.path.join(outfile_results_Cu, 'Cu_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
  
    pd.options.display.float_format = '{:.8f}'.format
    
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
        
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data = pd.concat([data, data[cols].div(data['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data = pd.concat([data, data[cols].div(data['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    
    datao[cols1] = data[cols1]
    del cols1[0:8] 
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)
    
    
    
    TrueStdNi = 0.13860
    fnistd = (np.log(TrueStdNi / ((datao['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
    datao['65Cu/63Cu_corrected'] = datao['65Cu/63Cu'] * ((64.927790 / 62.929598) ** fnistd)

    datao.to_csv(outfile_corrected_raw_data_Cu + basename + '.csv', sep='\t', header = True, index_label='Index_name')
    
    cols_datao = list(datao.columns)
    datao.columns = cols_datao

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
    databaxter_1 = databaxter_1.drop('65Cu/63Cu_corrected', axis = 1)
    databaxter_1['65Cu/63Cu_1SE'] = datao['65Cu/63Cu'].std(ddof = 1) / np.sqrt(np.size(datao['65Cu/63Cu']))
    databaxter_1['62Ni/60Ni_1SE'] = datao['62Ni/60Ni'].std(ddof = 1) / np.sqrt(np.size(datao['62Ni/60Ni']))
    
    baxter_header = databaxter_1.copy()
    baxter_header.to_csv(Baxter_Cu + 'Baxter_header.csv', sep='\t')
    databaxter_1.to_csv(Baxter_Cu + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    
def outlier_Cu_sample(file, append=True):
    global outfile
    global outfile_results_Cu 
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu
    global Baxter_Cu
    global infile
    
    fullname = os.path.join(outfile_results_Cu, 'Cu_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
 
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

    data_all = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)

    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    datao[cols] = data[cols]
    data = pd.concat([data, data[cols].div(data['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data = pd.concat([data, data[cols].div(data['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    colsa1 = list(data.columns)

    cols2 = list(data2.drop(columns='Time').columns)
    data2o = pd.DataFrame({'Time':data2['Time']})
    data2o[cols] = data2[cols]
    data2 = pd.concat([data2, data2[cols2].div(data2['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data2 = pd.concat([data2, data2[cols2].div(data2['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data2.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    colsa2 = list(data2.columns)

    cols3 = list(data3.drop(columns='Time').columns)
    data3o = pd.DataFrame({'Time':data3['Time']})
    data3o[cols] = data3[cols]
    data3 = pd.concat([data3, data3[cols3].div(data3['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data3 = pd.concat([data3, data3[cols3].div(data3['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data3.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    colsa3 = list(data3.columns)
    
    TrueStdNi = 0.13860
    

    
    datao[colsa1] = data[colsa1]
    del colsa1[0:8] 
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 1.5)
    
    data2o[colsa2] = data2[colsa2]
    del colsa2[0:8] 
    data2o[colsa2] = data2o[colsa2].where(np.abs(stats.zscore(data2o[colsa2])) < 2)
    
    data3o[colsa3] = data3[colsa3]
    del colsa3[0:8] 
    data3o[colsa3] = data3o[colsa3].where(np.abs(stats.zscore(data3o[colsa3])) < 2)
    
    
    fnistd = (np.log(TrueStdNi / ((datao['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
    datao['65Cu/63Cu_corrected'] = datao['65Cu/63Cu'] * ((64.927790 / 62.929598) ** fnistd)
    
    fnistd2 = (np.log(TrueStdNi / ((data2o['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
    data2o['65Cu/63Cu_corrected'] = data2o['65Cu/63Cu'] * ((64.927790 / 62.929598) ** fnistd2)

    fnistd3 = (np.log(TrueStdNi / ((data3o['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
    data3o['65Cu/63Cu_corrected'] = data3o['65Cu/63Cu'] * ((64.927790 / 62.929598) ** fnistd3)
    #''''''
    datao.to_csv(outfile_corrected_raw_data_Cu + basename + '_1.csv', sep='\t', header = True, index_label='Index_name')
    data2o.to_csv(outfile_corrected_raw_data_Cu + basename + '_2.csv', sep='\t', header = True, index_label='Index_name')
    data3o.to_csv(outfile_corrected_raw_data_Cu + basename + '_3.csv', sep='\t', header = True, index_label='Index_name')
    
    
    cols_datao = list(datao.columns)
    datao.columns = cols_datao
    mean_data = datao.mean()
    mean_filtered_transposed = pd.DataFrame(mean_data).T
    
    cols_datao2 = list(data2o.columns)
    data2o.columns = cols_datao2
    mean_data2 = data2o.mean()
    mean_filtered_transposed2 = pd.DataFrame(mean_data2).T
    
    cols_datao3 = list(data3o.columns)
    data3o.columns = cols_datao3
    mean_data3 = data3o.mean()
    mean_filtered_transposed3 = pd.DataFrame(mean_data3).T
        
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file + '_1')
    clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    
    mean_filtered_transposed2['Time'] = pd.to_datetime(mean_filtered_transposed2["Time"], unit='s')
    clean2 = mean_filtered_transposed2.drop(mean_filtered_transposed2.columns[[0]], axis=1) 
    clean2.insert(0, 'Inputfile', file + '_2')
    clean2.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')

    mean_filtered_transposed3['Time'] = pd.to_datetime(mean_filtered_transposed3["Time"], unit='s')
    
    clean3 = mean_filtered_transposed3.drop(mean_filtered_transposed3.columns[[0]], axis=1) 
    clean3.insert(0, 'Inputfile', file + '_3')
    clean3.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
     
    #ln 
    datalogNi = np.log(datao['62Ni'] / datao['60Ni'])
    datalogCu = np.log(datao['65Cu'] / datao['63Cu'])
    datalogNi2 = np.log(data2o['62Ni'] / data2o['60Ni'])
    datalogCu2 = np.log(data2o['65Cu'] / data2o['63Cu'])
    datalogNi3 = np.log(data3o['62Ni'] / data3o['60Ni'])
    datalogCu3 = np.log(data3o['65Cu'] / data3o['63Cu'])
    

    databaxter_1 = clean.copy()
    databaxter_2 = clean2.copy()
    databaxter_3 = clean3.copy()
    databaxter_1 = databaxter_1.drop('65Cu/63Cu_corrected', axis = 1)
    databaxter_2 = databaxter_2.drop('65Cu/63Cu_corrected', axis = 1)
    databaxter_3 = databaxter_3.drop('65Cu/63Cu_corrected', axis = 1)
    
    databaxter_1['65Cu/63Cu_1SE'] = datao['65Cu/63Cu'].std(ddof = 1) / np.sqrt(np.size(datao['65Cu/63Cu']))
    databaxter_1['62Ni/60Ni_1SE'] = datao['62Ni/60Ni'].std(ddof = 1) / np.sqrt(np.size(datao['62Ni/60Ni']))

    
    baxter_header = databaxter_1.copy()
    baxter_header.to_csv(Baxter_Cu + 'Baxter_header.csv', sep='\t')
    
    databaxter_2['65Cu/63Cu_1SE'] = data2o['65Cu/63Cu'].std(ddof = 1) / np.sqrt(np.size(data2o['65Cu/63Cu']))
    databaxter_2['62Ni/60Ni_1SE'] = data2o['62Ni/60Ni'].std(ddof = 1) / np.sqrt(np.size(data2o['62Ni/60Ni']))
    
    databaxter_3['65Cu/63Cu_1SE'] = data3o['65Cu/63Cu'].std(ddof = 1) / np.sqrt(np.size(data3o['65Cu/63Cu']))
    databaxter_3['62Ni/60Ni_1SE'] = data3o['62Ni/60Ni'].std(ddof = 1) / np.sqrt(np.size(data3o['62Ni/60Ni']))
    
    databaxter_1.to_csv(Baxter_Cu + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    databaxter_2.to_csv(Baxter_Cu + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    databaxter_3.to_csv(Baxter_Cu + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    
    return baxter_header
