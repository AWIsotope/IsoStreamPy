# -*- coding: utf-8 -*-
"""
This module containts the Outlier correction for Ag
!!! UNTESTED AND NOT IMPLEMENTED !!!

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
sb.set()
from sklearn.metrics import mean_squared_error, r2_score
plt.ioff()
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 16})

from modules.config import outfile, WorkspaceVariableInput, outfile_plt, outfile_corrected_raw_data, outfile_results 
import modules.plotAg as plotAg
infile = WorkspaceVariableInput


def outlier_Ag_blk(file, append=True): #False):
    global fullname
    global outfile
    global outfile_results
    global outfile_corrected_raw_data
    global outfile_plt
    fullname = os.path.join(outfile_results, 'Ag_export.csv')
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
    data['109Ag/107Ag'] = data['109Ag'] / data['107Ag']
    data['108Pd/105Pd'] = data['108Pd'] / data['105Pd']
    data['106Pd/105Pd'] = data['106Pd'] / data['105Pd']
    data['110Pd/105Pd'] = data['110Pd'] / data['105Pd']
    cols1 = list(data.columns)
    datao[cols1] = data[cols1]
    del cols1[0:8] 
    
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)
    
    
    datao.to_csv(outfile_corrected_raw_data + basename + '.csv', sep='\t', header = True, index_label='Index_name')

    mean_filtered_transposed = pd.DataFrame(data=np.mean(datao)).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file)

    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')
     
def outlier_Ag_std(file, append=True): #False):
    global fullname
    global outfile
    global outfile_results
    global outfile_corrected_raw_data
    global outfile_plt

    fullname = os.path.join(outfile_results, 'Ag_export.csv')
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

    
    data['109Ag/107Ag'] = data['109Ag'] / data['107Ag']
    data['108Pd/105Pd'] = data['108Pd'] / data['105Pd']
    data['106Pd/105Pd'] = data['106Pd'] / data['105Pd']
    data['110Pd/105Pd'] = data['110Pd'] / data['105Pd']
    
    cols1 = list(datao.drop(columns='Time').columns)
    del cols1[0:8]
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)

    
    
    TrueStd108 = 1.18899
    TrueStd106 = 1.222897
    TrueStd110 = 0.52664 
    fAgstd108 = (np.log(TrueStd108 / ((datao['108Pd/105Pd']))) / (np.log(107.903894 / 104.905084)))
    fAgstd106 = (np.log(TrueStd106 / ((datao['106Pd/105Pd']))) / (np.log(105.903483 / 104.905084)))
    fAgstd110 = (np.log(TrueStd110 / ((datao['110Pd/105Pd']))) / (np.log(109.905152 / 104.905084)))

    datao['109Ag/107Ag_corrected_108'] = datao['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAgstd108)
    datao['109Ag/107Ag_corrected_106'] = datao['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAgstd106)
    datao['109Ag/107Ag_corrected_110'] = datao['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAgstd110)

        
    datao.to_csv(outfile_corrected_raw_data + basename + '.csv', sep='\t', header = True, index_label='Index_name')
    
    mean_filtered_transposed = pd.DataFrame(data=np.mean(datao)).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file)

    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')

    return f"datao  {datao}"

    
def outlier_Ag_sample(file, append=True): #False):
    
    global fullname
    global outfile
    global outfile_results 
    global outfile_corrected_raw_data
    global outfile_plt

    fullname = os.path.join(outfile_results, 'Ag_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    corrected_fullname = os.path.join(outfile_results, 'Ag_corrected.csv')
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

    data['109Ag/107Ag'] = data['109Ag'] / data['107Ag']
    data['108Pd/105Pd'] = data['108Pd'] / data['105Pd']
    data['106Pd/105Pd'] = data['106Pd'] / data['105Pd']
    data['110Pd/105Pd'] = data['110Pd'] / data['105Pd']

    data2['109Ag/107Ag'] = data2['109Ag'] / data2['107Ag']
    data2['108Pd/105Pd'] = data2['108Pd'] / data2['105Pd']
    data2['106Pd/105Pd'] = data2['106Pd'] / data2['105Pd']
    data2['110Pd/105Pd'] = data2['110Pd'] / data2['105Pd']

    data3['109Ag/107Ag'] = data3['109Ag'] / data3['107Ag']
    data3['108Pd/105Pd'] = data3['108Pd'] / data3['105Pd']
    data3['106Pd/105Pd'] = data3['106Pd'] / data3['105Pd']
    data3['110Pd/105Pd'] = data3['110Pd'] / data3['105Pd']
    
    colsa1 = list(data.columns)
    colsa2 = list(data2.columns)
    colsa3 = list(data3.columns)

    datao[colsa1] = data[colsa1]
    data2o[colsa2] = data2[colsa2]
    data3o[colsa3] = data3[colsa3]
    
    del colsa1[0:8]
    del colsa2[0:8]
    del colsa3[0:8]
    
    #Outlier correction
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 2)
    data2o[colsa2] = data2o[colsa2].where(np.abs(stats.zscore(data2o[colsa2])) < 2)
    data3o[colsa3] = data3o[colsa3].where(np.abs(stats.zscore(data3o[colsa3])) < 2)
    
    
    # Mass-bias correction
    TrueStd108 = 1.18899
    TrueStd106 = 1.222897
    TrueStd110 = 0.52664 
    fAgstd108 = (np.log(TrueStd108 / ((datao['108Pd/105Pd']))) / (np.log(107.903894 / 104.905084)))
    fAgstd106 = (np.log(TrueStd106 / ((datao['106Pd/105Pd']))) / (np.log(105.903483 / 104.905084)))
    fAgstd110 = (np.log(TrueStd110 / ((datao['110Pd/105Pd']))) / (np.log(109.905152 / 104.905084)))

    datao['109Ag/107Ag_corrected_108'] = datao['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAgstd108)
    datao['109Ag/107Ag_corrected_106'] = datao['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAgstd106)
    datao['109Ag/107Ag_corrected_110'] = datao['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAgstd110)

    fAg2std108 = (np.log(TrueStd108 / ((data2o['108Pd/105Pd']))) / (np.log(107.903894 / 104.905084)))
    fAg2std106 = (np.log(TrueStd106 / ((data2o['106Pd/105Pd']))) / (np.log(105.903483 / 104.905084)))
    fAg2std110 = (np.log(TrueStd110 / ((data2o['110Pd/105Pd']))) / (np.log(109.905152 / 104.905084)))

    data2o['109Ag/107Ag_corrected_108'] = data2o['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAg2std108)
    data2o['109Ag/107Ag_corrected_106'] = data2o['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAg2std106)
    data2o['109Ag/107Ag_corrected_110'] = data2o['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAg2std110)

    fAg3std108 = (np.log(TrueStd108 / ((data3o['108Pd/105Pd']))) / (np.log(107.903894 / 104.905084)))
    fAg3std106 = (np.log(TrueStd106 / ((data3o['106Pd/105Pd']))) / (np.log(105.903483 / 104.905084)))
    fAg3std110 = (np.log(TrueStd110 / ((data3o['110Pd/105Pd']))) / (np.log(109.905152 / 104.905084)))

    data3o['109Ag/107Ag_corrected_108'] = data3o['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAg3std108)
    data3o['109Ag/107Ag_corrected_106'] = data3o['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAg3std106)
    data3o['109Ag/107Ag_corrected_110'] = data3o['109Ag/107Ag'] * ((108.904756 / 106.905093) ** fAg3std110)


    #''''''
    datao.to_csv(outfile_corrected_raw_data + basename + '_1.csv', sep='\t', header = True, index_label='Index_name')
    data2o.to_csv(outfile_corrected_raw_data + basename + '_2.csv', sep='\t', header = True, index_label='Index_name')
    data3o.to_csv(outfile_corrected_raw_data + basename + '_3.csv', sep='\t', header = True, index_label='Index_name')
    
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

    mean_filtered_transposed3 = pd.DataFrame(data=np.mean(data3o)).T
    mean_filtered_transposed3['Time'] = pd.to_datetime(mean_filtered_transposed3["Time"], unit='s')
    mean_all = pd.concat([mean_filtered_transposed, mean_filtered_transposed2, mean_filtered_transposed3])
    mean = pd.DataFrame(data=np.mean(mean_all)).T
    clean3 = mean_filtered_transposed3.drop(mean_filtered_transposed3.columns[[0]], axis=1) 
    clean3.insert(0, 'Inputfile', file + '_3')
    clean3.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
     
