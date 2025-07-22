# -*- coding: utf-8 -*-
"""
This module containts the plotting for Ag 

!!! NOT TESTED !!!


@author: Andreas Wittke
"""

"""Plot everything. """

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
from modules.config import outfile, WorkspaceVariableInput, outfile_plt, outfile_corrected_raw_data, outfile_results 
from modules import outlierSn 
infile = WorkspaceVariableInput


def plot_Ag_blk(file): #False):
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
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=30, index_col=False, dtype=float)

    cols = list(data.drop(columns='Time').columns)
    #cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data['109Ag/107Ag'] = data['109Ag'] / data['107Ag']
    data['108Pd/105Pd'] = data['108Pd'] / data['105Pd']
    data['106Pd/105Pd'] = data['106Pd'] / data['105Pd']
    data['110Pd/105Pd'] = data['110Pd'] / data['105Pd']
    cols1 = list(data.columns)
    #datao = pd.read_csv(outfile_corrected_raw_data + basename + '.csv', sep='\t')
    datao[cols1] = data[cols1]
    del cols1[0:8]
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)
    
    fig, ax = plt.subplots(2, 2, sharex = True, figsize = (50, 20))
    plt.rcParams['figure.dpi'] = 75 
    fig.suptitle(basename)
    #ax.flat[-1].set_visible(False)
    ax[4,2].set_axis_off()
    ax[4,1].set_axis_off()
    ax[4,3].set_axis_off()
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.4)
    
    
    for idx, (c,ax) in enumerate(zip(cols1, ax.flatten())):
        ax.scatter(data['Time'], data[c], color='r')
        ax.scatter(datao['Time'], datao[c], color='b')    
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax.margins(y=1.0)
        ax.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    
    fig.savefig(outfile_plt + basename + "_filtered.png", dpi=150)
    plt.close()
    
    
def plot_Ag_std(file): #False):
    global fullname
    global outfile
    global outfile_results
    global outfile_corrected_raw_data
    global outfile_plt


 
    
    #from outlier_Sn_std import cols
    fullname = os.path.join(outfile_results, 'Sn_export.csv')
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
    data['109Ag/107Ag'] = data['109Ag'] / data['107Ag']
    data['108Pd/105Pd'] = data['108Pd'] / data['105Pd']
    data['106Pd/105Pd'] = data['106Pd'] / data['105Pd']
    data['110Pd/105Pd'] = data['110Pd'] / data['105Pd']
    cols1 = list(data.columns)
    datao[cols1] = data[cols1]
    del cols1[0:8]
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)
    
    
    
    fig, ax = plt.subplots(5, 4, sharex = True, figsize = (50, 20))
    plt.rcParams['figure.dpi'] = 75 
    fig.suptitle(basename)
    ax[4,2].set_axis_off()
    ax[4,1].set_axis_off()
    ax[4,3].set_axis_off()
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.4)
    
    
    for idx, (c,ax) in enumerate(zip(cols1, ax.flatten())):
        ax.scatter(data['Time'], data[c], color='r')
        ax.scatter(datao['Time'], datao[c], color='b')    
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax.margins(y=1.0)
        ax.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    
    fig.savefig(outfile_plt + basename + "_filtered.png", dpi=150)
    plt.close()
    
def plot_Ag_sample(file): 
    
    global fullname
    global outfile
    global outfile_results 
    global outfile_corrected_raw_data
    global outfile_plt

  
    fullname = os.path.join(outfile_results, 'Sn_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    corrected_fullname = os.path.join(outfile_results, 'Sn_corrected.csv')
    pd.options.display.float_format = '{:.8f}'.format

    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    length = len(pd.read_csv(file))
    split = int(length / 3) - 1
    double_split = split * 2
    
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=split, index_col=False, dtype=float) 
    data2 = pd.read_csv(file, sep='\t', names=col_names, skiprows=6+split, nrows=split, index_col=False, dtype=float) 
    data3 = pd.read_csv(file, sep='\t', names=col_names, skiprows=6+double_split, index_col=False, dtype=float)
    data_all = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
    
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
    
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 2)
    data2o[colsa2] = data2o[colsa2].where(np.abs(stats.zscore(data2o[colsa2])) < 2)
    data3o[colsa3] = data3o[colsa3].where(np.abs(stats.zscore(data3o[colsa3])) < 2)
    
    

    
    fig, ax = plt.subplots(2, 2, sharex = True, figsize = (50, 20))
    plt.rcParams['figure.dpi'] = 75 
    fig.suptitle(basename)
    ax.flat[-1].set_visible(False)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.4)
   
    for idx, (c,ax) in enumerate(zip(colsa1, ax.flatten())):
        ax.scatter(data['Time'], data[c], color='r')
        ax.scatter(datao['Time'], datao[c], color='b')  
        ax.scatter(data2['Time'], data2[c], color='r')
        ax.scatter(data2o['Time'], data2o[c], color='g')  
        ax.scatter(data3['Time'], data3[c], color='r')
        ax.scatter(data3o['Time'], data3o[c], color='m')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax.margins(y=1.0)
        ax.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    
    fig.savefig(outfile_plt + basename + "_filtered.png", dpi=150)
    plt.close()
    
