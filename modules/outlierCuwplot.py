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
from sklearn.metrics import mean_squared_error, r2_score

sb.set()
plt.ioff()
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rcParams.update({'font.size': 16})

from modules.config import outfile, WorkspaceVariableInput, outfile_plt, outfile_corrected_raw_data, outfile_results 
infile = WorkspaceVariableInput

def outlier_Cu_blk(file, append=True): #False):
    global fullname
    global outfile
    global outfile_results
    global outfile_corrected_raw_data
    global outfile_plt

    fullname = os.path.join(outfile_results, 'Cu_export.csv')
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
    
    
    datao.to_csv(outfile_corrected_raw_data + basename + '.csv', sep='\t', header = True, index_label='Index_name')

    mean_filtered_transposed = pd.DataFrame(data=np.mean(datao)).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file)

    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')
     
    fig, ax = plt.subplots(2, 1, sharex = False, figsize = (15, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig.suptitle(basename)
    ax.flat[-1].set_visible(False)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    for idx, (c,ax) in enumerate(zip(cols1, ax.flatten())):
        ax.scatter(data['Time'], data[c], color='r')
        ax.scatter(datao['Time'], datao[c], color='b')    
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax.margins(y=1.0)
        ax.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    
    fig.savefig(outfile_plt + basename + "_filtered.png", dpi=75)
    plt.close()
    
def outlier_Cu_std(file, append=True):
    global outfile
    global outfile_results
    global outfile_corrected_raw_data
    global outfile_plt

    fullname = os.path.join(outfile_results, 'Cu_export.csv')
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

    datao.to_csv(outfile_corrected_raw_data + basename + '.csv', sep='\t', header = True, index_label='Index_name')
  
    mean_filtered_transposed = pd.DataFrame(data=np.mean(datao)).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file)

    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')
        
    fig, ax = plt.subplots(2, 1, sharex = False, figsize = (20, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig.suptitle(basename)
    ax.flat[-1].set_visible(False)
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
    
    fig.savefig(outfile_plt + basename + "_filtered.png", dpi=75)
    plt.close()
    
    datalogNi = np.log(datao['62Ni'] / datao['60Ni'])
    datalogCu = np.log(datao['65Cu'] / datao['63Cu'])
    
    fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (20, 20))
    fig2.suptitle(basename)
    fig2.tight_layout()
    plt.rcParams.update({'font.size': 14})
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    ax1.scatter(datao['60Ni'], datao['62Ni'], color='r', marker='x')

    ax1.set_xlabel('$^{60}$Ni')
    ax1.set_ylabel('$^{62}$Ni')

    ax2.scatter(datao['63Cu'], datao['65Cu'], color='r', marker='x')
    
    ax2.set_xlabel('$^{63}$Cu')
    ax2.set_ylabel('$^{65}$Cu')
    
    ax3.scatter(datalogNi, datalogCu, color = 'r', marker = 'x')

    ax3.set_xlabel('ln($^{62}$Ni/$^{60}$Ni)')
    ax3.set_ylabel('ln($^{65}$Cu/$^{63}$Cu)')
    ci = 0.1 * np.std(datao['65Cu']) / np.mean(datao['65Cu'])
    ax2.fill_between(datao['63Cu'], (datao['65Cu']-ci), (datao['65Cu']+ci), alpha=0.5)
    
    ax1.margins(y=1.0)
    ax2.margins(y=1.0)
    ax3.margins(y=1.0)
    ax1.set_xticks(ax1.get_xticks()[::3])
    ax2.set_xticks(ax2.get_xticks()[::3])
    ax3.set_xticks(ax3.get_xticks()[::3])
    fig2.savefig(outfile_plt + basename + "_Ni+Cu-Verhältnisse_ln.png", dpi=75)
    plt.close()
    
def outlier_Cu_sample(file = '220309014_NBS2.TXT', append=True): #False):
    global outfile
    global outfile_results 
    global outfile_corrected_raw_data
    global outfile_plt

    fullname = os.path.join(outfile_results, 'Cu_export.csv')
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
    datao = pd.concat([datao, datao[cols].div(datao['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    datao = pd.concat([datao, datao[cols].div(datao['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    datao.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    colsa1 = list(data.columns)

    cols2 = list(data2.drop(columns='Time').columns)
    data2o = pd.DataFrame({'Time':data2['Time']})
    data2o[cols] = data2[cols]
    data2o = pd.concat([data2o, data2o[cols2].div(data2o['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data2o = pd.concat([data2o, data2o[cols2].div(data2o['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data2o.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    colsa2 = list(data2.columns)

    cols3 = list(data3.drop(columns='Time').columns)
    data3o = pd.DataFrame({'Time':data3['Time']})
    data3o[cols] = data3[cols]
    data3o = pd.concat([data3o, data3o[cols3].div(data3o['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data3o = pd.concat([data3o, data3o[cols3].div(data3o['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data3o.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    colsa3 = list(data3.columns)
    
    TrueStdNi = 0.13860
    
    fnistd = (np.log(TrueStdNi / ((datao['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
    datao['65Cu/63Cu_corrected'] = datao['65Cu/63Cu'] * ((64.927790 / 62.929598) ** fnistd)
    
    fnistd2 = (np.log(TrueStdNi / ((data2o['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
    data2o['65Cu/63Cu_corrected'] = data2o['65Cu/63Cu'] * ((64.927790 / 62.929598) ** fnistd2)

    fnistd3 = (np.log(TrueStdNi / ((data3o['62Ni/60Ni']))) / (np.log(61.928345 / 59.930786)))
    data3o['65Cu/63Cu_corrected'] = data3o['65Cu/63Cu'] * ((64.927790 / 62.929598) ** fnistd3)
    
    datao[colsa1] = data[colsa1]
    del colsa1[0:7] 
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 2)
    
    data2o[colsa2] = data2[colsa2]
    del colsa2[0:7] 
    data2o[colsa2] = data2o[colsa2].where(np.abs(stats.zscore(data2o[colsa2])) < 2)
    
    data3o[colsa3] = data3[colsa3]
    del colsa3[0:7] 
    data3o[colsa3] = data3o[colsa3].where(np.abs(stats.zscore(data3o[colsa3])) < 2)
    
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
     
    #ln 
    datalogNi = np.log(datao['62Ni'] / datao['60Ni'])
    datalogCu = np.log(datao['65Cu'] / datao['63Cu'])
    datalogNi2 = np.log(data2o['62Ni'] / data2o['60Ni'])
    datalogCu2 = np.log(data2o['65Cu'] / data2o['63Cu'])
    datalogNi3 = np.log(data3o['62Ni'] / data3o['60Ni'])
    datalogCu3 = np.log(data3o['65Cu'] / data3o['63Cu'])

    fig, ax = plt.subplots(2, 1, sharex = False, figsize = (20, 15))
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
    
    fig.savefig(outfile_plt + basename + "_filtered.png", dpi=75)
    plt.close()
    
    fig2, ax = plt.subplots(nrows =2, ncols = 2, figsize = (20, 15))
    fig2.suptitle(basename)
    fig2.tight_layout()
    plt.rcParams.update({'font.size': 14})
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    ax.flat[-1].set_visible(False)
    ax[0, 0].scatter(datao['60Ni'], datao['62Ni'], color='r', marker='x')
    ax[0, 0].scatter(data2o['60Ni'], data2o['62Ni'], color='b', marker='x')
    ax[0, 0].scatter(data3o['60Ni'], data3o['62Ni'], color='g', marker='x')

    ax[0, 0].set_xlabel('$^{60}$Ni')
    ax[0, 0].set_ylabel('$^{62}$Ni')

    ax[0, 1].scatter(datao['63Cu'], datao['65Cu'], color='r', marker='x')
    ax[0, 1].scatter(data2o['63Cu'], data2o['65Cu'], color='b', marker='x')
    ax[0, 1].scatter(data3o['63Cu'], data3o['65Cu'], color='g', marker='x')
    
    ax[0, 1].set_xlabel('$^{63}$Cu')
    ax[0, 1].set_ylabel('$^{65}$Cu')
    
    ax[1, 0].scatter(datalogNi, datalogCu, color = 'r', marker = 'x')
    ax[1, 0].scatter(datalogNi2, datalogCu2, color = 'b', marker = 'x')
    ax[1, 0].scatter(datalogNi3, datalogCu3, color = 'g', marker = 'x')
    ax[1, 0].set_xlabel('ln($^{62}$Ni/$^{60}$Ni)')
    ax[1, 0].set_ylabel('ln($^{65}$Cu/$^{63}$Cu)')
    
    ax[0, 0].margins(y=1.0)
    
    ax[0, 1].margins(y=1.0)
    ax[1, 0].margins(y=1.0)
    ax[0, 0].set_xticks(ax[0, 0].get_xticks()[::3])
    ax[0, 1].set_xticks(ax[0, 1].get_xticks()[::3])
    ax[1, 0].set_xticks(ax[1, 0].get_xticks()[::3])
    fig2.savefig(outfile_plt + basename + "_Ni+Cu-Verhältnisse_ln.png", dpi=75)
    plt.close()
    
