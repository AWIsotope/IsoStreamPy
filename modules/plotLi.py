# -*- coding: utf-8 -*-
"""

This module containts the plotting for Li
!!! NOT TESTED; NOT IMPLENENTED IN ISOSTREAMPY YET !!!

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
from modules.config import WorkspaceVariableInput, outfile_plt_Li, outfile_corrected_raw_data_Li, outfile_results_Li, outfile_plt_Li_Std #outfile
from modules import outlierLi 
infile = WorkspaceVariableInput
save_dStandards = os.path.join(outfile_results_Li, 'Delta of Li Standards.csv') 
save_dStandardsExcel = os.path.join(outfile_results_Li, 'Delta of Li Standards.xlsx')

def plot_Li_blk(file):
    global fullname
    global outfile
    global outfile_results_Li
    global outfile_corrected_raw_data_Li
    global outfile_plt_Li

    fullname = os.path.join(outfile_results_Li, 'Li_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=20, index_col=False, dtype=float)

    dataLirrentplot = data.copy()
    dataLirrentplot = dataLirrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data = pd.concat([data, data[cols].div(data['6Li'], axis=0).add_suffix('/6Li')], axis=1)
    data.drop(['6.53/6Li'], axis = 1, inplace = True)
    data.drop(['6Li/6Li'], axis = 1, inplace = True)
    data.drop(['6.53'], axis = 1, inplace = True)

    cols1 = list(data.columns)
    datao[cols1] = data[cols1]
    del cols1[0:3]  
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)
    
    fig, ax = plt.subplots(2, 1, sharex = False, figsize = (15, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig.suptitle(basename)
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
    
    fig.savefig(outfile_plt_Li + basename + "_filtered.png", dpi=75)
    plt.close()
    

    fig2, ax2 = plt.subplots(figsize=(20,15))
    sb.scatterplot(data=dataLirrentplot, s = 150, ax = ax2)
    ax2.set_xlim(-1,)
    ax2.set_xlabel('Time (s)',size=20)
    ax2.set_ylabel('Voltage (V)',size=20)
    fig2.savefig(outfile_plt_Li + basename + "_Voltages.png", dpi=75)
    
    
def plot_Li_std(file):
    global fullname
    global outfile
    global outfile_results_Li
    global outfile_corrected_raw_data_Li
    global outfile_plt_Li

    fullname = os.path.join(outfile_results_Li, 'Li_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
  
    pd.options.display.float_format = '{:.8f}'.format
 
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
    dataLirrentplot = data.copy()
    dataLirrentplot = dataLirrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data = pd.concat([data, data[cols].div(data['6Li'], axis=0).add_suffix('/6Li')], axis=1)
    data.drop(['6.53/6Li'], axis = 1, inplace = True)
    data.drop(['6Li/6Li'], axis = 1, inplace = True)
    data.drop(['6.53'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    
    datao[cols1] = data[cols1]
    del cols1[0:3] 
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)
    
    
    fig, ax = plt.subplots(2, 1, sharex = False, figsize = (20, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig.suptitle(basename)
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
    
    fig.savefig(outfile_plt_Li + basename + "_filtered.png", dpi=75)
    plt.close()
    
    datalogLi = np.log(datao['7Li'] / datao['6Li'])
    
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

    ax2.scatter(datao['6Li'], datao['7Li'], color='r', marker='x')
    
    ax2.set_xlabel('$^{6}$Li')
    ax2.set_ylabel('$^{7}$Li')
    
    ci = 0.1 * np.std(datao['7Li']) / np.mean(datao['7Li'])
    ax2.fill_between(datao['6Li'], (datao['7Li']-ci), (datao['7Li']+ci), alpha=0.5)
    
    ax1.margins(y=1.0)
    ax2.margins(y=1.0)
    ax3.margins(y=1.0)
    ax1.set_xticks(ax1.get_xticks()[::3])
    ax2.set_xticks(ax2.get_xticks()[::3])
    fig2.savefig(outfile_plt_Li + basename + "_Ni+Li-Verhältnisse_ln.png", dpi=75)
    plt.close()
    
    fig4, ax4 = plt.subplots(figsize=(20,15))
    sb.scatterplot(data=dataLirrentplot, s = 150, ax = ax4)
    ax4.set_xlim(-1,)
    ax4.set_xlabel('Time (s)',size=20)
    ax4.set_ylabel('Voltage (V)',size=20)
    fig4.savefig(outfile_plt_Li + basename + "_Voltages.png", dpi=75)

    
##################################################
def plot_Li_sample(file):
    
    global fullname
    global outfile
    global outfile_results_Li 
    global outfile_corrected_raw_data_Li
    global outfile_plt_Li

    fullname = os.path.join(outfile_results_Li, 'Li_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    corrected_fullname = os.path.join(outfile_results_Li, 'Li_corrected.csv')
    pd.options.display.float_format = '{:.8f}'.format


    length = len(pd.read_csv(file))
    split = int(length / 2) - 1
    
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=split, index_col=False, dtype=float)
    data2 = pd.read_csv(file, sep='\t', names=col_names, skiprows=6+split, nrows=split, index_col=False, dtype=float)

    data_all = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
    dataLirrentplot = data_all.copy()
    dataLirrentplot = dataLirrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    datao[cols] = data[cols]
    data = pd.concat([data, data[cols].div(data['6Li'], axis=0).add_suffix('/6Li')], axis=1)
    data.drop(['6.53/6Li'], axis = 1, inplace = True)
    data.drop(['6Li/6Li'], axis = 1, inplace = True)
    data.drop(['6.53'], axis = 1, inplace = True)
    colsa1 = list(data.columns)

    cols2 = list(data2.drop(columns='Time').columns)
    data2o = pd.DataFrame({'Time':data2['Time']})
    data2o[cols] = data2[cols]
    data2 = pd.concat([data2, data2[cols2].div(data2['6Li'], axis=0).add_suffix('/6Li')], axis=1)
    data2.drop(['6.53/6Li'], axis = 1, inplace = True)
    data2.drop(['6Li/6Li'], axis = 1, inplace = True)
    data2.drop(['6.53'], axis = 1, inplace = True)
    colsa2 = list(data2.columns)

    TrueStdNi = 0.13860
    

    
    datao[colsa1] = data[colsa1]
    del colsa1[0:3] 
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 2)
    
    data2o[colsa2] = data2[colsa2]
    data2o[colsa2] = data2o[colsa2].where(np.abs(stats.zscore(data2o[colsa2])) < 2)
    
    
    datalogLi = np.log(datao['7Li'] / datao['6Li'])
    datalogLi2 = np.log(data2o['7Li'] / data2o['6Li'])

    datalog = pd.DataFrame()
    data2log = pd.DataFrame()

    datalog['ln(7Li/6Li)'] = datalogLi
    data2log['ln(7Li/6Li)'] = datalogLi2

    datalogLi_all = np.log(data_all['7Li']/data_all['6Li'])    
    
 
    fig, ax = plt.subplots(2, 1, sharex = True, figsize = (15, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig.suptitle(basename)
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
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax.margins(y=1.0)
        ax.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    
    fig.savefig(outfile_plt_Li + basename + "_filtered.png", dpi=75)
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
 
    ax[0, 0].scatter(datao['6Li'], datao['7Li'], color='r', marker='x')
    ax[0, 0].scatter(data2o['6Li'], data2o['7Li'], color='b', marker='x')
    ax[0, 0].set_xlabel('$^{6}$Li')
    ax[0, 0].set_ylabel('$^{7}$Li')
    ax[0, 0].margins(y=1.0)
  
    ax[0, 1].margins(y=1.0)
    ax[1, 0].margins(y=1.0)
    ax[0, 0].set_xticks(ax[0, 0].get_xticks()[::3])
    ax[0, 1].set_xticks(ax[0, 1].get_xticks()[::3])
    ax[1, 0].set_xticks(ax[1, 0].get_xticks()[::3])
    fig2.savefig(outfile_plt_Li + basename + "_Li-Verhältnisse_ln.png", dpi=75)
    plt.close()
 
    colso = list(datalog.columns) 
    
    fig3, ax3  = plt.subplots(2, 1, figsize = (20, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig3.suptitle(basename)

    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.6)
    
    
    for idx, (c,ax3) in enumerate(zip(colso, ax3.flatten())):
        ax3.scatter(datalogLi, datalog[c], color='b')    
        ax3.set_xlabel('ln($^{7}$Li/$^{6}$Li)')
        ax3.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax3.margins(y=1.0)
        ax3.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    fig3.savefig(outfile_plt_Li + basename + "_1_ln.png", dpi=75)
    plt.close()
    
    colso2 = list(data2log.columns) 
    
    fig4, ax4 = plt.subplots(2, 1, figsize = (20, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig4.suptitle(basename)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.6)
    
    
    for idx, (c,ax4) in enumerate(zip(colso2, ax4.flatten())):
        ax4.scatter(datalogLi2, data2log[c], color='b')    
        ax4.set_xlabel('ln($^{7}$Li/$^{6}$Li)')
        ax4.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax4.margins(y=1.0)
        ax4.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    fig4.savefig(outfile_plt_Li + basename + "_2_ln.png", dpi=75)
    plt.close()

    fig6, ax4 = plt.subplots(figsize=(20,15))
    sb.scatterplot(data=dataLirrentplot, s = 150, ax = ax4)
    ax4.set_xlim(-1,)
    ax4.set_xlabel('Time (s)',size=20)
    ax4.set_ylabel('Voltage (V)',size=20)
    fig6.savefig(outfile_plt_Li + basename + "_Voltages.png", dpi=75)
    