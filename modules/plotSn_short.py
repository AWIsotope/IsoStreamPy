# -*- coding: utf-8 -*-
"""
This module containts the Plotting for Sn "Short"

!!! NOT TESTED; NOT IMPLEMENTED IN ISOSTREAMPY - ONLY FOR TESTING !!!

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
from modules.config import WorkspaceVariableInput, outfile_plt_Sn, outfile_corrected_raw_data_Sn, outfile_results_Sn 
from modules import outlierSn 
infile = WorkspaceVariableInput


def plot_Sn_blk(file):
    global fullname
    global outfile_results_Sn
    global outfile_corrected_raw_data_Sn
    global outfile_plt_Sn
    fullname = os.path.join(outfile_results_Sn, 'Sn_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=30, index_col=False, dtype=float)

    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data = pd.concat([data, data[cols].div(data['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    data = pd.concat([data, data[cols].div(data['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    data['123Sb/121Sb'] = data['123Sb'] / data['121Sb']
    data.drop(['120Sn/120Sn', '116Sn/116Sn'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    datao[cols1] = data[cols1]
    del cols1[0:10]
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
    
    
def plot_Sn_std(file): 
    global fullname
    global outfile_results_Sn
    global outfile_corrected_raw_data_Sn
    global outfile_plt_Sn
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
    data = pd.concat([data, data[cols].div(data['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    data = pd.concat([data, data[cols].div(data['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    data['123Sb/121Sb'] = data['123Sb'] / data['121Sb']
    data.drop(['120Sn/120Sn', '116Sn/116Sn'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    datao[cols1] = data[cols1]
    del cols1[0:10]
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
    
    datalog116Sn120Sn = np.log(datao['116Sn/120Sn'])
    datalogSb = np.log(datao['123Sb/121Sb'])
    datalog118Sn120Sn = np.log(datao['118Sn/120Sn'])
    datalog122Sn120Sn = np.log(datao['122Sn/120Sn'])
    datalog124Sn120Sn = np.log(datao['124Sn/120Sn'])
    datalog124Sn116Sn = np.log(datao['124Sn/116Sn'])
    datalog122Sn116Sn = np.log(datao['122Sn/116Sn'])
    
    datalog = pd.DataFrame()
    datalog['ln(116Sn/120Sn)'] = datalog116Sn120Sn
    datalog['ln(123Sb/121Sb)'] = datalogSb
    datalog['ln(118Sn/120Sn)'] = datalog118Sn120Sn
    datalog['ln(122Sn/120Sn)'] = datalog122Sn120Sn
    datalog['ln(124Sn/120Sn)'] = datalog124Sn120Sn
    datalog['ln(124Sn/116Sn)'] = datalog124Sn116Sn
    datalog['ln(122Sn/116Sn)'] = datalog122Sn116Sn
    
    
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
    ax1.scatter(datao['123Sb'], datao['121Sb'], color='r', marker='x')

    ax1.set_xlabel('$^{123}$Sb')
    ax1.set_ylabel('$^{121}$Sb')

    ax2.scatter(datao['116Sn'], datao['120Sn'], color='r', marker='x')
    
    ax2.set_xlabel('$^{116}$Sn')
    ax2.set_ylabel('$^{120}$Sn')
    
    ax3.scatter(datalogSb, datalog116Sn120Sn, color = 'r', marker = 'x')

    ax3.set_xlabel('ln($^{123}$Sb/$^{121}$Sb)')
    ax3.set_ylabel('ln($^{116}$Sn/$^{120}$Sn)')

    
    ax1.margins(y=1.0)
    ax2.margins(y=1.0)
    ax3.margins(y=1.0)
    ax1.set_xticks(ax1.get_xticks()[::3])
    ax2.set_xticks(ax2.get_xticks()[::3])
    ax3.set_xticks(ax3.get_xticks()[::3])
    fig2.savefig(outfile_plt + basename + "_Sb+Sn-Verhältnisse_ln.png", dpi=75)
    plt.close()
    
    
    colso = list(datalog.columns) 
    
    fig4, ax4 = plt.subplots(4, 3, figsize = (20, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig4.suptitle(basename)
    ax4.flat[-1].set_visible(False)
    ax4.flat[-2].set_visible(False)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.6)
    
    
    for idx, (c,ax4) in enumerate(zip(colso, ax4.flatten())):
        ax4.scatter(datalog['ln(124Sn/120Sn)'], datalog[c], color='b')    
        ax4.set_xlabel('ln($^{124}$Sn/$^{120}$Sn)')
        ax4.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax4.margins(y=1.0)
        ax4.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    fig4.savefig(outfile_plt + basename + "_ln.png", dpi=75)
    plt.close()
    
def plot_Sn_sample(file): 
    
    global fullname
    global outfile_results_Sn
    global outfile_corrected_raw_data_Sn
    global outfile_plt_Sn
    fullname = os.path.join(outfile_results_Sn, 'Sn_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    corrected_fullname = os.path.join(outfile_results_Sn, 'Sn_corrected.csv')
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

    data = pd.concat([data, data[cols].div(data['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    data = pd.concat([data, data[cols].div(data['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    data['123Sb/121Sb'] = data['123Sb'] / data['121Sb']
    data.drop(['120Sn/120Sn', '116Sn/116Sn'], axis = 1, inplace = True)
    
    data2 = pd.concat([data2, data2[cols2].div(data2['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    data2 = pd.concat([data2, data2[cols2].div(data2['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    data2['123Sb/121Sb'] = data2['123Sb'] / data2['121Sb']
    data2.drop(['120Sn/120Sn', '116Sn/116Sn'], axis = 1, inplace = True)
    
    data3 = pd.concat([data3, data3[cols3].div(data3['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    data3 = pd.concat([data3, data3[cols3].div(data3['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    data3['123Sb/121Sb'] = data3['123Sb'] / data3['121Sb']
    data3.drop(['120Sn/120Sn', '116Sn/116Sn'], axis = 1, inplace = True)
    
    colsa1 = list(data.columns)
    colsa2 = list(data2.columns)
    colsa3 = list(data3.columns)

    datao[colsa1] = data[colsa1]
    data2o[colsa2] = data2[colsa2]
    data3o[colsa3] = data3[colsa3]
    
    del colsa1[0:10]
    del colsa2[0:10]
    del colsa3[0:10]
    
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 2)
    data2o[colsa2] = data2o[colsa2].where(np.abs(stats.zscore(data2o[colsa2])) < 2)
    data3o[colsa3] = data3o[colsa3].where(np.abs(stats.zscore(data3o[colsa3])) < 2)
    
    
    datalog116Sn120Sn = np.log(datao['116Sn/120Sn'])
    datalogSb = np.log(datao['123Sb/121Sb'])
    datalog118Sn120Sn = np.log(datao['118Sn/120Sn'])
    datalog122Sn120Sn = np.log(datao['122Sn/120Sn'])
    datalog124Sn120Sn = np.log(datao['124Sn/120Sn'])
    datalog124Sn116Sn = np.log(datao['124Sn/116Sn'])
    datalog122Sn116Sn = np.log(datao['122Sn/116Sn'])
    
    datalog = pd.DataFrame()
    datalog['ln(116Sn/120Sn)'] = datalog116Sn120Sn
    datalog['ln(123Sb/121Sb)'] = datalogSb
    datalog['ln(118Sn/120Sn)'] = datalog118Sn120Sn
    datalog['ln(122Sn/120Sn)'] = datalog122Sn120Sn
    datalog['ln(124Sn/120Sn)'] = datalog124Sn120Sn
    datalog['ln(124Sn/116Sn)'] = datalog124Sn116Sn
    datalog['ln(122Sn/116Sn)'] = datalog122Sn116Sn

    
    data2log116Sn120Sn = np.log(data2o['116Sn/120Sn'])
    data2logSb = np.log(data2o['123Sb/121Sb'])
    data2log118Sn120Sn = np.log(data2o['118Sn/120Sn'])
    data2log122Sn120Sn = np.log(data2o['122Sn/120Sn'])
    data2log124Sn120Sn = np.log(data2o['124Sn/120Sn'])
    data2log124Sn116Sn = np.log(data2o['124Sn/116Sn'])
    data2log122Sn116Sn = np.log(data2o['122Sn/116Sn'])
    
    data2log = pd.DataFrame()
    data2log['ln(116Sn/120Sn)'] = data2log116Sn120Sn
    data2log['ln(123Sb/121Sb)'] = data2logSb
    data2log['ln(118Sn/120Sn)'] = data2log118Sn120Sn
    data2log['ln(122Sn/120Sn)'] = data2log122Sn120Sn
    data2log['ln(124Sn/120Sn)'] = data2log124Sn120Sn
    data2log['ln(124Sn/116Sn)'] = data2log124Sn116Sn
    data2log['ln(122Sn/116Sn)'] = data2log122Sn116Sn
  
    data3log116Sn120Sn = np.log(data3o['116Sn/120Sn'])
    data3logSb = np.log(data3o['123Sb/121Sb'])
    data3log118Sn120Sn = np.log(data3o['118Sn/120Sn'])
    data3log122Sn120Sn = np.log(data3o['122Sn/120Sn'])
    data3log124Sn120Sn = np.log(data3o['124Sn/120Sn'])
    data3log124Sn116Sn = np.log(data3o['124Sn/116Sn'])
    data3log122Sn116Sn = np.log(data3o['122Sn/116Sn'])
    
    data3log = pd.DataFrame()
    data3log['ln(116Sn/120Sn)'] = data3log116Sn120Sn
    data3log['ln(123Sb/121Sb)'] = data3logSb
    data3log['ln(118Sn/120Sn)'] = data3log118Sn120Sn
    data3log['ln(122Sn/120Sn)'] = data3log122Sn120Sn
    data3log['ln(124Sn/120Sn)'] = data3log124Sn120Sn
    data3log['ln(124Sn/116Sn)'] = data3log124Sn116Sn
    data3log['ln(122Sn/116Sn)'] = data3log122Sn116Sn

     
    #ln 
    datalogSn = np.log(datao['116Sn/120Sn'])
    datalogSb = np.log(datao['123Sb/121Sb'])
    datalog2Sn = np.log(data2o['116Sn/120Sn'])
    datalog2Sb = np.log(data2o['123Sb/121Sb'])
    datalog3Sn = np.log(data3o['116Sn/120Sn'])
    datalog3Sb = np.log(data3o['123Sb/121Sb'])

    datalogSn_all = np.log(data_all['116Sn']/data_all['120Sn'])
    datalogSb_all = np.log(data_all['123Sb']/data_all['121Sb'])    
    
    #regression

    reg = np.polyfit(datalogSb_all, datalogSn_all, 1)
    predict = np.poly1d(reg) # Slope and intercept
    trend = np.polyval(reg, datalogSb_all)
    std = datalogSn_all.std() # Standard deviation
    r2 = np.round(r2_score(datalogSn_all, predict(datalogSb_all)), 5) #R-squared
    r2string = str(r2)
    predict = np.poly1d(reg)
    
    fig, ax = plt.subplots(5, 4, sharex = True, figsize = (50, 20))
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
    
    fig2, ax = plt.subplots(nrows =2, ncols = 2, figsize = (20, 15), label='Inline label')
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
    ax[0, 0].scatter(datao['123Sb'], datao['121Sb'], color='r', marker='x', label='1st Block')
    ax[0, 0].scatter(data2o['123Sb'], data2o['121Sb'], color='b', marker='x', label='2nd Block')
    ax[0, 0].scatter(data3o['123Sb'], data3o['121Sb'], color='g', marker='x', label='3rd Block')

    ax[0, 0].set_xlabel('$^{123}$Sb')
    ax[0, 0].set_ylabel('$^{121}$Sb')

    ax[0, 1].scatter(datao['116Sn'], datao['120Sn'], color='r', marker='x', label='1st Block')
    ax[0, 1].scatter(data2o['116Sn'], data2o['120Sn'], color='b', marker='x', label='2nd Block')
    ax[0, 1].scatter(data3o['116Sn'], data3o['120Sn'], color='g', marker='x', label='3rd Block')
    
    ax[0, 1].set_xlabel('$^{116}$Sn')
    ax[0, 1].set_ylabel('$^{120}$Sn')
    
    ax[1, 0].scatter(datalogSb, datalogSn, color = 'r', marker = 'o', label='1st Block')
    ax[1, 0].scatter(datalog2Sb, datalog2Sn, color = 'b', marker = 'o', label='2nd Block')
    ax[1, 0].scatter(datalog3Sb, datalog3Sn, color = 'g', marker = 'o', label='3rd Block')
    ax[1, 0].set_xlabel('ln($^{123}$Sb/$^{121}$Sb)')
    ax[1, 0].set_ylabel('ln($^{116}$Sn/$^{120}$Sn)')
    ax[1, 0].plot(datalogSb_all, trend, 'k--')
    ax[1, 0].plot(datalogSb_all, trend - std, 'c--')
    ax[1, 0].plot(datalogSb_all, trend + std, 'c--')
    ax[1, 0].plot([], [], ' ', label='R$^{2}$ = '+r2string)
    ax[0, 0].margins(y=1.0)
    ax[0, 1].margins(y=1.0)
    ax[1, 0].margins(y=1.0)
    
    ax[0, 0].set_xticks(ax[0, 0].get_xticks()[::1])
    ax[0, 1].set_xticks(ax[0, 1].get_xticks()[::1])

    ax[0, 0].legend(loc=1)
    ax[0, 1].legend(loc=1)
    ax[1, 0].legend(loc=1)
    
    fig2.savefig(outfile_plt + basename + "_Sb+Sn-Verhältnisse_ln.pdf", dpi=75)
    plt.close()
    
    colso = list(datalog.columns) 
    
    fig3, ax3  = plt.subplots(4, 3, figsize = (20, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig3.suptitle(basename)
    ax3.flat[-1].set_visible(False)
    ax3.flat[-2].set_visible(False)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.6)
    
    
    for idx, (c,ax3) in enumerate(zip(colso, ax3.flatten())):
        ax3.scatter(datalog['ln(124Sn/120Sn)'], datalog[c], color='b')    
        ax3.set_xlabel('ln($^{124}$Sn/$^{120}$Sn)')
        ax3.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax3.margins(y=1.0)
        ax3.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    fig3.savefig(outfile_plt + basename + "_1_ln.png", dpi=75)
    plt.close()
    
    colso2 = list(data2log.columns) 
    
    fig4, ax4 = plt.subplots(4, 3, figsize = (20, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig4.suptitle(basename)
    ax4.flat[-1].set_visible(False)
    ax4.flat[-2].set_visible(False)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.6)
    
    
    for idx, (c,ax4) in enumerate(zip(colso2, ax4.flatten())):
        ax4.scatter(data2log['ln(124Sn/120Sn)'], data2log[c], color='b')    
        ax4.set_xlabel('ln($^{124}$Sn/$^{120}$Sn)')
        ax4.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax4.margins(y=1.0)
        ax4.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    fig4.savefig(outfile_plt + basename + "_2_ln.png", dpi=75)
    plt.close()
    
    colso3 = list(data3log.columns)
    
    fig5, ax5 = plt.subplots(4, 3, figsize = (20, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig5.suptitle(basename)
    ax5.flat[-1].set_visible(False)
    ax5.flat[-2].set_visible(False)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.6)
    
    
    for idx, (c,ax5) in enumerate(zip(colso3, ax5.flatten())):
        ax5.scatter(data3log['ln(124Sn/120Sn)'], data3log[c], color='b')    
        ax5.set_xlabel('ln($^{124}$Sn/$^{120}$Sn)')
        ax5.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax5.margins(y=1.0)
        ax5.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    fig5.savefig(outfile_plt + basename + "_3_ln.png", dpi=75)
    plt.close()
    
    
    fig6, ax6  = plt.subplots(4, 3, figsize = (20, 15))
    plt.rcParams['figure.dpi'] = 75 
    fig6.suptitle(basename)
    ax6.flat[-1].set_visible(False)
    ax6.flat[-2].set_visible(False)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.6)
    
    
    for idx, (c,ax6) in enumerate(zip(colso, ax6.flatten())):
        ax6.scatter(datalog['ln(124Sn/120Sn)'], datalog[c], color='b')    
        ax6.set_xlabel('ln($^{124}$Sn/$^{120}$Sn)')
        ax6.set_ylabel('${}${}'.format(c[:2], c[2:]))
        ax6.margins(y=1.0)
        ax6.set_title('${}${}'.format(c[:2], c[2:]), color='b')
    fig6.savefig(outfile_plt + basename + "_1_ln.png", dpi=75)
    plt.close()
    
