# -*- coding: utf-8 -*-
"""
This module containts the Plotting for Sn

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
plt.rcParams.update({'font.size': 16})
import sys
sys.path.append('')
from modules.config import WorkspaceVariableInput, outfile_plt_Sb, outfile_corrected_raw_data_Sb, outfile_results_Sb, outfile_plt_Sb_Std 
from modules import outlierSn 
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pickle
from statistics import mean


file_path = os.path.join("modules", "variable_data.pkl")

# Variable mit pickle laden
with open(file_path, 'rb') as file:
    standardoutputpfad = pickle.load(file)
    


file_path2 = os.path.join("modules", "variable_data_input.pkl")
# Variable mit pickle laden
with open(file_path2, 'rb') as file2:
    standardinputpfad = pickle.load(file2)

def load_files(path, extension):
    return sorted([f for f in os.listdir(path) if f.endswith(extension)])


infile = WorkspaceVariableInput
save_dStandards = os.path.join(outfile_results_Sb, 'Delta of Sb Standards.csv') 
save_dStandardsExcel = os.path.join(outfile_results_Sb, 'Delta of Sb Standards.xlsx')

def plot_Sb_blk(file): #False):
    global fullname
    global outfile
    global outfile_results_Sb
    global outfile_corrected_raw_data_Sb
    global outfile_plt_Sb

    fullname = os.path.join(outfile_results_Sb, 'Sb_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=30, index_col=False, dtype=float)

    dataSbrrentplot = data.copy()
    dataSbrrentplot = dataSbrrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    
    data.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    cols_sn = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb = ['123Sb']  # Nur 123Sb/121Sb

    # Berechnung und Hinzufügen der Verhältnisse
    data = pd.concat([data, data[cols_sn].div(data['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data = pd.concat([data, data[cols_sb].div(data['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)

    cols1 = list(data.columns)
    datao[cols1] = data[cols1]
    del cols1[0:5] 
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

    fig2, ax2 = plt.subplots(figsize=(20,15))
    sb.scatterplot(data=dataSbrrentplot, s = 150, ax = ax2)
    ax2.set_xlim(-1,)
    ax2.set_xlabel('Time (s)',size=20)
    ax2.set_ylabel('Voltage (V)',size=20)
    st.pyplot(fig)
    st.pyplot(fig2)
    
    
    
def plot_Sb_std(file):
    global fullname
    global outfile
    global outfile_results_Sb
    global outfile_corrected_raw_data_Sb
    global outfile_plt_Sb

    fullname = os.path.join(outfile_results_Sb, 'Sb_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
  
    pd.options.display.float_format = '{:.8f}'.format
 
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
    dataSbrrentplot = data.copy()
    dataSbrrentplot = dataSbrrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})

    data.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    cols_sn = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb = ['123Sb']  # Nur 123Sb/121Sb

    # Berechnung und Hinzufügen der Verhältnisse
    data = pd.concat([data, data[cols_sn].div(data['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data = pd.concat([data, data[cols_sb].div(data['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)
    cols1 = list(data.columns)
    
    
    
    
    datao[cols1] = data[cols1]
    del cols1[0:5] 
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)
    
    

    datalogSn = np.log(datao['120Sn'] / datao['118Sn'])
    datalogSb = np.log(datao['123Sb'] / datao['121Sb'])
    
 
    # Erstelle DataFrames mit den Daten
    df21 = pd.DataFrame({
        "ln(120Sn/118Sn)": datalogSn,
        "ln(123Sb/121Sb)": datalogSb,
        "Color": ['1. Block'] * len(datalogSn)
    })


    # Füge die DataFrames zusammen
  
# Erstelle einen Scatter-Plot mit Plotly Express
    fig222 = px.scatter(df21, y="ln(120Sn/118Sn)", x="ln(123Sb/121Sb)", color="Color")

    st.plotly_chart(fig222, use_container_width=True)
    
##################################################
def plot_Sb_sample(file):
    
    global fullname
    global outfile
    global outfile_results_Sb 
    global outfile_corrected_raw_data_Sb
    global outfile_plt_Sb

    print(outfile_results_Sb)
    fullname = os.path.join(outfile_results_Sb, 'Sb_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    corrected_fullname = os.path.join(outfile_results_Sb, 'Sb_corrected.csv')
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
    dataSbrrentplot = data_all.copy()
    dataSbrrentplot = dataSbrrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    datao[cols] = data[cols]

    data.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    datao.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    cols_sn = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb = ['123Sb']  # Nur 123Sb/121Sb

    # Berechnung und Hinzufügen der Verhältnisse
    data = pd.concat([data, data[cols_sn].div(data['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data = pd.concat([data, data[cols_sb].div(data['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)
    
    colsa1 = list(data.columns)

    cols2 = list(data2.drop(columns='Time').columns)
    data2o = pd.DataFrame({'Time':data2['Time']})
    data2o[cols] = data2[cols]

    data2.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    data2o.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    cols_sn2 = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb2 = ['123Sb']  # Nur 123Sb/121Sb

    # Berechnung und Hinzufügen der Verhältnisse
    data2 = pd.concat([data2, data2[cols_sn2].div(data2['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data2 = pd.concat([data2, data2[cols_sb2].div(data2['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)

    colsa2 = list(data2.columns)

    cols3 = list(data3.drop(columns='Time').columns)
    data3o = pd.DataFrame({'Time':data3['Time']})
    data3o[cols] = data3[cols]

    data3.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    data3o.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    cols_sn3 = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb3 = ['123Sb']  # Nur 123Sb/121Sb

    # Berechnung und Hinzufügen der Verhältnisse
    data3 = pd.concat([data3, data3[cols_sn3].div(data3['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data3 = pd.concat([data3, data3[cols_sb3].div(data3['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)

    
    colsa3 = list(data3.columns)
    
    masses = pd.DataFrame()
    masses.loc[0, '118Sn'] = 117.901606625
    masses.loc[0, '120Sn'] = 119.902202063
    masses.loc[0, '121Sb'] = 120.903810584
    masses.loc[0, '123Sb'] = 122.904211755
    
    TrueStdSn = 32.593 / 24.223
    
    datao[colsa1] = data[colsa1]
    del colsa1[0:5] 
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 1.5)
    
    data2o[colsa2] = data2[colsa2]
    del colsa2[0:5] 
    data2o[colsa2] = data2o[colsa2].where(np.abs(stats.zscore(data2o[colsa2])) < 2)
    
    data3o[colsa3] = data3[colsa3]
    del colsa3[0:5] 
    data3o[colsa3] = data3o[colsa3].where(np.abs(stats.zscore(data3o[colsa3])) < 2)
    
    
    fSnstd = (np.log(TrueStdSn / ((datao['120Sn/118Sn']))) / (np.log(119.902202063 / 117.901606625)))
    datao['123Sb/121Sb_corrected'] = datao['123Sb/121Sb'] * ((masses['123Sb'] / masses['121Sb']) ** fSnstd)
    
    fSnstd2 = (np.log(TrueStdSn / ((data2o['120Sn/118Sn']))) / (np.log(119.902202063 / 117.901606625)))
    data2o['123Sb/121Sb_corrected'] = data2o['123Sb/121Sb'] * ((masses['123Sb'] / masses['121Sb']) ** fSnstd2)

    fSnstd3 = (np.log(TrueStdSn / ((data3o['120Sn/118Sn']))) / (np.log(119.902202063 / 117.901606625)))
    data3o['123Sb/121Sb_corrected'] = data3o['123Sb/121Sb'] * ((masses['123Sb'] / masses['121Sb']) ** fSnstd3)

    
    datalogSn = np.log(datao['120Sn'] / datao['118Sn'])
    datalogSb = np.log(datao['123Sb'] / datao['121Sb'])
    datalogSn2 = np.log(data2o['120Sn'] / data2o['118Sn'])
    datalogSb2 = np.log(data2o['123Sb'] / data2o['121Sb'])
    datalogSn3 = np.log(data3o['120Sn'] / data3o['118Sn'])
    datalogSb3 = np.log(data3o['123Sb'] / data3o['121Sb'])
    
    datalog = pd.DataFrame()
    data2log = pd.DataFrame()
    data3log = pd.DataFrame()

    datalog['ln(120Sn/118Sn)'] = datalogSn
    datalog['ln(123Sb/121Sb)'] = datalogSb
    data2log['ln(120Sn/118Sn)'] = datalogSn2
    data2log['ln(123Sb/121Sb)'] = datalogSb2
    data3log['ln(120Sn/118Sn)'] = datalogSn3
    data3log['ln(123Sb/121Sb)'] = datalogSb3
    
    datalogSn_all = np.log(data_all['120Sn']/data_all['118Sn'])
    datalogSb_all = np.log(data_all['123Sb']/data_all['121Sb'])    
    

    # Erstelle DataFrames mit den Daten
    df21 = pd.DataFrame({
        "ln(120Sn/118Sn)": datalogSn,
        "ln(123Sb/121Sb)": datalogSb,
        "Color": ['1. Block'] * len(datalogSn)
    })

    df22 = pd.DataFrame({
        "ln(120Sn/118Sn)": datalogSn2,
        "ln(123Sb/121Sb)": datalogSb2,
        "Color": ['2. Block'] * len(datalogSn2)
    })

    df23 = pd.DataFrame({
        "ln(120Sn/118Sn)": datalogSn3,
        "ln(123Sb/121Sb)": datalogSb3,
        "Color": ['3. Block'] * len(datalogSn3)
    })

    # Füge die DataFrames zusammen
    df222 = pd.concat([df21, df22, df23])

# Erstelle einen Scatter-Plot mit Plotly Express
    fig222 = px.scatter(df222, y="ln(120Sn/118Sn)", x="ln(123Sb/121Sb)", color="Color")
    st.plotly_chart(fig222, use_container_width=True)

    
def plot_Sb_Standards():

    global outfile_plt_Sb_Std
    global outfile_results_Sb
    global outfile_corrected_raw_data_Sb
    global outfile_plt_Sb
    
    save = outfile_plt_Sb + '/Standards'
    sb.set(rc={'figure.figsize':(25.7,8.27)})
    
    fullname = os.path.join(outfile_results_Sb, 'Sb_export.csv')
    pd.options.mode.chained_assignment = None
    data = pd.read_csv(fullname, sep='\t')
    data['Name'] = data['Filename']
    data = data.set_index('Filename')

    datalog = pd.DataFrame(index=data.index)
    datalog['log123Sb/121Sb'] = np.log((data['123Sb']/data['121Sb']))
    datalog['log120Sn/118Sn'] = np.log((data['120Sn']/data['118Sn']))
    datalog['standard'] = datalog.index.str.contains('_Nis0|Nis1')
    datalog['Name'] = data['Name']

    datalogStandard=datalog[datalog['standard'] == True]
    datalogStandard.insert(4, 'ID', range(1, 1+len(datalogStandard)))
    datalogStandard['ID'] = datalogStandard['ID'].round(decimals=0).astype(object)

    slope, intercept, r, p, se = stats.linregress(datalogStandard['log120Sn/118Sn'], datalogStandard['log123Sb/121Sb'])
    rsquared = r**2   


    
    fig = px.scatter(datalogStandard, x='log120Sn/118Sn', y='log123Sb/121Sb', text='ID', labels={'x':'ln($^{120}$Sn/$^{118}$Sn)', 'y':'ln($^{123}$Sb/$^{121}$Sb)'})
    fig.update_traces(textposition='top center')
    fig.add_trace(px.line(datalogStandard, x='log120Sn/118Sn', y=intercept + slope*datalogStandard['log120Sn/118Sn']).data[0])
    fig.update_layout(title_text='ln($^{120}$Sn/$^{118}$Sn) vs ln($^{123}$Sb/$^{121}$Sb) of Standards', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)



def plot_dSb_Standards():
    global outfile_plt_Sb_Std
    global outfile_results_Sb
    global outfile_corrected_raw_data_Sb
    global outfile_plt_Sb
    global save_dStandards
    global save_dStandardsExcel
    save = outfile_plt_Sb + '/Standards'
    sb.set_theme(rc={'figure.figsize':(25.7,8.27)})
    
    fullname = os.path.join(outfile_results_Sb, 'Sb_export.csv')
    pd.options.mode.chained_assignment = None
    df = pd.read_csv(fullname, sep='\t')
    
    
    True_Snckel = 0.0312145 / 0.2120231
    
    df = df[df['Filename'].str.contains('Blk')==False]
    df = df[df['Filename'].str.contains('_Nis0|Nis1')]
    for index in df.index:
        df['118Sn/120Sn'] = (df['118Sn'] / df['120Sn'])
        
    True_118Sn_120Sn_mean = np.mean(df['118Sn/120Sn'])
    True_118Sn_120Sn_mean_2SD = np.std(df['118Sn/120Sn']) * 2
    True_118Sn_120Sn_mean_2RSD = True_118Sn_120Sn_mean_2SD / True_118Sn_120Sn_mean * 1000
    
    
    True_123Sb_121Sb_mean = np.mean(df['123Sb/121Sb'])
    True_123Sb_121Sb_mean_2SD = np.std(df['123Sb/121Sb']) * 2
    True_123Sb_121Sb_mean_2RSD = True_123Sb_121Sb_mean_2SD / True_123Sb_121Sb_mean * 1000
    
    
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

    # Erster Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['121Sb'], y=df['118Sn'], mode='markers', name='Datenpunkte'))
    fig.add_trace(go.Scatter(x=df['121Sb'], y=trend, mode='lines', name='TrendliSne'))

    fig.update_layout(
        title="Voltages of Sb and Sn of Standards",
        xaxis_title="$^{121}$Sb (V)",
        yaxis_title="$^{118}$Sn (V)",
        legend_title="Legende",
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
    )

    for i in range(len(df)):
        fig.add_annotation(x=df['121Sb'].iloc[i], y=df['118Sn'].iloc[i], text=str(df['ID'].iloc[i]))
    fig.add_annotation(text="R² = "+r2string, showarrow=False, font=dict(size=16))
    fig.add_annotation(text="Intercept = "+str(np.round(reg[1], 5)), showarrow=False, font=dict(size=16))
    fig.add_annotation(text="Slope = "+str(np.round(reg[0], 5)), showarrow=False, font=dict(size=16))

    # Zweiter Plot
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=df['d123Sb'], y=df['d118Sn'], mode='markers', name='Datenpunkte'))

    fig2.update_layout(
        title="d$^{121}$Sb vs d$^{118}$Sn",
        xaxis_title="d$^{121}$Sb",
        yaxis_title="d$^{118}$Sn",
        legend_title="Legende",
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
    )

    for i in range(len(df)):
        fig2.add_annotation(x=df['d123Sb'].iloc[i], y=df['d118Sn'].iloc[i], text=str(df['ID'].iloc[i]))
        
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    
    
    
def plot_Sb_outlier_Blk(file):
    global fullname
    global outfile
    global outfile_results_Sb
    global outfile_corrected_raw_data_Sb
    global outfile_plt_Sb
    global standardinputpfad
    global standardoutputpfad
    
    fullname = os.path.join(standardoutputpfad)
    
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float) 
    dataSbrrentplot = data.copy()
    dataSbrrentplot = dataSbrrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    
    datao = pd.DataFrame({'Time':data['Time']})
    dataofig = datao.copy()
    data.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    cols_sn = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb = ['123Sb']  # Nur 123Sb/121Sb
    data = pd.concat([data, data[cols_sn].div(data['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data = pd.concat([data, data[cols_sb].div(data['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)
    datafig = data.copy()

    cols1 = list(data.columns)
    
    datao['123Sb/121Sb'] = data['123Sb/121Sb']
    datao['123Sb/121Sb'] = datao['123Sb/121Sb'].where(np.abs(stats.zscore(datao['123Sb/121Sb'])) < 2)
    
    fig = go.Figure()
    for c in cols1:
        fig.add_trace(go.Scatter(x=data['Time'], y=data['123Sb/121Sb'], mode='markers', name='Original', marker_color='red'))
        fig.add_trace(go.Scatter(x=datao['Time'], y=datao['123Sb/121Sb'], mode='markers', name='Filtered', marker_color='blue'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='123Sb/121Sb vs Time') #basename
    
    colsfig1 = list(datafig.columns)
    
    dataofig['120Sn/118Sn'] = datafig['120Sn/118Sn']
    dataofig['120Sn/118Sn'] = dataofig['120Sn/118Sn'].where(np.abs(stats.zscore(dataofig['120Sn/118Sn'])) < 2)
    fig2 = go.Figure()
    for c in colsfig1:
        fig2.add_trace(go.Scatter(x=datafig['Time'], y=datafig['120Sn/118Sn'], mode='markers', name='Original', marker_color='red'))
        fig2.add_trace(go.Scatter(x=dataofig['Time'], y=dataofig['120Sn/118Sn'], mode='markers', name='Filtered', marker_color='blue'))
    fig2.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='120Sn/118Sn vs Time') #basename

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.plotly_chart(fig2)


def plot_Sb_outlier_Std(file):
    global fullname
    global outfile
    global outfile_results_Sb
    global outfile_corrected_raw_data_Sb
    global outfile_plt_Sb
    global standardinputpfad
    global standardoutputpfad
    
    fullname = os.path.join(standardoutputpfad)
    
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)

    dataSbrrentplot = data.copy()
    dataSbrrentplot = dataSbrrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    
    datao = pd.DataFrame({'Time':data['Time']})
    dataofig = datao.copy()
    
    data.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    cols_sn = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb = ['123Sb']  # Nur 123Sb/121Sb
    data = pd.concat([data, data[cols_sn].div(data['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data = pd.concat([data, data[cols_sb].div(data['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)
    datafig = data.copy()
    cols1 = list(data.columns)
    
    datao[cols1] = data[cols1]
    
    datao['123Sb/121Sb'] = datao['123Sb/121Sb'].where(np.abs(stats.zscore(datao['123Sb/121Sb'])) < 2)
    
    fig = go.Figure()
    for c in cols1:
        fig.add_trace(go.Scatter(x=data['Time'], y=data['123Sb/121Sb'], mode='markers', name='Original', marker_color='red'))
        fig.add_trace(go.Scatter(x=datao['Time'], y=datao['123Sb/121Sb'], mode='markers', name='Filtered', marker_color='blue'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='123Sb/121Sb vs Time') 
    
    colsfig1 = list(datafig.columns)
    
    dataofig['120Sn/118Sn'] = datafig['120Sn/118Sn']
    dataofig['120Sn/118Sn'] = dataofig['120Sn/118Sn'].where(np.abs(stats.zscore(dataofig['120Sn/118Sn'])) < 2)
    fig2 = go.Figure()
    for c in colsfig1:
        fig2.add_trace(go.Scatter(x=datafig['Time'], y=datafig['120Sn/118Sn'], mode='markers', name='Original', marker_color='red'))
        fig2.add_trace(go.Scatter(x=dataofig['Time'], y=dataofig['120Sn/118Sn'], mode='markers', name='Filtered', marker_color='blue'))
    fig2.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='120Sn/118Sn vs Time')

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.plotly_chart(fig2)


def plot_Sb_outlier_Sample(file):
    global fullname
    global outfile
    global outfile_results_Sb
    global outfile_corrected_raw_data_Sb
    global outfile_plt_Sb
    global standardinputpfad
    global standardoutputpfad
    
    length = len(pd.read_csv(file))
    split = int(length / 3) - 1
    double_split = split * 2
    
    fullname = os.path.join(standardoutputpfad)
    
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=split, index_col=False, dtype=float)
    data2 = pd.read_csv(file, sep='\t', names=col_names, skiprows=6+split, nrows=split, index_col=False, dtype=float)
    data3 = pd.read_csv(file, sep='\t', names=col_names, skiprows=6+double_split, index_col=False, dtype=float)


    data_all = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
    
    
    dataSbrrentplot = data_all.copy()
    dataSbrrentplot = dataSbrrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    dataofig = data.copy()
    data.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    cols_sn = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb = ['123Sb']  # Nur 123Sb/121Sb
    data = pd.concat([data, data[cols_sn].div(data['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data = pd.concat([data, data[cols_sb].div(data['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)
    datafig = data.copy()

    colsa1 = list(data.columns)

    cols2 = list(data2.drop(columns='Time').columns)
    data2o = pd.DataFrame({'Time':data2['Time']})
    
    data2ofig = data2.copy()
    data2.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    cols_sn2 = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb2 = ['123Sb']  # Nur 123Sb/121Sb
    data2 = pd.concat([data2, data2[cols_sn2].div(data2['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data2 = pd.concat([data2, data2[cols_sb2].div(data2['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)
    datafig2 = data2.copy()

    colsa2 = list(data2.columns)

    cols3 = list(data3.drop(columns='Time').columns)
    data3o = pd.DataFrame({'Time':data3['Time']})
    
    data3ofig = data3.copy()
    data3.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis = 1, inplace = True)
    cols_sn3 = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb3 = ['123Sb']  # Nur 123Sb/121Sb
    data3 = pd.concat([data3, data3[cols_sn3].div(data3['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data3 = pd.concat([data3, data3[cols_sb3].div(data3['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)
    datafig2 = data3.copy()

    colsa3 = list(data3.columns)
    
    
    datao['123Sb/121Sb'] = data['123Sb/121Sb']
    datao['123Sb/121Sb'] = datao['123Sb/121Sb'].where(np.abs(stats.zscore(datao['123Sb/121Sb'])) < 2)
    
    data2o['123Sb/121Sb'] = data2['123Sb/121Sb']
    data2o['123Sb/121Sb'] = data2o['123Sb/121Sb'].where(np.abs(stats.zscore(data2o['123Sb/121Sb'])) < 2)
    
    data3o['123Sb/121Sb'] = data3['123Sb/121Sb']
    data3o['123Sb/121Sb'] = data3o['123Sb/121Sb'].where(np.abs(stats.zscore(data3o['123Sb/121Sb'])) < 2)
    
    
    fig = go.Figure()
    for c in colsa1:
        fig.add_trace(go.Scatter(x=data['Time'], y=data['123Sb/121Sb'], mode='markers', name='Original', marker_color='red'))
        fig.add_trace(go.Scatter(x=data2['Time'], y=data2['123Sb/121Sb'], mode='markers', name='Original', marker_color='red'))
        fig.add_trace(go.Scatter(x=data3['Time'], y=data3['123Sb/121Sb'], mode='markers', name='Original', marker_color='red'))
        fig.add_trace(go.Scatter(x=datao['Time'], y=datao['123Sb/121Sb'], mode='markers', name='Filtered', marker_color='green'))
        fig.add_trace(go.Scatter(x=data2o['Time'], y=data2o['123Sb/121Sb'], mode='markers', name='Filtered', marker_color='magenta'))
        fig.add_trace(go.Scatter(x=data3o['Time'], y=data3o['123Sb/121Sb'], mode='markers', name='Filtered', marker_color='blue'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='123Sb/121Sb vs Time') #basename
    
    colsfig1 = list(dataofig.columns)
    
    datao['120Sn/118Sn'] = data['120Sn/118Sn']
    datao['120Sn/118Sn'] = datao['120Sn/118Sn'].where(np.abs(stats.zscore(datao['120Sn/118Sn'])) < 2)
    
    data2o['120Sn/118Sn'] = data2['120Sn/118Sn']
    data2o['120Sn/118Sn'] = data2o['120Sn/118Sn'].where(np.abs(stats.zscore(data2o['120Sn/118Sn'])) < 2)
    
    data3o['120Sn/118Sn'] = data3['120Sn/118Sn']
    data3o['120Sn/118Sn'] = data3o['120Sn/118Sn'].where(np.abs(stats.zscore(data3o['120Sn/118Sn'])) < 2)
    
    fig2 = go.Figure()
    for c in colsfig1:
        fig2.add_trace(go.Scatter(x=data['Time'], y=data['120Sn/118Sn'], mode='markers', name='Original', marker_color='red'))
        fig2.add_trace(go.Scatter(x=data2['Time'], y=data2['120Sn/118Sn'], mode='markers', name='Original', marker_color='red'))
        fig2.add_trace(go.Scatter(x=data3['Time'], y=data3['120Sn/118Sn'], mode='markers', name='Original', marker_color='red'))
        fig2.add_trace(go.Scatter(x=datao['Time'], y=datao['120Sn/118Sn'], mode='markers', name='Filtered', marker_color='blue'))
        fig2.add_trace(go.Scatter(x=data2o['Time'], y=data2o['120Sn/118Sn'], mode='markers', name='Filtered', marker_color='green'))
        fig2.add_trace(go.Scatter(x=data3o['Time'], y=data3o['120Sn/118Sn'], mode='markers', name='Filtered', marker_color='magenta'))
    fig2.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='120Sn/118Sn vs Time') #basename

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.plotly_chart(fig2)


def Voltages_Sb(file):
        global fullname
        global outfile
        global outfile_results_Sb
        global outfile_corrected_raw_data_Sb
        global outfile_plt_Sb

        fullname = os.path.join(standardoutputpfad)
        basename = os.path.basename(file)
        
        pd.options.display.float_format = '{:.8}'.format
        col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
        col_names.remove('Trace for Mass:')
        col_names.insert(0, 'Time')
        data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
        st.write('121Sb/118Sn ratio = ' + str(round(mean(data['121Sb']/data['118Sn']), 2)))
        dataSbrrentplot = data.copy()
        dataSbrrentplot = dataSbrrentplot.set_index('Time')
        fig3 = px.scatter(dataSbrrentplot, x=dataSbrrentplot.index, y=dataSbrrentplot.columns)
        fig3.update_layout(xaxis_title='Time (s)', yaxis_title='Voltage (V)', title=basename, autosize=True)
        st.plotly_chart(fig3, use_container_width=True)