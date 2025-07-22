# -*- coding: utf-8 -*-
"""
This module containts the Plotting for Sn

@author: Andreas Wittke
"""

"""Plot everything. """
import plotly.subplots as sp

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
# infile = WorkspaceVariableInput
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pickle
from statistics import mean
from plotly.subplots import make_subplots


file_path = os.path.join("modules", "variable_data.pkl")
baxternameSn = os.path.join(outfile_results_Sn, 'Sn_delta_Baxter.csv')
# Variable mit pickle laden
with open(file_path, 'rb') as file:
    standardoutputpfad = pickle.load(file)
    


file_path2 = os.path.join("modules", "variable_data_input.pkl")
# Variable mit pickle laden
with open(file_path2, 'rb') as file2:
    standardinputpfad = pickle.load(file2)

def load_files(path, extension):
    return sorted([f for f in os.listdir(path) if f.endswith(extension)])

def plot_Sn_outlier_Blk(file):
    global fullname
    global outfile_results_Sn
    global outfile_corrected_raw_data_Sn
    global outfile_plt_Sn
    global standardinputpfad
    global standardoutputpfad
    
    fullname = os.path.join(standardoutputpfad)
    
    pd.options.display.float_format = '{:.8}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=30, index_col=False, dtype=float)
    datacurrentplot = data.copy()
    datacurrentplot = datacurrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)


    datao = pd.DataFrame({'Time':data['Time']})
    dataofig = datao.copy()
    
    
    data = pd.concat([data, data[cols].div(data['120Sn'], axis=0).add_suffix('/120Sn')], axis=1)
    data = pd.concat([data, data[cols].div(data['116Sn'], axis=0).add_suffix('/116Sn')], axis=1)
    data['117Sn/119Sn'] = data['117Sn'] / data['119Sn']
    data['123Sb/121Sb'] = data['123Sb'] / data['121Sb']
    
    
    
    data.drop(['120Sn/120Sn', '116Sn/116Sn', '117Sn/116Sn'], axis = 1, inplace = True)
    datafig = data.copy()
    
    cols1 = list(data.columns)

    datao[cols1] = data[cols1]
    del cols1[0:10]
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)

    basename = os.path.basename(file)
    # # Aktualisieren Sie das Layout
    fig = sp.make_subplots(rows=5, cols=4, subplot_titles=cols1)
    for idx, c in enumerate(cols1):
        fig.add_trace(go.Scatter(x=data['Time'], y=data[c], mode='markers', name='Original', marker_color='red'), row=idx//4+1, col=idx%4+1)
        fig.add_trace(go.Scatter(x=datao['Time'], y=datao[c], mode='markers', name='Filtered', marker_color='blue'), row=idx//4+1, col=idx%4+1)

    fig.update_layout(height=1000, width=2000, title_text=basename)

    # Zeigen Sie den Plot an
    st.plotly_chart(fig, use_container_width=True)
        
##########################################################################    
def plot_Sn_outlier_Std(file):
    global fullname
    global outfile_results_Sn
    global outfile_corrected_raw_data_Sn
    global outfile_plt_Sn

    
    fullname = os.path.join(standardoutputpfad)
    basename = os.path.basename(file)
  
    pd.options.display.float_format = '{:.8f}'.format
 
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')

    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=90, index_col=False, dtype=float)
    datacurrentplot = data.copy()
    datacurrentplot = datacurrentplot.set_index('Time')
    
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
    
    fig = sp.make_subplots(rows=5, cols=4, subplot_titles=cols1)

    for idx, c in enumerate(cols1):
        fig.add_trace(go.Scatter(x=data['Time'], y=data[c], mode='markers', name='Original', marker_color='red'), row=idx//4+1, col=idx%4+1)
        fig.add_trace(go.Scatter(x=datao['Time'], y=datao[c], mode='markers', name='Filtered', marker_color='blue'), row=idx//4+1, col=idx%4+1)

    fig.update_layout(height=1000, width=2000, title_text=basename)

    
    datalog116Sn120Sn = np.log(datao['116Sn/120Sn'])
    datalogSb = np.log(datao['123Sb/121Sb'])
    datalog117Sn120Sn = np.log(datao['117Sn/120Sn'])
    datalog118Sn120Sn = np.log(datao['118Sn/120Sn'])
    datalog119Sn120Sn = np.log(datao['119Sn/120Sn'])
    datalog122Sn120Sn = np.log(datao['122Sn/120Sn'])
    datalog124Sn120Sn = np.log(datao['124Sn/120Sn'])
    datalog124Sn116Sn = np.log(datao['124Sn/116Sn'])
    datalog122Sn116Sn = np.log(datao['122Sn/116Sn'])
    datalog117Sn119Sn = np.log(datao['117Sn/119Sn'])
    
    datalog = pd.DataFrame()
    datalog['ln(116Sn/120Sn)'] = datalog116Sn120Sn
    datalog['ln(123Sb/121Sb)'] = datalogSb
    datalog['ln(117Sn/120Sn)'] = datalog117Sn120Sn
    datalog['ln(118Sn/120Sn)'] = datalog118Sn120Sn
    datalog['ln(119Sn/120Sn)'] = datalog119Sn120Sn
    datalog['ln(122Sn/120Sn)'] = datalog122Sn120Sn
    datalog['ln(124Sn/120Sn)'] = datalog124Sn120Sn
    datalog['ln(124Sn/116Sn)'] = datalog124Sn116Sn
    datalog['ln(122Sn/116Sn)'] = datalog122Sn116Sn
    datalog['ln(117Sn/119Sn)'] = datalog117Sn119Sn
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=datalogSb, y=datalog116Sn120Sn, mode='markers', name='Original', marker_color='red'))

    fig2.update_xaxes(title_text="ln($^{123}$Sb/$^{121}$Sb)")
    fig2.update_yaxes(title_text="ln($^{116}$Sn/$^{120}$Sn)")
    
    fig2.update_layout(height=600, width=1000, title_text=basename)

    colso = list(datalog.columns) 

    fig4 = sp.make_subplots(rows=4, cols=3, subplot_titles=colso)

    for idx, c in enumerate(colso):
        fig4.add_trace(go.Scatter(x=datalog['ln(124Sn/120Sn)'], y=datalog[c], mode='markers', name='Original', marker_color='blue'), row=idx//3+1, col=idx%3+1)

    fig4.update_layout(height=1200, width=2000, title_text=basename)
    
    tabs = st.tabs(["Outlier Detection", "lnSn vs lnSb", "ln_xx_Sn vs ln_124/120Sn"])
    if 'main_tab' not in st.session_state:
        st.session_state.main_tab = 0
    st.session_state.main_tab = tabs.index
    with tabs[0]:
        st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        st.plotly_chart(fig2, use_container_width=True)
    with tabs[2]:
        st.plotly_chart(fig4, use_container_width=True)
        
 
    
    
    fig5, ax5 = plt.subplots(figsize=(20,15))
    sb.scatterplot(data=datacurrentplot, s = 150, ax = ax5)
    ax5.set_xlim(-1,)
    ax5.set_xlabel('Time (s)',size=20)
    ax5.set_ylabel('Voltage (V)',size=20)
   
def plot_Sn_outlier_Sample(file): 
    
    global fullname
    global outfile_results_Sn
    global outfile_corrected_raw_data_Sn
    global outfile_plt_Sn

    
    fullname = os.path.join(standardoutputpfad)
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
    datacurrentplot = data_all.copy()
    datacurrentplot = datacurrentplot.set_index('Time')

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
    dataofig = data.copy()
    
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
    
    datao[colsa1] = datao[colsa1].where(np.abs(stats.zscore(datao[colsa1])) < 1.5)
    data2o[colsa2] = data2o[colsa2].where(np.abs(stats.zscore(data2o[colsa2])) < 1.5)
    data3o[colsa3] = data3o[colsa3].where(np.abs(stats.zscore(data3o[colsa3])) < 1.5)
    
    
    datalog116Sn120Sn = np.log(datao['116Sn/120Sn'])
    datalogSb = np.log(datao['123Sb/121Sb'])
    datalog117Sn120Sn = np.log(datao['117Sn/120Sn'])
    datalog118Sn120Sn = np.log(datao['118Sn/120Sn'])
    datalog119Sn120Sn = np.log(datao['119Sn/120Sn'])
    datalog122Sn120Sn = np.log(datao['122Sn/120Sn'])
    datalog124Sn120Sn = np.log(datao['124Sn/120Sn'])
    datalog124Sn116Sn = np.log(datao['124Sn/116Sn'])
    datalog122Sn116Sn = np.log(datao['122Sn/116Sn'])
    datalog117Sn119Sn = np.log(datao['117Sn/119Sn'])
    
    datalog = pd.DataFrame()
    datalog['ln(116Sn/120Sn)'] = datalog116Sn120Sn
    datalog['ln(123Sb/121Sb)'] = datalogSb
    datalog['ln(117Sn/120Sn)'] = datalog117Sn120Sn
    datalog['ln(118Sn/120Sn)'] = datalog118Sn120Sn
    datalog['ln(119Sn/120Sn)'] = datalog119Sn120Sn
    datalog['ln(122Sn/120Sn)'] = datalog122Sn120Sn
    datalog['ln(124Sn/120Sn)'] = datalog124Sn120Sn
    datalog['ln(124Sn/116Sn)'] = datalog124Sn116Sn
    datalog['ln(122Sn/116Sn)'] = datalog122Sn116Sn
    datalog['ln(117Sn/119Sn)'] = datalog117Sn119Sn

    
    data2log116Sn120Sn = np.log(data2o['116Sn/120Sn'])
    data2logSb = np.log(data2o['123Sb/121Sb'])
    data2log117Sn120Sn = np.log(data2o['117Sn/120Sn'])
    data2log118Sn120Sn = np.log(data2o['118Sn/120Sn'])
    data2log119Sn120Sn = np.log(data2o['119Sn/120Sn'])
    data2log122Sn120Sn = np.log(data2o['122Sn/120Sn'])
    data2log124Sn120Sn = np.log(data2o['124Sn/120Sn'])
    data2log124Sn116Sn = np.log(data2o['124Sn/116Sn'])
    data2log122Sn116Sn = np.log(data2o['122Sn/116Sn'])
    data2log117Sn119Sn = np.log(data2o['117Sn/119Sn'])
    
    data2log = pd.DataFrame()
    data2log['ln(116Sn/120Sn)'] = data2log116Sn120Sn
    data2log['ln(123Sb/121Sb)'] = data2logSb
    data2log['ln(117Sn/120Sn)'] = data2log117Sn120Sn
    data2log['ln(118Sn/120Sn)'] = data2log118Sn120Sn
    data2log['ln(119Sn/120Sn)'] = data2log119Sn120Sn
    data2log['ln(122Sn/120Sn)'] = data2log122Sn120Sn
    data2log['ln(124Sn/120Sn)'] = data2log124Sn120Sn
    data2log['ln(124Sn/116Sn)'] = data2log124Sn116Sn
    data2log['ln(122Sn/116Sn)'] = data2log122Sn116Sn
    data2log['ln(117Sn/119Sn)'] = data2log117Sn119Sn
    
  
    data3log116Sn120Sn = np.log(data3o['116Sn/120Sn'])
    data3logSb = np.log(data3o['123Sb/121Sb'])
    data3log117Sn120Sn = np.log(data3o['117Sn/120Sn'])
    data3log118Sn120Sn = np.log(data3o['118Sn/120Sn'])
    data3log119Sn120Sn = np.log(data3o['119Sn/120Sn'])
    data3log122Sn120Sn = np.log(data3o['122Sn/120Sn'])
    data3log124Sn120Sn = np.log(data3o['124Sn/120Sn'])
    data3log124Sn116Sn = np.log(data3o['124Sn/116Sn'])
    data3log122Sn116Sn = np.log(data3o['122Sn/116Sn'])
    data3log117Sn119Sn = np.log(data3o['117Sn/119Sn'])
    
    data3log = pd.DataFrame()
    data3log['ln(116Sn/120Sn)'] = data3log116Sn120Sn
    data3log['ln(123Sb/121Sb)'] = data3logSb
    data3log['ln(117Sn/120Sn)'] = data3log117Sn120Sn
    data3log['ln(118Sn/120Sn)'] = data3log118Sn120Sn
    data3log['ln(119Sn/120Sn)'] = data3log119Sn120Sn
    data3log['ln(122Sn/120Sn)'] = data3log122Sn120Sn
    data3log['ln(124Sn/120Sn)'] = data3log124Sn120Sn
    data3log['ln(124Sn/116Sn)'] = data3log124Sn116Sn
    data3log['ln(122Sn/116Sn)'] = data3log122Sn116Sn
    data3log['ln(117Sn/119Sn)'] = data3log117Sn119Sn

     
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
    
    
    fig = sp.make_subplots(rows=5, cols=4, subplot_titles=colsa1)

    for idx, c in enumerate(colsa1):
        fig.add_trace(go.Scatter(x=data['Time'], y=data[c], mode='markers', name='Original', marker_color='red'), row=(idx//4)+1, col=(idx%4)+1)
        fig.add_trace(go.Scatter(x=datao['Time'], y=datao[c], mode='markers', name='Filtered', marker_color='blue'), row=(idx//4)+1, col=(idx%4)+1)

    for idx, c in enumerate(colsa2):
        fig.add_trace(go.Scatter(x=data2['Time'], y=data2[c], mode='markers', name='Original', marker_color='red'), row=(idx//4)+1, col=(idx%4)+1)
        fig.add_trace(go.Scatter(x=data2o['Time'], y=data2o[c], mode='markers', name='Filtered', marker_color='green'), row=(idx//4)+1, col=(idx%4)+1)

    for idx, c in enumerate(colsa3):
        fig.add_trace(go.Scatter(x=data3['Time'], y=data3[c], mode='markers', name='Original', marker_color='red'), row=(idx//4)+1, col=(idx%4)+1)
        fig.add_trace(go.Scatter(x=data3o['Time'], y=data3o[c], mode='markers', name='Filtered', marker_color='darkviolet'), row=(idx//4)+1, col=(idx%4)+1)


    fig.update_layout(height=1200, width=2000, title_text=basename)

    dataofig = data.copy()
    colsfig1 = list(dataofig.columns)    
    colsfig1 = list(dataofig.columns)
    

    fig2 = go.Figure()
   
    # Füge die Datenpunkte und Linien hinzu
    fig2.add_trace(go.Scatter(
        x=datalogSb, y=datalogSn, mode='markers', 
        marker=dict(color='red', symbol='circle'), 
        name='1st Block'
    ))

    fig2.add_trace(go.Scatter(
        x=datalog2Sb, y=datalog2Sn, mode='markers', 
        marker=dict(color='blue', symbol='circle'), 
        name='2nd Block'
    ))

    fig2.add_trace(go.Scatter(
        x=datalog3Sb, y=datalog3Sn, mode='markers', 
        marker=dict(color='green', symbol='circle'), 
        name='3rd Block'
    ))

    fig2.add_trace(go.Scatter(
        x=datalogSb_all, y=trend, mode='lines', 
        line=dict(color='black', dash='dash'), 
        name='Trend'
    ))

    fig2.add_trace(go.Scatter(
        x=datalogSb_all, y=trend - std, mode='lines', 
        line=dict(color='cyan', dash='dash'), 
        name='Trend - STD'
    ))

    fig2.add_trace(go.Scatter(
        x=datalogSb_all, y=trend + std, mode='lines', 
        line=dict(color='cyan', dash='dash'), 
        name='Trend + STD'
    ))

    # Setze die Achsenbeschriftungen und den Titel
    fig2.update_layout(
        title=basename,
        xaxis_title='ln($^{123}$Sb/$^{121}$Sb)',
        yaxis_title='ln($^{116}$Sn/$^{120}$Sn)',
        font=dict(size=14),
        legend_title="Blocks",
        margin=dict(l=50, r=50, t=50, b=50)
    )
        
    
    colso = list(datalog.columns) 

    
    basename = 'xx vs ln(124Sn/120Sn)'

    # Erstelle eine Plotly-Figur mit Subplots
    fig8 = make_subplots(rows=4, cols=3, subplot_titles=colso[:12], vertical_spacing=0.1)  # Erstelle 4x3 Raster für Subplots

    # Füge Scatter-Traces hinzu
    for idx, c in enumerate(colso):
        row = idx // 3 + 1  # Berechnet die Zeile (1-indexiert)
        col = idx % 3 + 1   # Berechnet die Spalte (1-indexiert)
        
        fig8.add_trace(
            go.Scatter(
                x=datalog['ln(124Sn/120Sn)'],
                y=datalog[c],
                mode='markers',
                name=f'{c[:2]}{c[2:]}'
            ),
            row=row, col=col
        )

    # Update der Layout-Einstellungen
    fig8.update_layout(
        title=basename,
        title_x=0.5,
        xaxis_title='ln(124Sn/120Sn)',  # HTML-ähnliche Syntax für Hochstellungen
        showlegend=False,
        height=1000,  # Höhe des gesamten Plots (in Pixeln)
        width=1200
    )

    # Aktualisiere die Achsentitel für alle Subplots
    for i in range(1, 13):
        fig8.update_xaxes(title_text='ln(124Sn/120Sn)', row=(i-1)//3 + 1, col=(i-1)%3 + 1)

    # Entferne leere Subplots, falls vorhanden (Plotly entfernt automatisch nicht verwendete Subplots)
    # Hier entfernen wir den 10., 11., und 12. Plot, indem wir keine Daten an sie anhängen.
    colso2 = list(data2log.columns) 
    # Erstelle eine Plotly-Figur mit Subplots
    fig9 = make_subplots(rows=4, cols=3, subplot_titles=colso2[:12], vertical_spacing=0.1)  # Erstelle 4x3 Raster für Subplots

    # Füge Scatter-Traces hinzu
    for idx, c in enumerate(colso2):
        row = idx // 3 + 1  # Berechnet die Zeile (1-indexiert)
        col = idx % 3 + 1   # Berechnet die Spalte (1-indexiert)
        
        fig9.add_trace(
            go.Scatter(
                x=data2log['ln(124Sn/120Sn)'],
                y=data2log[c],
                mode='markers',
                name=f'{c[:2]}{c[2:]}'
            ),
            row=row, col=col
        )

    # Update der Layout-Einstellungen
    fig9.update_layout(
        title=basename,
        title_x=0.5,
        xaxis_title='ln(124Sn/120Sn)',  # HTML-ähnliche Syntax für Hochstellungen
        showlegend=False,
        height=1000,  # Höhe des gesamten Plots (in Pixeln)
        width=1200
    )

    # Aktualisiere die Achsentitel für alle Subplots
    for i in range(1, 13):
        fig9.update_xaxes(title_text='ln(124Sn/120Sn)', row=(i-1)//3 + 1, col=(i-1)%3 + 1)
    
    colso3 = list(data3log.columns) 

    
    # Erstelle eine Plotly-Figur mit Subplots
    fig10 = make_subplots(rows=4, cols=3, subplot_titles=colso3[:12], vertical_spacing=0.1)  # Erstelle 4x3 Raster für Subplots

    # Füge Scatter-Traces hinzu
    for idx, c in enumerate(colso3):
        row = idx // 3 + 1  # Berechnet die Zeile (1-indexiert)
        col = idx % 3 + 1   # Berechnet die Spalte (1-indexiert)
        
        fig10.add_trace(
            go.Scatter(
                x=data3log['ln(124Sn/120Sn)'],
                y=data3log[c],
                mode='markers',
                name=f'{c[:2]}{c[2:]}'
            ),
            row=row, col=col
        )

    # Update der Layout-Einstellungen
    fig10.update_layout(
        title=basename,
        title_x=0.5,
        xaxis_title='ln(124Sn/120Sn)',  # HTML-ähnliche Syntax für Hochstellungen
        showlegend=False,
        height=1000,  # Höhe des gesamten Plots (in Pixeln)
        width=1200
    )

    # Aktualisiere die Achsentitel für alle Subplots
    for i in range(1, 13):
        fig10.update_xaxes(title_text='ln(124Sn/120Sn)', row=(i-1)//3 + 1, col=(i-1)%3 + 1)

    fig7, ax7 = plt.subplots(figsize=(20,15))
    sb.scatterplot(data=datacurrentplot, s = 150, ax = ax7)
    ax7.set_xlim(-1,)
    ax7.set_xlabel('Time (s)',size=20)
    ax7.set_ylabel('Voltage (V)',size=20)


    tabs = st.tabs(["Outlier Detection", "lnSn vs. lnSb", "ln-Block1", "ln-Block2", "ln-Block3"])
    if 'main_tab' not in st.session_state:
        st.session_state.main_tab = 0
    st.session_state.main_tab = tabs.index
    with tabs[0]:
        st.plotly_chart(fig)
    with tabs[1]:
        st.plotly_chart(fig2)
    with tabs[2]:
        st.plotly_chart(fig8, use_container_width=True)   
    with tabs[3]:
        st.plotly_chart(fig9, use_container_width=True)   
    with tabs[4]:
        st.plotly_chart(fig10, use_container_width=True)   


def load_data_PlotSn():
    # Datei mit Pickle laden
    file_path = os.path.join("modules", "variable_data.pkl")
    with open(file_path, 'rb') as file:
        data_directory = pickle.load(file)  # Lade den Pfad
    # Jetzt lade die CSV-Datei mit dem geladenen Pfad
    csv_path_Sn_export = os.path.join(data_directory, "Sn_export.csv")
    Sn_export = pd.read_csv(csv_path_Sn_export, sep='\t')  # Passe den Separator an
    return Sn_export

    
def plot_Sn_Masses(Sn_export):
    data = Sn_export
    data['Name'] = data['Filename']
    data = data.set_index('Filename')

    # Daten vorbereiten
    datalog = pd.DataFrame(index=data.index)
    datalog['log116Sn/120Sn'] = np.log(data['116Sn'] / data['120Sn'])
    datalog['log123Sb/121Sb'] = np.log(data['123Sb'] / data['121Sb'])
    datalog['standard'] = datalog.index.str.contains('_Nis0|Nis1')
    datalog['Name'] = data['Name']

    # Filter für Standards
    datalogStandard = datalog[datalog['standard'] == True]
    datalogStandard.insert(4, 'ID', range(1, 1 + len(datalogStandard)))
    datalogStandard['ID'] = datalogStandard['ID'].round(decimals=0).astype(object)

    # Regressionsanalyse
    slope, intercept, r, p, se = stats.linregress(datalogStandard['log123Sb/121Sb'], datalogStandard['log116Sn/120Sn'])
    rsquared = r**2   

    # Plotly Scatter-Plot
    fig3 = go.Figure()

    # Scatter-Punkte hinzufügen
    fig3.add_trace(go.Scatter(
        x=datalogStandard['log123Sb/121Sb'],
        y=datalogStandard['log116Sn/120Sn'],
        mode='markers+text',
        marker=dict(size=10, color='black'),
        text=datalogStandard['ID'],
        textposition='top center',
        name='Datenpunkte'
    ))

    # Regressionlinie hinzufügen
    trend_x = np.linspace(datalogStandard['log123Sb/121Sb'].min(), datalogStandard['log123Sb/121Sb'].max(), 100)
    trend_y = intercept + slope * trend_x

    fig3.add_trace(go.Scatter(
        x=trend_x,
        y=trend_y,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Lineare Regression'
    ))

    # Layout anpassen
    fig3.update_layout(
        title='ln($^{123}$Sb/$^{121}$Sb) vs ln($^{116}$Sn/$^{120}$Sn) of Standards',
        xaxis_title='ln($^{123}$Sb/$^{121}$Sb)',
        yaxis_title='ln($^{116}$Sn/$^{120}$Sn)',
        height=800,
        showlegend=True,
    )

    # R², Intercept und Slope als Text hinzufügen
    fig3.add_annotation(
        text=f"R² = {round(rsquared, 4)}<br>Intercept = {round(intercept, 4)}<br>Slope = {round(slope, 4)}",
        xref="paper", yref="paper",
         x=0.95, y=0.95,
        showarrow=False,
        font=dict(size=12)
    )

    return fig3
  
    
def Voltages_Sn(file):
        global fullname
        global outfile
        global outfile_results_Sn
        global outfile_corrected_raw_data_Sn
        global outfile_plt_Sn

        fullname = os.path.join(standardoutputpfad)
        basename = os.path.basename(file)
        
        pd.options.display.float_format = '{:.8}'.format
        col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
        col_names.remove('Trace for Mass:')
        col_names.insert(0, 'Time')
        data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
        st.write('120Sn/121Sb ratio = ' + str(round(mean(data['120Sn']/data['121Sb']), 2)))
        datacurrentplot = data.copy()
        datacurrentplot = datacurrentplot.set_index('Time')
        fig3 = px.scatter(datacurrentplot, x=datacurrentplot.index, y=datacurrentplot.columns)
        fig3.update_layout(xaxis_title='Time (s)', yaxis_title='Voltage (V)', title=basename, autosize=True)
        st.plotly_chart(fig3, use_container_width=True)
        
        
def PlotSnBaxterMasses():
    global fullnameSn
    global resultnameSn
    global plotSn
    global baxternameSn
    
    Sn_delta = pd.read_csv(baxternameSn, sep = '\t')
    
    fig, ax = plt.subplots(nrows =3, ncols = 2, figsize = (40, 30), label='Inline label', sharex = False)
    fig.suptitle('\u03B4 Plot of all Sn samples')
    fig.tight_layout()
    plt.rcParams.update({'font.size': 14})
    plt.subplots_adjust(left=0.1,
                bottom=0.1, 
                right=0.9, 
                top=0.9, 
                wspace=0.4, 
                hspace=0.4)
    
    # Plot d124Sn/120Sn vs d116Sn/120Sn
    ax[0, 0].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], color='r', marker='o')
    ax[0, 0].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.5'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    reg_ax0 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 1)
    predict_ax0 = np.poly1d(reg_ax0) # Slope and intercept
    trend_ax0 = np.polyval(reg_ax0, Sn_delta['d116Sn/120Sn'])
    std_ax0 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax0 = np.round(r2_score(Sn_delta['d124Sn/120Sn'], predict_ax0(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax0 = str(r2_ax0)
    
    ax[0, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax0, 'k--')
    ax[0, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax0 - std_ax0, 'c--')
    ax[0, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax0 + std_ax0, 'c--')
    ax[0, 0].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[0, 0].set_ylabel('\u03B4$^{124}$Sn/$^{120}$Sn')
    ax[0, 0].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax0)
    
    ax[0, 0].grid(True, which='both')
    ax[0, 0].axhline(y=0, color='k')
    ax[0, 0].axvline(x=0, color='k')
    ax[0, 0].set_xticks(ax[0, 0].get_xticks()[::1])
    ax[0, 0].legend(loc=1)
    
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d116Sn/120Sn']*1.02, Sn_delta['d124Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[0, 0])
    ax[0, 0].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[0, 0].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    
    # Plot d122Sn/120Sn vs d116Sn/120Sn
    ax[0, 1].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], color='r', marker='o')
    ax[0, 1].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.6'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    reg_ax1 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], 1)
    predict_ax1 = np.poly1d(reg_ax1) # Slope and intercept
    trend_ax1 = np.polyval(reg_ax1, Sn_delta['d116Sn/120Sn'])
    std_ax1 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax1 = np.round(r2_score(Sn_delta['d122Sn/120Sn'], predict_ax1(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax1 = str(r2_ax1)
    
    ax[0, 1].plot(Sn_delta['d116Sn/120Sn'], trend_ax1, 'k--')
    ax[0, 1].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 - std_ax1, 'c--')
    ax[0, 1].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 + std_ax1, 'c--')
    ax[0, 1].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[0, 1].set_ylabel('\u03B4$^{122}$Sn/$^{120}$Sn')
    ax[0, 1].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax1)
    
    ax[0, 1].grid(True, which='both')
    ax[0, 1].axhline(y=0, color='k')
    ax[0, 1].axvline(x=0, color='k')
    ax[0, 1].set_xticks(ax[0, 0].get_xticks()[::1])
    ax[0, 1].legend(loc=1)
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d116Sn/120Sn']*1.02, Sn_delta['d122Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[0, 1])
    
    ax[0, 1].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[0, 1].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    
    # Plot d124Sn/120Sn vs d118Sn/120Sn
    ax[1, 0].scatter(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], color='r', marker='o')
    ax[1, 0].errorbar(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], xerr=Sn_delta['2SD.2'], yerr=Sn_delta['2SD.5'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    reg_ax2 = np.polyfit(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 1)
    predict_ax2 = np.poly1d(reg_ax2) # Slope and intercept
    trend_ax2 = np.polyval(reg_ax2, Sn_delta['d118Sn/120Sn'])
    std_ax2 = Sn_delta['d118Sn/120Sn'].std() # Standard deviation
    r2_ax2 = np.round(r2_score(Sn_delta['d124Sn/120Sn'], predict_ax2(Sn_delta['d118Sn/120Sn'])), 5) #R-squared
    r2string_ax2 = str(r2_ax2)
    
    ax[1, 0].plot(Sn_delta['d118Sn/120Sn'], trend_ax2, 'k--')
    ax[1, 0].plot(Sn_delta['d118Sn/120Sn'], trend_ax2 - std_ax2, 'c--')
    ax[1, 0].plot(Sn_delta['d118Sn/120Sn'], trend_ax2 + std_ax2, 'c--')
    ax[1, 0].set_xlabel('\u03B4$^{118}$Sn/$^{120}$Sn')
    ax[1, 0].set_ylabel('\u03B4$^{124}$Sn/$^{120}$Sn')
    ax[1, 0].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax2)
    
    ax[1, 0].grid(True, which='both')
    ax[1, 0].axhline(y=0, color='k')
    ax[1, 0].axvline(x=0, color='k')
    ax[1, 0].set_xticks(ax[1, 0].get_xticks()[::1])
    ax[1, 0].legend(loc=1)
    ax[1, 0].set_xlim(-0.4, 0.1)
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d118Sn/120Sn']*1.02, Sn_delta['d124Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[1, 0])
    ax[1, 0].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[1, 0].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    # Plot d124Sn/120Sn vs d122Sn/120Sn
    ax[1, 1].scatter(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], color='r', marker='o')
    ax[1, 1].errorbar(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], xerr=Sn_delta['2SD.2'], yerr=Sn_delta['2SD.5'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    
    reg_ax1 = np.polyfit(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 1)
    predict_ax1 = np.poly1d(reg_ax1) # Slope and intercept
    trend_ax1 = np.polyval(reg_ax1, Sn_delta['d122Sn/120Sn'])
    std_ax1 = Sn_delta['d122Sn/120Sn'].std() # Standard deviation
    r2_ax1 = np.round(r2_score(Sn_delta['d124Sn/120Sn'], predict_ax1(Sn_delta['d122Sn/120Sn'])), 5) #R-squared
    r2string_ax1 = str(r2_ax1)
    
    ax[1, 1].plot(Sn_delta['d122Sn/120Sn'], trend_ax1, 'k--')
    ax[1, 1].plot(Sn_delta['d122Sn/120Sn'], trend_ax1 - std_ax1, 'c--')
    ax[1, 1].plot(Sn_delta['d122Sn/120Sn'], trend_ax1 + std_ax1, 'c--')
    ax[1, 1].set_xlabel('\u03B4$^{122}$Sn/$^{120}$Sn')
    ax[1, 1].set_ylabel('\u03B4$^{124}$Sn/$^{120}$Sn')
    ax[1, 1].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax1)
    
    ax[1, 1].grid(True, which='both')
    ax[1, 1].axhline(y=0, color='k')
    ax[1, 1].axvline(x=0, color='k')
    ax[1, 1].set_xlim(-0.1, 0.4)
    ax[1, 1].set_xticks(ax[1, 1].get_xticks()[::1])
    ax[1, 1].legend(loc=1)
    
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d122Sn/120Sn']*1.02, Sn_delta['d124Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[1, 1])
    
    
    ax[1, 1].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[1, 1].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    
    # Plot d118Sn/120Sn and d117Sn/119Sn vs d116Sn/120Sn
    ax[2, 0].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d118Sn/120Sn'], color='r', marker='o')
    ax[2, 0].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d118Sn/120Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.2'], linestyle="none", color = 'r', zorder = 0,  capsize=4)
    ax[2, 0].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d117Sn/119Sn'], color='b', marker='o')
    ax[2, 0].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d117Sn/119Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.8'], linestyle="none", color = 'b', zorder = 0,  capsize=4)
    
    
    reg_ax1 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d118Sn/120Sn'], 1)
    predict_ax1 = np.poly1d(reg_ax1) # Slope and intercept
    trend_ax1 = np.polyval(reg_ax1, Sn_delta['d116Sn/120Sn'])
    std_ax1 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax1 = np.round(r2_score(Sn_delta['d118Sn/120Sn'], predict_ax1(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax1 = str(r2_ax1)
    
    reg_ax2 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d117Sn/119Sn'], 1)
    predict_ax2 = np.poly1d(reg_ax2) # Slope and intercept
    trend_ax2 = np.polyval(reg_ax2, Sn_delta['d116Sn/120Sn'])
    std_ax2 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax2 = np.round(r2_score(Sn_delta['d117Sn/119Sn'], predict_ax2(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax2 = str(r2_ax2)
    
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax1, 'k--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 - std_ax1, 'c--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 + std_ax1, 'c--')
    ax[2, 0].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[2, 0].set_ylabel('\u03B4$^{118}$Sn/$^{120}$Sn / \u03B4$^{117}$Sn/$^{119}$Sn')
    ax[2, 0].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax1)
    
    ax[2, 0].grid(True, which='both')
    ax[2, 0].axhline(y=0, color='k')
    ax[2, 0].axvline(x=0, color='k')
    ax[2, 0].set_xticks(ax[2, 0].get_xticks()[::1])
    ax[2, 0].legend(loc=1)
    
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d116Sn/120Sn']*1.02, Sn_delta['d118Sn/120Sn']-0.1, Sn_delta['Lab Nummer'], ax[2, 0])
    
    
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax2, 'k--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax2 - std_ax2, 'c--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax2 + std_ax2, 'c--')
    ax[2, 0].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[2, 0].plot([], [], ' ', label='R$^{2}$ of d117Sn/119Sn = '+r2string_ax2)
    
    ax[2, 0].set_ylim(-0.4, 0.1)
    
    ax[2, 0].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[2, 0].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    ax[2,1].set_visible(False)
    
    fig.savefig(outfile_plt_Sn + "PlotSnMasses.pdf", dpi=200)
    
    plt.close()