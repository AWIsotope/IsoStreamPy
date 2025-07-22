# -*- coding: utf-8 -*-
"""
This module containts the plotting for Cu

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
from modules.config import WorkspaceVariableInput, outfile_plt_Cu, outfile_corrected_raw_data_Cu, outfile_results_Cu, outfile_plt_Cu_Std #outfile
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
save_dStandards = os.path.join(outfile_results_Cu, 'Delta of Cu Standards.csv') 
save_dStandardsExcel = os.path.join(outfile_results_Cu, 'Delta of Cu Standards.xlsx')

def plot_Cu_blk(file): #False):
    global fullname
    global outfile
    global outfile_results_Cu
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu

    fullname = os.path.join(outfile_results_Cu, 'Cu_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=30, index_col=False, dtype=float)

    datacurrentplot = data.copy()
    datacurrentplot = datacurrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data = pd.concat([data, data[cols].div(data['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data = pd.concat([data, data[cols].div(data['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)


    cols1 = list(data.columns)
    datao[cols1] = data[cols1]
    del cols1[0:8] 
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
    sb.scatterplot(data=datacurrentplot, s = 150, ax = ax2)
    ax2.set_xlim(-1,)
    ax2.set_xlabel('Time (s)',size=20)
    ax2.set_ylabel('Voltage (V)',size=20)
    st.pyplot(fig)
    st.pyplot(fig2)
    
    
    
def plot_Cu_std(file):
    global fullname
    global outfile
    global outfile_results_Cu
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu

    fullname = os.path.join(outfile_results_Cu, 'Cu_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
  
    pd.options.display.float_format = '{:.8f}'.format
 
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
    datacurrentplot = data.copy()
    datacurrentplot = datacurrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data = pd.concat([data, data[cols].div(data['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data = pd.concat([data, data[cols].div(data['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    cols1 = list(data.columns)
    
    datao[cols1] = data[cols1]
    del cols1[0:8] 
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)
    
    

    datalogNi = np.log(datao['62Ni'] / datao['60Ni'])
    datalogCu = np.log(datao['65Cu'] / datao['63Cu'])
    
 
    # Erstelle DataFrames mit den Daten
    df21 = pd.DataFrame({
        "ln(62Ni/60Ni)": datalogNi,
        "ln(65Cu/63Cu)": datalogCu,
        "Color": ['1. Block'] * len(datalogNi)
    })


    # Füge die DataFrames zusammen
  
# Erstelle einen Scatter-Plot mit Plotly Express
    fig222 = px.scatter(df21, y="ln(62Ni/60Ni)", x="ln(65Cu/63Cu)", color="Color")

        
    st.plotly_chart(fig222, use_container_width=True)
    
##################################################
def plot_Cu_sample(file): #False):
    
    global fullname
    global outfile
    global outfile_results_Cu 
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu

    print(outfile_results_Cu)
    fullname = os.path.join(outfile_results_Cu, 'Cu_export.csv')
    entries = Path(infile + '/Blk')
    plot_name = Path(entries).stem
    basename = os.path.basename(file)
    corrected_fullname = os.path.join(outfile_results_Cu, 'Cu_corrected.csv')
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
    datacurrentplot = data_all.copy()
    datacurrentplot = datacurrentplot.set_index('Time')
    
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

    
    datalogNi = np.log(datao['62Ni'] / datao['60Ni'])
    datalogCu = np.log(datao['65Cu'] / datao['63Cu'])
    datalogNi2 = np.log(data2o['62Ni'] / data2o['60Ni'])
    datalogCu2 = np.log(data2o['65Cu'] / data2o['63Cu'])
    datalogNi3 = np.log(data3o['62Ni'] / data3o['60Ni'])
    datalogCu3 = np.log(data3o['65Cu'] / data3o['63Cu'])
    
    datalog = pd.DataFrame()
    data2log = pd.DataFrame()
    data3log = pd.DataFrame()

    datalog['ln(62Ni/60Ni)'] = datalogNi
    datalog['ln(65Cu/63Cu)'] = datalogCu
    data2log['ln(62Ni/60Ni)'] = datalogNi2
    data2log['ln(65Cu/63Cu)'] = datalogCu2
    data3log['ln(62Ni/60Ni)'] = datalogNi3
    data3log['ln(65Cu/63Cu)'] = datalogCu3
    
    datalogNi_all = np.log(data_all['62Ni']/data_all['60Ni'])
    datalogCu_all = np.log(data_all['65Cu']/data_all['63Cu'])    
    
    

    # Erstelle DataFrames mit den Daten
    df21 = pd.DataFrame({
        "ln(62Ni/60Ni)": datalogNi,
        "ln(65Cu/63Cu)": datalogCu,
        "Color": ['1. Block'] * len(datalogNi)
    })

    df22 = pd.DataFrame({
        "ln(62Ni/60Ni)": datalogNi2,
        "ln(65Cu/63Cu)": datalogCu2,
        "Color": ['2. Block'] * len(datalogNi2)
    })

    df23 = pd.DataFrame({
        "ln(62Ni/60Ni)": datalogNi3,
        "ln(65Cu/63Cu)": datalogCu3,
        "Color": ['3. Block'] * len(datalogNi3)
    })

    # Füge die DataFrames zusammen
    df222 = pd.concat([df21, df22, df23])

# Erstelle einen Scatter-Plot mit Plotly Express
    fig222 = px.scatter(df222, y="ln(62Ni/60Ni)", x="ln(65Cu/63Cu)", color="Color")

        
    st.plotly_chart(fig222, use_container_width=True)

    
    
    


    
def plot_Cu_Standards():

    global outfile_plt_Cu_Std
    global outfile_results_Cu
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu
    
    save = outfile_plt_Cu + '/Standards'
    sb.set(rc={'figure.figsize':(25.7,8.27)})
    
    fullname = os.path.join(outfile_results_Cu, 'Cu_export.csv')
    pd.options.mode.chained_assignment = None
    data = pd.read_csv(fullname, sep='\t')
    data['Name'] = data['Filename']
    data = data.set_index('Filename')

    datalog = pd.DataFrame(index=data.index)
    datalog['log65Cu/63Cu'] = np.log((data['65Cu']/data['63Cu']))
    datalog['log62Ni/60Ni'] = np.log((data['62Ni']/data['60Ni']))
    datalog['standard'] = datalog.index.str.contains('_s0|s1')
    datalog['Name'] = data['Name']

    datalogStandard=datalog[datalog['standard'] == True]
    datalogStandard.insert(4, 'ID', range(1, 1+len(datalogStandard)))
    datalogStandard['ID'] = datalogStandard['ID'].round(decimals=0).astype(object)

    slope, intercept, r, p, se = stats.linregress(datalogStandard['log62Ni/60Ni'], datalogStandard['log65Cu/63Cu'])
    rsquared = r**2   


    
    fig = px.scatter(datalogStandard, x='log62Ni/60Ni', y='log65Cu/63Cu', text='ID', labels={'x':'ln($^{62}$Ni/$^{60}$Ni)', 'y':'ln($^{65}$Cu/$^{63}$Cu)'})
    fig.update_traces(textposition='top center')
    fig.add_trace(px.line(datalogStandard, x='log62Ni/60Ni', y=intercept + slope*datalogStandard['log62Ni/60Ni']).data[0])
    fig.update_layout(title_text='ln($^{62}$Ni/$^{60}$Ni) vs ln($^{65}$Cu/$^{63}$Cu) of Standards', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)



def plot_dCu_Standards():
    global outfile_plt_Cu_Std
    global outfile_results_Cu
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu
    global save_dStandards
    global save_dStandardsExcel
    save = outfile_plt_Cu + '/Standards'
    sb.set_theme(rc={'figure.figsize':(25.7,8.27)})
    
    fullname = os.path.join(outfile_results_Cu, 'Cu_export.csv')
    pd.options.mode.chained_assignment = None
    df = pd.read_csv(fullname, sep='\t')
    
    
    True_Nickel = 0.036345 / 0.262231
    
    df = df[df['Filename'].str.contains('Blk')==False]
    df = df[df['Filename'].str.contains('s0|s1')]
    for index in df.index:
        df['60Ni/61Ni'] = (df['60Ni'] / df['61Ni'])
        
    True_60Ni_61Ni_mean = np.mean(df['60Ni/61Ni'])
    True_60Ni_61Ni_mean_2SD = np.std(df['60Ni/61Ni']) * 2
    True_60Ni_61Ni_mean_2RSD = True_60Ni_61Ni_mean_2SD / True_60Ni_61Ni_mean * 1000
    
    
    True_65Cu_63Cu_mean = np.mean(df['65Cu/63Cu'])
    True_65Cu_63Cu_mean_2SD = np.std(df['65Cu/63Cu']) * 2
    True_65Cu_63Cu_mean_2RSD = True_65Cu_63Cu_mean_2SD / True_65Cu_63Cu_mean * 1000
    
    
    for index in df.index:
        df['d60Ni'] = ((df['60Ni/61Ni'] / True_60Ni_61Ni_mean) - 1) * 1000 
        df['d65Cu'] = ((df['65Cu/63Cu'] / True_65Cu_63Cu_mean) - 1) * 1000 
    
    
    df['d60Ni_Std_mean'] = np.mean(df['d60Ni'])
    df['d65Cu_Std_mean'] = np.mean(df['d65Cu'])

    df['d60Ni_Std_mean_2SD'] = np.std(df['d60Ni']) * 2
    df['d65Cu_Std_mean_2SD'] = np.std(df['d65Cu']) * 2
    
    
    reg = np.polyfit(df['63Cu'], df['60Ni'], 1)
    predict = np.poly1d(reg) # Slope and intercept
    trend = np.polyval(reg, df['63Cu'])
    std = df['60Ni'].std() # Standard deviation
    r2 = np.round(r2_score(df['60Ni'], predict(df['63Cu'])), 5) #R-squared
    r2string = str(r2)
    
    df.insert(1, 'ID', range(1, 1+len(df)))
    df['ID'] = df['ID'].round(decimals=0).astype(object)
    
    

    

    # Erster Plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['63Cu'], y=df['60Ni'], mode='markers', name='Datenpunkte'))
    fig.add_trace(go.Scatter(x=df['63Cu'], y=trend, mode='lines', name='Trendlinie'))

    fig.update_layout(
        title="Voltages of Cu and Ni of Standards",
        xaxis_title="$^{63}$Cu (V)",
        yaxis_title="$^{60}$Ni (V)",
        legend_title="Legende",
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
    )

    for i in range(len(df)):
        fig.add_annotation(x=df['63Cu'].iloc[i], y=df['60Ni'].iloc[i], text=str(df['ID'].iloc[i]))
    fig.add_annotation(text="R² = "+r2string, showarrow=False, font=dict(size=16))
    fig.add_annotation(text="Intercept = "+str(np.round(reg[1], 5)), showarrow=False, font=dict(size=16))
    fig.add_annotation(text="Slope = "+str(np.round(reg[0], 5)), showarrow=False, font=dict(size=16))

    # Zweiter Plot
    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(x=df['d65Cu'], y=df['d60Ni'], mode='markers', name='Datenpunkte'))

    fig2.update_layout(
        title="d$^{63}$Cu vs d$^{60}$Ni",
        xaxis_title="d$^{63}$Cu",
        yaxis_title="d$^{60}$Ni",
        legend_title="Legende",
        autosize=False,
        width=800,
        height=800,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
    )

    for i in range(len(df)):
        fig2.add_annotation(x=df['d65Cu'].iloc[i], y=df['d60Ni'].iloc[i], text=str(df['ID'].iloc[i]))
        
    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    
    
    
def plot_Cu_outlier_Blk(file):
    global fullname
    global outfile
    global outfile_results_Cu
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu
    global standardinputpfad
    global standardoutputpfad
    
    fullname = os.path.join(standardoutputpfad)
    
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float) 
    datacurrentplot = data.copy()
    datacurrentplot = datacurrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    
    datao = pd.DataFrame({'Time':data['Time']})
    dataofig = datao.copy()
    
    data = pd.concat([data, data[cols].div(data['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data = pd.concat([data, data[cols].div(data['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    datafig = data.copy()
    
    data.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu', '62Ni/60Ni'], axis = 1, inplace = True)
    datafig.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu', '65Cu/63Cu'], axis = 1, inplace = True)
    
    cols1 = list(data.columns)
    
    datao['65Cu/63Cu'] = data['65Cu/63Cu']
    
    datao['65Cu/63Cu'] = datao['65Cu/63Cu'].where(np.abs(stats.zscore(datao['65Cu/63Cu'])) < 2)
    
    fig = go.Figure()
    for c in cols1:
        fig.add_trace(go.Scatter(x=data['Time'], y=data['65Cu/63Cu'], mode='markers', name='Original', marker_color='red'))
        fig.add_trace(go.Scatter(x=datao['Time'], y=datao['65Cu/63Cu'], mode='markers', name='Filtered', marker_color='blue'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='65Cu/63Cu vs Time') #basename
    
    colsfig1 = list(datafig.columns)
    
    dataofig['62Ni/60Ni'] = datafig['62Ni/60Ni']
    dataofig['62Ni/60Ni'] = dataofig['62Ni/60Ni'].where(np.abs(stats.zscore(dataofig['62Ni/60Ni'])) < 2)
    fig2 = go.Figure()
    for c in colsfig1:
        fig2.add_trace(go.Scatter(x=datafig['Time'], y=datafig['62Ni/60Ni'], mode='markers', name='Original', marker_color='red'))
        fig2.add_trace(go.Scatter(x=dataofig['Time'], y=dataofig['62Ni/60Ni'], mode='markers', name='Filtered', marker_color='blue'))
    fig2.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='62Ni/60Ni vs Time') #basename

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.plotly_chart(fig2)


def plot_Cu_outlier_Std(file):
    global fullname
    global outfile
    global outfile_results_Cu
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu
    global standardinputpfad
    global standardoutputpfad
    
    fullname = os.path.join(standardoutputpfad)
    
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8}'.format
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)

    datacurrentplot = data.copy()
    datacurrentplot = datacurrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    
    datao = pd.DataFrame({'Time':data['Time']})
    dataofig = datao.copy()
    
    data = pd.concat([data, data[cols].div(data['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data = pd.concat([data, data[cols].div(data['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    datafig = data.copy()
    
    data.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu', '62Ni/60Ni'], axis = 1, inplace = True)
    datafig.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu', '65Cu/63Cu'], axis = 1, inplace = True)
    
    cols1 = list(data.columns)
    
    datao[cols1] = data[cols1]
    
    datao['65Cu/63Cu'] = datao['65Cu/63Cu'].where(np.abs(stats.zscore(datao['65Cu/63Cu'])) < 2)
    
    fig = go.Figure()
    for c in cols1:
        fig.add_trace(go.Scatter(x=data['Time'], y=data['65Cu/63Cu'], mode='markers', name='Original', marker_color='red'))
        fig.add_trace(go.Scatter(x=datao['Time'], y=datao['65Cu/63Cu'], mode='markers', name='Filtered', marker_color='blue'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='65Cu/63Cu vs Time') #basename
    
    colsfig1 = list(datafig.columns)
    
    dataofig['62Ni/60Ni'] = datafig['62Ni/60Ni']
    dataofig['62Ni/60Ni'] = dataofig['62Ni/60Ni'].where(np.abs(stats.zscore(dataofig['62Ni/60Ni'])) < 2)
    fig2 = go.Figure()
    for c in colsfig1:
        fig2.add_trace(go.Scatter(x=datafig['Time'], y=datafig['62Ni/60Ni'], mode='markers', name='Original', marker_color='red'))
        fig2.add_trace(go.Scatter(x=dataofig['Time'], y=dataofig['62Ni/60Ni'], mode='markers', name='Filtered', marker_color='blue'))
    fig2.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='62Ni/60Ni vs Time') #basename

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.plotly_chart(fig2)


def plot_Cu_outlier_Sample(file):
    global fullname
    global outfile
    global outfile_results_Cu
    global outfile_corrected_raw_data_Cu
    global outfile_plt_Cu
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
    
    
    datacurrentplot = data_all.copy()
    datacurrentplot = datacurrentplot.set_index('Time')
    
    cols = list(data.drop(columns='Time').columns)
    datao = pd.DataFrame({'Time':data['Time']})
    data = pd.concat([data, data[cols].div(data['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data = pd.concat([data, data[cols].div(data['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    dataofig = data.copy()
    dataofig.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu', '65Cu/63Cu'], axis = 1, inplace = True)
    
    data.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    colsa1 = list(data.columns)

    cols2 = list(data2.drop(columns='Time').columns)
    data2o = pd.DataFrame({'Time':data2['Time']})
    data2 = pd.concat([data2, data2[cols2].div(data2['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data2 = pd.concat([data2, data2[cols2].div(data2['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data2.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    colsa2 = list(data2.columns)

    cols3 = list(data3.drop(columns='Time').columns)
    data3o = pd.DataFrame({'Time':data3['Time']})
    data3 = pd.concat([data3, data3[cols3].div(data3['60Ni'], axis=0).add_suffix('/60Ni')], axis=1)
    data3 = pd.concat([data3, data3[cols3].div(data3['63Cu'], axis=0).add_suffix('/63Cu')], axis=1)
    data3.drop(['60Ni/60Ni', '61Ni/60Ni', '63Cu/60Ni', '64Ni/60Ni', '65Cu/60Ni', '66Zn/60Ni', '60Ni/63Cu', '61Ni/63Cu', '62Ni/63Cu', '63Cu/63Cu', '64Ni/63Cu', '66Zn/63Cu'], axis = 1, inplace = True)
    colsa3 = list(data3.columns)
    
    
    datao['65Cu/63Cu'] = data['65Cu/63Cu']
    datao['65Cu/63Cu'] = datao['65Cu/63Cu'].where(np.abs(stats.zscore(datao['65Cu/63Cu'])) < 2)
    
    data2o['65Cu/63Cu'] = data2['65Cu/63Cu']
    data2o['65Cu/63Cu'] = data2o['65Cu/63Cu'].where(np.abs(stats.zscore(data2o['65Cu/63Cu'])) < 2)
    
    data3o['65Cu/63Cu'] = data3['65Cu/63Cu']
    data3o['65Cu/63Cu'] = data3o['65Cu/63Cu'].where(np.abs(stats.zscore(data3o['65Cu/63Cu'])) < 2)
    
    
    fig = go.Figure()
    for c in colsa1:
        fig.add_trace(go.Scatter(x=data['Time'], y=data['65Cu/63Cu'], mode='markers', name='Original', marker_color='red'))
        fig.add_trace(go.Scatter(x=data2['Time'], y=data2['65Cu/63Cu'], mode='markers', name='Original', marker_color='red'))
        fig.add_trace(go.Scatter(x=data3['Time'], y=data3['65Cu/63Cu'], mode='markers', name='Original', marker_color='red'))
        fig.add_trace(go.Scatter(x=datao['Time'], y=datao['65Cu/63Cu'], mode='markers', name='Filtered', marker_color='green'))
        fig.add_trace(go.Scatter(x=data2o['Time'], y=data2o['65Cu/63Cu'], mode='markers', name='Filtered', marker_color='magenta'))
        fig.add_trace(go.Scatter(x=data3o['Time'], y=data3o['65Cu/63Cu'], mode='markers', name='Filtered', marker_color='blue'))
    fig.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='65Cu/63Cu vs Time') #basename
    
    colsfig1 = list(dataofig.columns)
    
    datao['62Ni/60Ni'] = data['62Ni/60Ni']
    datao['62Ni/60Ni'] = datao['62Ni/60Ni'].where(np.abs(stats.zscore(datao['62Ni/60Ni'])) < 2)
    
    data2o['62Ni/60Ni'] = data2['62Ni/60Ni']
    data2o['62Ni/60Ni'] = data2o['62Ni/60Ni'].where(np.abs(stats.zscore(data2o['62Ni/60Ni'])) < 2)
    
    data3o['62Ni/60Ni'] = data3['62Ni/60Ni']
    data3o['62Ni/60Ni'] = data3o['62Ni/60Ni'].where(np.abs(stats.zscore(data3o['62Ni/60Ni'])) < 2)

    fig2 = go.Figure()
    for c in colsfig1:
        fig2.add_trace(go.Scatter(x=data['Time'], y=data['62Ni/60Ni'], mode='markers', name='Original', marker_color='red'))
        fig2.add_trace(go.Scatter(x=data2['Time'], y=data2['62Ni/60Ni'], mode='markers', name='Original', marker_color='red'))
        fig2.add_trace(go.Scatter(x=data3['Time'], y=data3['62Ni/60Ni'], mode='markers', name='Original', marker_color='red'))
        fig2.add_trace(go.Scatter(x=datao['Time'], y=datao['62Ni/60Ni'], mode='markers', name='Filtered', marker_color='blue'))
        fig2.add_trace(go.Scatter(x=data2o['Time'], y=data2o['62Ni/60Ni'], mode='markers', name='Filtered', marker_color='green'))
        fig2.add_trace(go.Scatter(x=data3o['Time'], y=data3o['62Ni/60Ni'], mode='markers', name='Filtered', marker_color='magenta'))
    fig2.update_layout(xaxis_title='Time (s)', yaxis_title='${}${}'.format(c[:2], c[2:]), title='62Ni/60Ni vs Time') 

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig)
    with col2:
        st.plotly_chart(fig2)


def Voltages_Cu(file):
        global fullname
        global outfile
        global outfile_results_Cu
        global outfile_corrected_raw_data_Cu
        global outfile_plt_Cu

        fullname = os.path.join(standardoutputpfad)
        basename = os.path.basename(file)
        
        pd.options.display.float_format = '{:.8}'.format
        col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
        col_names.remove('Trace for Mass:')
        col_names.insert(0, 'Time')
        data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)
        st.write('63Cu/60Ni ratio = ' + str(round(mean(data['63Cu']/data['60Ni']), 2)))
        datacurrentplot = data.copy()
        datacurrentplot = datacurrentplot.set_index('Time')
        fig3 = px.scatter(datacurrentplot, x=datacurrentplot.index, y=datacurrentplot.columns)
        fig3.update_layout(xaxis_title='Time (s)', yaxis_title='Voltage (V)', title=basename, autosize=True)
        st.plotly_chart(fig3, use_container_width=True)