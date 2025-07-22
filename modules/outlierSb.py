# -*- coding: utf-8 -*-
"""
This module containts the Outlier correction for Sb

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
from modules.config import WorkspaceVariableInput, outfile_plt_Sb, outfile_corrected_raw_data_Sb, outfile_results_Sb, Baxter_Sb, exportSb #outfile

file_path = os.path.join("modules", "variable_data.pkl")

# Variable mit pickle laden
with open(file_path, 'rb') as file:
    standardoutputpfad = pickle.load(file)
    
file_path2 = os.path.join("modules", "variable_data_input.pkl")
# Variable mit pickle laden
with open(file_path2, 'rb') as file2:
    standardinputpfad = pickle.load(file2)
    

infile = standardinputpfad
     
def outlier_Sb_blk(file, append=True):
    global fullname
    global outfile_results_Sb
    global exportSb

    fullname = os.path.join(outfile_results_Sb, 'Sb_export.csv')
    basename = os.path.basename(file)
    
    pd.options.display.float_format = '{:.8f}'.format

    # Spaltennamen einlesen und anpassen
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    
    # Daten einlesen
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)

    # Unerwünschte Spalten entfernen
    data.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis=1, inplace=True)

    cols_sn = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb = ['123Sb']   # Nur 123Sb/121Sb

    # Berechnung und Hinzufügen der Verhältnisse
    data = pd.concat([data, data[cols_sn].div(data['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data = pd.concat([data, data[cols_sb].div(data['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)

    # Erstellen eines neuen DataFrames für die Ausreißeranalyse
    datao = pd.DataFrame({'Time': data['Time']})
    cols1 = list(data.columns)
    datao[cols1] = data[cols1]

    # Ausreißerfilterung
    del cols1[0:5] 
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)

    # Ergebnisse speichern
    datao.to_csv(exportSb + basename + '.csv', sep='\t', header=True, index_label='Index_name')
    
    # Mittelwertberechnung und Umwandlung in DataFrame
    mean_data = datao.mean()
    mean_filtered_transposed = pd.DataFrame(mean_data).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1)
    clean.insert(0, 'Inputfile', file)

    # Speichern der Mittelwerte
    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')


def outlier_Sb_std(file, append=True):
    global outfile
    global outfile_results_Sb
    global outfile_corrected_raw_data_Sb
    global Baxter_Sb

    fullname = os.path.join(outfile_results_Sb, 'Sb_export.csv')
    basename = os.path.basename(file)
  
    pd.options.display.float_format = '{:.8f}'.format

    # Einlesen der Daten mit angepassten Spaltennamen
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    data = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, index_col=False, dtype=float)

    # Entfernen unerwünschter Spalten
    data.drop(['116Sn', '117Sn', '119Sn', '122Sn', '124Sn'], axis=1, inplace=True)
    
    cols_sn = ['120Sn']  # Nur 120Sn/118Sn
    cols_sb = ['123Sb']  # Nur 123Sb/121Sb

    # Berechnung und Hinzufügen der Verhältnisse
    data = pd.concat([data, data[cols_sn].div(data['118Sn'], axis=0).add_suffix('/118Sn')], axis=1)
    data = pd.concat([data, data[cols_sb].div(data['121Sb'], axis=0).add_suffix('/121Sb')], axis=1)

    # Erstellen eines neuen DataFrames für die Ausreißeranalyse
    datao = pd.DataFrame({'Time': data['Time']})
    cols1 = list(data.columns)
    datao[cols1] = data[cols1]

    # Ausreißerfilterung
    del cols1[0:5] 
    datao[cols1] = datao[cols1].where(np.abs(stats.zscore(datao[cols1])) < 2)

    # Definition der Massenwerte
    masses = pd.DataFrame({
        '118Sn': 117.901606625,
        '120Sn': 119.902202063,
        '121Sb': 120.903810584,
        '123Sb': 122.904211755
    }, index=[0])
    
    # Berechnung der Korrektur
    TrueStdSn = 32.593 / 24.223
    fSnstd = (np.log(TrueStdSn / ((datao['120Sn/118Sn']))) / (np.log(119.902202063 / 117.901606625)))
    # Berechnung der Korrektur für jede Zeile in datao
    datao['123Sb/121Sb_corrected'] = datao['123Sb/121Sb'] * ((masses.loc[0, '123Sb'] / masses.loc[0, '121Sb']) ** fSnstd)


    # Speichern der korrigierten Daten
    datao.to_csv(outfile_corrected_raw_data_Sb + basename + '.csv', sep='\t', header=True, index_label='Index_name')
    
    # Mittelwertberechnung und Speichern der Mittelwerte
    mean_data = datao.mean()
    mean_filtered_transposed = pd.DataFrame(mean_data).T
    mean_filtered_transposed['Time'] = pd.to_datetime(mean_filtered_transposed["Time"], unit='s')
    clean = mean_filtered_transposed.drop(mean_filtered_transposed.columns[[0]], axis=1) 
    clean.insert(0, 'Inputfile', file)

    # Speichern der Mittelwerte
    if append:
        clean.to_csv(fullname, sep='\t', mode="a", header=False, index_label='Index_name')
    else:
        clean.to_csv(fullname, sep='\t', mode="w", header=True, index_label='Index_name')
        
    # Baxter-Daten vorbereiten
    databaxter_1 = clean.copy()
    databaxter_1 = databaxter_1.drop('123Sb/121Sb_corrected', axis=1)
    databaxter_1['123Sb/121Sb_1SE'] = datao['123Sb/121Sb'].std(ddof=1) / np.sqrt(np.size(datao['123Sb/121Sb']))
    databaxter_1['120Sn/118Sn_1SE'] = datao['120Sn/118Sn'].std(ddof=1) / np.sqrt(np.size(datao['120Sn/118Sn']))

    # Speichern der Baxter-Daten
    databaxter_1.to_csv(Baxter_Sb + 'Baxter.csv', sep='\t', mode="a", header=False, index_label='Index_name')
    

def outlier_Sb_sample(file, append=True):
    global outfile
    global outfile_results_Sb 
    global outfile_corrected_raw_data_Sb
    global outfile_plt_Sb
    global Baxter_Sb
    global infile
    
    fullname = os.path.join(outfile_results_Sb, 'Sb_export.csv')
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
    datao['123Sb/121Sb_corrected'] = datao['123Sb/121Sb'] * ((122.904211755 / 120.903810584) ** fSnstd)
    
    fSnstd2 = (np.log(TrueStdSn / ((data2o['120Sn/118Sn']))) / (np.log(119.902202063 / 117.901606625)))
    data2o['123Sb/121Sb_corrected'] = data2o['123Sb/121Sb'] * ((122.904211755 / 120.903810584) ** fSnstd2)

    fSnstd3 = (np.log(TrueStdSn / ((data3o['120Sn/118Sn']))) / (np.log(119.902202063 / 117.901606625)))
    data3o['123Sb/121Sb_corrected'] = data3o['123Sb/121Sb'] * ((122.904211755 / 120.903810584) ** fSnstd3)
    #''''''
    datao.to_csv(outfile_corrected_raw_data_Sb + basename + '_1.csv', sep='\t', header = True, index_label='Index_name')
    data2o.to_csv(outfile_corrected_raw_data_Sb + basename + '_2.csv', sep='\t', header = True, index_label='Index_name')
    data3o.to_csv(outfile_corrected_raw_data_Sb + basename + '_3.csv', sep='\t', header = True, index_label='Index_name')
    
    
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
    datalogNi = np.log(datao['120Sn'] / datao['118Sn'])
    datalogSb = np.log(datao['123Sb'] / datao['121Sb'])
    datalogNi2 = np.log(data2o['120Sn'] / data2o['118Sn'])
    datalogSb2 = np.log(data2o['123Sb'] / data2o['121Sb'])
    datalogNi3 = np.log(data3o['120Sn'] / data3o['118Sn'])
    datalogSb3 = np.log(data3o['123Sb'] / data3o['121Sb'])
    

    databaxter_1 = clean.copy()
    databaxter_2 = clean2.copy()
    databaxter_3 = clean3.copy()
    databaxter_1 = databaxter_1.drop('123Sb/121Sb_corrected', axis = 1)
    databaxter_2 = databaxter_2.drop('123Sb/121Sb_corrected', axis = 1)
    databaxter_3 = databaxter_3.drop('123Sb/121Sb_corrected', axis = 1)
    
    databaxter_1['123Sb/121Sb_1SE'] = datao['123Sb/121Sb'].std(ddof = 1) / np.sqrt(np.size(datao['123Sb/121Sb']))
    databaxter_1['120Sn/118Sn_1SE'] = datao['120Sn/118Sn'].std(ddof = 1) / np.sqrt(np.size(datao['120Sn/118Sn']))

    
    baxter_header = databaxter_1.copy()
    baxter_header.to_csv(Baxter_Sb + 'Baxter_header.csv', sep='\t')
    databaxter_2['123Sb/121Sb_1SE'] = data2o['123Sb/121Sb'].std(ddof = 1) / np.sqrt(np.size(data2o['123Sb/121Sb']))
    databaxter_2['120Sn/118Sn_1SE'] = data2o['120Sn/118Sn'].std(ddof = 1) / np.sqrt(np.size(data2o['120Sn/118Sn']))
    
    databaxter_3['123Sb/121Sb_1SE'] = data3o['123Sb/121Sb'].std(ddof = 1) / np.sqrt(np.size(data3o['123Sb/121Sb']))
    databaxter_3['120Sn/118Sn_1SE'] = data3o['120Sn/118Sn'].std(ddof = 1) / np.sqrt(np.size(data3o['120Sn/118Sn']))
    
    databaxter_1.to_csv(Baxter_Sb + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    databaxter_2.to_csv(Baxter_Sb + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    databaxter_3.to_csv(Baxter_Sb + 'Baxter.csv', sep='\t', mode = "a", header = False, index_label='Index_name')
    
    return baxter_header
