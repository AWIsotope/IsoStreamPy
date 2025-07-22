# -*- coding: utf-8 -*-
"""
This module contains the processing functions and multiprocessing.

@author: Andreas Wittke
"""

# Import Modules
import time
import os
import multiprocessing as mp
import warnings
import sys
sys.path.append('')
from datetime import date
from pathlib import Path
warnings.simplefilter(action='ignore', category=FutureWarning)

# load individual modules
from modules.config import WorkspaceVariableInput
import modules.outlierSn as outlierSn
import modules.outlierCu as outlierCu
import modules.outlierLi as outlierLi
import modules.outlierSb as outlierSb
import modules.outlierSn_short as outlierSn_short
import modules.plotSn as plotSn
import modules.plotSn_short as plotSn_short
import modules.plotCu as plotCu
import modules.plotLi as plotLi
import modules.plotSb as plotSb

import pickle

# Pfad zur Datei, aus der die Variable geladen werden soll
file_path = os.path.join("modules", "variable_data.pkl")

# Variable mit pickle laden
with open(file_path, 'rb') as file:
    standardoutputpfad = pickle.load(file)
    
file_path2 = os.path.join("modules", "variable_data_input.pkl")

# Variable mit pickle laden
with open(file_path2, 'rb') as file2:
    standardinputpfad = pickle.load(file2)



t = time.process_time()

# load and create dirs
infile = standardinputpfad
entries = Path(infile)


# Calculation functions

def Cu(file):
    file = str(file)
    if file.endswith('Blk.TXT') or file.endswith('BLK_.TXT'):
        outlierCu.outlier_Cu_blk(file)
    elif file.endswith('.TXT') and 's' in file:
        outlierCu.outlier_Cu_std(file)
    else:
        outlierCu.outlier_Cu_sample(file)
    return f"Finished processing {file}"    

def Sn(file):
    file = str(file)
    print(file)
    if file.endswith('Blk.TXT') or file.endswith('BLK_.TXT'):
        outlierSn.outlier_Sn_blk(file)
    elif file.endswith('.TXT') and 'Nis' in file:
        outlierSn.outlier_Sn_std(file)
    elif file.endswith('.TXT') and 'Ni10' in file:
        outlierSn.outlier_Sn_std(file)
    elif file.endswith('.TXT') and 'Ni11' in file:
        outlierSn.outlier_Sn_std(file)
    else:
        outlierSn.outlier_Sn_sample(file)
    return f"Finished processing {file}"

def Sn_short(file):
    file = str(file)
    if file.endswith('Blk.TXT') or file.endswith('BLK_.TXT'):
        outlierSn_short.outlier_Sn_blk(file)
        plotSn_short.plot_Sn_blk(file)
    elif file.endswith('.TXT') and 'Nis' in file:
        outlierSn_short.outlier_Sn_std(file)
        plotSn_short.plot_Sn_std(file)
    elif file.endswith('.TXT') and 'Ni10' in file:
        outlierSn_short.outlier_Sn_std(file)
        plotSn_short.plot_Sn_std(file)
    elif file.endswith('.TXT') and 'Ni11' in file:
        outlierSn_short.outlier_Sn_std(file)
        plotSn_short.plot_Sn_std(file)
    else:
        outlierSn_short.outlier_Sn_sample(file)
        plotSn_short.plot_Sn_sample(file)
    return f"Finished processing {file}"

def PSn(file):
    file = str(file)
    if file.endswith('Blk.TXT') or file.endswith('BLK.TXT'):
        outlierSn.outlier_Sn_blk(file)
        plotSn.plot_Sn_blk(file)
    elif file.endswith('.TXT') and 'JM' in file:
        outlierSn.outlier_Sn_std(file)
        plotSn.plot_Sn_std(file)
    elif file.endswith('.TXT') and 'jm' in file:
        outlierSn.outlier_Sn_std(file)
        plotSn.plot_Sn_std(file)
    
    else:
        outlierSn.outlier_Sn_sample(file)
        plotSn.plot_Sn_sample(file)
    return f"Finished processing {file}"

def Ag(file):
    file = str(file)
    if file.endswith('Blk.TXT') or file.endswith('BLK.TXT'):
        outlierSn.outlier_Ag_blk(file)
        plotSn.plot_Ag_blk(file)
    elif file.endswith('.TXT') and 'JM' in file:
        outlierSn.outlier_Ag_std(file)
        plotSn.plot_Ag_std(file)
    elif file.endswith('.TXT') and 'jm' in file:
        outlierSn.outlier_Ag_std(file)
        plotSn.plot_Ag_std(file)
    
    else:
        outlierSn.outlier_Ag_sample(file)
        plotSn.plot_Ag_sample(file)
    return f"Finished processing {file}"
    
    
    
def Li(file):
    file = str(file)
    if file.endswith('Blk.TXT') or file.endswith('BLK_.TXT'):
        outlierLi.outlier_Li_blk(file)
        plotLi.plot_Li_blk(file)
    elif file.endswith('.TXT') and 's' in file:
        outlierLi.outlier_Li_std(file)
        plotLi.plot_Li_std(file)
    else:
        outlierLi.outlier_Li_sample(file)
        plotLi.plot_Li_sample(file)
    return f"Finished processing {file}"  


def Sb(file):
    file = str(file)
    #print("FILE", file)
    if file.endswith('Blk.TXT') or file.endswith('BLK_.TXT'):
        outlierSb.outlier_Sb_blk(file)
        # plotSb.plot_Sb_blk(file)
    elif file.endswith('.TXT') and 'Ni' in file:
        outlierSb.outlier_Sb_std(file)
        # plotSb.plot_Sb_std(file)
    else:
        outlierSb.outlier_Sb_sample(file)
        # plotSb.plot_Sb_sample(file)
    return f"Finished processing {file}"    


# Multiprocessing

def multiprocessCu():
    data = sorted(entries.glob("*.TXT"))
    with mp.Pool() as p:
        for res in p.imap_unordered(Cu, data): 
            print(res)
            
    
def multiprocessSn():
    data = sorted(entries.glob("*.TXT"))

    with mp.Pool() as p:
        for res in p.imap_unordered(Sn, data): 
            print(res)
            
def multiprocessSn_short():
    data = sorted(entries.glob("*.TXT"))

    with mp.Pool() as p:
        for res in p.imap_unordered(Sn_short, data): 
            print(res)
            
def multiprocessPSn():
    data = sorted(entries.glob("*.TXT"))

    with mp.Pool() as p:
        for res in p.imap_unordered(PSn, data): 
            print(res)
            
def multiprocessAg():
    data = sorted(entries.glob("*.TXT"))

    with mp.Pool() as p:
        for res in p.imap_unordered(Ag, data):
            print(res)
            
            
            
def multiprocessLi():
    data = sorted(entries.glob("*.TXT"))

    with mp.Pool() as p:
        for res in p.imap_unordered(Li, data): 
            print(res)
            
#def multiprocessSnPlot():
#    data = entries.glob("*.TXT")

#    with mp.Pool() as p:
#        for res in p.imap_unordered(SnPlot, data): 
#            print(res)

def multiprocessSb():
    data = sorted(entries.glob("*.TXT"))
    #print('DATA', data)
    with mp.Pool() as p:
        for res in p.imap_unordered(Sb, data):
            print(res)
            