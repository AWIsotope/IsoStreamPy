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
import modules.outlierSn_short as outlierSn_short
import modules.plotSn as plotSn
import modules.plotSn_short as plotSn_short
import modules.plotCu as plotCu
import modules.baxterSn as baxterSn
t = time.process_time()

from modules.config import outfile_plt_Sn, outfile_corrected_raw_data_Sn, outfile_results_Sn, Baxter_Sn

# load and create dirs
infile = WorkspaceVariableInput 
entries = Path(infile)

# Calculation functions

def Cu(file):
    file = str(file)
    if file.endswith('Blk.TXT') or file.endswith('BLK_.TXT'):
        outlierCu.outlier_Cu_blk(file)
        plotCu.plot_Cu_blk(file)
    elif file.endswith('.TXT') and 's' in file:
        outlierCu.outlier_Cu_std(file)
        plotCu.plot_Cu_std(file)
    else:
        outlierCu.outlier_Cu_sample(file)
        plotCu.plot_Cu_sample(file)
    return f"Finished processing {file}"    

def Sn(file):
    file = str(file)
    global Baxter_Sn
    global WorkspaceVariableInput
    Folder = WorkspaceVariableInput
    file_list = []
    for file in os.listdir(Folder):
        if file.endswith('Blk.TXT'):
            pass
        elif file.endswith('.log'):
            pass
        elif file.endswith('.dat'):
            pass
        elif file.endswith('.exp'):
            pass
        elif file.endswith('.TDT'):
            pass
        elif file.endswith('.corrupted'):
            pass
        elif file.endswith('.ini'):
            pass
        else:
            file_list.append(file)
                
    if 'Nis' in file_list[-1]:
        print('Using Bracketing Method')
        if file.endswith('Blk.TXT') or file.endswith('BLK_.TXT'):
            outlierSn.outlier_Sn_blk(file)
            plotSn.plot_Sn_blk(file)
        elif file.endswith('.TXT') and 'Nis' in file:
            outlierSn.outlier_Sn_std(file)
            plotSn.plot_Sn_std(file)
        elif file.endswith('.TXT') and 'Ni10' in file:
            outlierSn.outlier_Sn_std(file)
            plotSn.plot_Sn_std(file)
        elif file.endswith('.TXT') and 'Ni11' in file:
            outlierSn.outlier_Sn_std(file)
            plotSn.plot_Sn_std(file)
        else:
            outlierSn.outlier_Sn_sample(file)
            plotSn.plot_Sn_sample(file)
            return f"Finished processing {file}"
    else:
        print('Since last file is ' + file_list[-1] + ' I will only use the Baxter Method')        
        time_baxter_start = time.perf_counter()
        baxterSn.baxterSn()
        time_baxter_ende = time.perf_counter()
        print('')
        print('##  Baxter finished  ##')
        print('')
        print('')
        print("##  The Baxter calculation took " "{:1.3f} seconds to complete  ##".format(time_baxter_ende - time_baxter_start))
    
    
    

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
            
#def multiprocessSnPlot():
#    data = entries.glob("*.TXT")

#    with mp.Pool() as p:
#        for res in p.imap_unordered(SnPlot, data): #map(Cu, data):#imap_unordered(Cu, data):
#            print(res)