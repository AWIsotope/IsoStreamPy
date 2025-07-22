#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This contains the calculation for Sn Isotopes

@author: Andreas Wittke
"""
import time
import pandas as pd
import os
import warnings
import multiprocessing as mp
from datetime import date
from pathlib import Path
import sys
sys.path.append(".")
from csv import DictReader

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')
import argparse

# load individual modules
import modules.config as conf
import modules.outlierSn as outlierSn
import modules.ssb as ssb
import modules.process as process
import modules.plotSn as plotSn
import modules.plotdSn as plotdSn
import modules.plotCu as plotCu
import modules.PlotSnMasses as PlotSnMasses
import modules.corruptdata as corruptdata
from modules.config import WorkspaceVariableInput, outfile_corrected_raw_data_Sn
import modules.baxterSn as baxterSn
import modules.baxterCu as baxterCu
import pickle
import shutil

file_path = os.path.join("modules", "variable_data_outputfolder.pkl")
with open(file_path, 'rb') as file:
    outputfolder = pickle.load(file)
t = time.process_time()

# load and create dirs
infile = (conf.WorkspaceVariableInput)
available_elements = conf.available_elements


# Main Programm

inf = infile

Folder = outfile_corrected_raw_data_Sn
start = time.perf_counter()


   

print('#############')
print('')
print('##  Create Output Folders  ##')
print('')
print('#############')

def get_next_available_folder_name(base_folder):
    counter = 1
    new_folder = f"{base_folder}_{counter}"
    while os.path.exists(new_folder):
        counter += 1
        new_folder = f"{base_folder}_{counter}"
    return new_folder

def create_or_rename_folder(base_folder):
    if os.path.exists(base_folder):
        # Wenn der Ordner bereits existiert, verschiebe ihn
        new_folder = get_next_available_folder_name(base_folder)
        shutil.move(base_folder, new_folder)
        print(f"Ordner {base_folder} umbenannt in {new_folder}.")
    
    # Erstelle den Ordner
    os.makedirs(base_folder)
    print(f"Ordner {base_folder} erstellt.")



# Erstelle oder benenne die Ordner um
create_or_rename_folder(outputfolder + '/results')
create_or_rename_folder(outputfolder + '/corrected_raw_data')
create_or_rename_folder(outputfolder + '/Baxter')


                
try:
    os.makedirs(outputfolder + '/results')
except FileExistsError:
# directory already exists
    pass
outfile_results_Sn = outputfolder + '/results'


try:
    os.makedirs(outputfolder + '/corrected_raw_data')
except FileExistsError:
# directory already exists
    pass
outfile_corrected_raw_data_Sn = outputfolder + '/corrected_raw_data/'

try:
    os.makedirs(outputfolder + '/Baxter')
except FileExistsError:
# directory already exists
    pass
Baxter_Sn = outputfolder + '/Baxter/'

time.sleep(0.1)
print('')
print('## Folders created ##')
print('')
print('############')
print('')
time.sleep(0.1)
print('')
print('## Check for corrupted data')
        
print('')
print('############')
print('')
print('')


    
print('')
time.sleep(0.1)
print('##  Processing the ' + available_elements[2] + ' Isotope System  ##')
print('')
print('#############')
print('')
print('')
time.sleep(0.1)
print('##  Start with outlier correction  ##')
print('')
time_outl_start = time.perf_counter()

# entries = Path(inf)
if __name__ == '__main__':
    process.multiprocessSn()
    
time_outl_ende = time.perf_counter() 
print('')
print('##  Outlier correction finished  ##')
print('')
print('')
print("##  The outlier correction took " "{:1.3f} seconds to complete  ##".format(time_outl_ende - time_outl_start))


"""Insert Header"""
outname = os.path.join(outfile_results_Sn, 'Sn_header.csv')
fullname = os.path.join(outfile_results_Sn, 'Sn_export.csv')
file_list = os.listdir(outfile_corrected_raw_data_Sn)
deltaname = os.path.join(outfile_results_Sn, 'Sn_delta.csv')
files = [x for x in file_list if "Nis01" in x]#or "Nis01" or "Nis02" or "Nis05" or "Nis09" or "Nis08" in x] 
files = str(files)[2:-2]
files = outfile_corrected_raw_data_Sn + str(files)
col_named = DictReader(open(files, 'r'), delimiter='\t').fieldnames
col_named.remove('Index_name')
#col_named.remove('Time')
col_named.insert(1, 'Filename')
reread = pd.read_csv(fullname, sep='\t', names = col_named, index_col = False)
reread.sort_values(by=['Filename'], inplace = True)
reread.reset_index(drop=True, inplace = True)
#reread.drop(['Time'], axis = 1, inplace = True)
reread.to_csv(fullname, sep = '\t', header=True, index=False)

outputfolderBaxter_Sn = outputfolder + '/Baxter/'
"""Insert Header for Baxter calculation"""
baxter_file = os.path.join(outputfolderBaxter_Sn, 'Baxter.csv')
baxter_header = os.path.join(outputfolderBaxter_Sn, 'Baxter_header.csv')
baxter_header_df = pd.read_csv(baxter_header, nrows=1, sep = '\t')
column_names_baxter = baxter_header_df.columns.values.tolist()
baxter_file_df = pd.read_csv(baxter_file, sep = '\t', names = column_names_baxter, index_col = False)
baxter_file_df.to_csv(baxter_file, sep = '\t', header = True, index = False)

print('')
print('')
print('##  Inserted Header  ##')
time.sleep(0.1)
print('')
print('#############')
print('')
print('')

file_list = []
for file in os.listdir(Folder):
    if file.endswith('Blk.TXT.csv'):# or file.endswith('BLK_.TXT'):
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
file_list.sort()
if 'Nis' in file_list[-1]:
    print('Since last file is ' + file_list[-1] + ' I will use SSB Method first')

    time_ssb_start = time.perf_counter()
    print('##  Calculate dSn with SSB Method  ##')
    resultname = os.path.join(outfile_results_Sn, 'Sn_delta.csv')
    ssb.ssbSn()
    time_ssb_ende = time.perf_counter()
    print('')
    print('')
    print('#############')
    print('')
    print('')
    print("##  The ssb correction took " "{:1.3f} seconds to complete  ##".format(time_ssb_ende - time_ssb_start))
    print('')
    print('#############')
    print('')
    print('#############')
    print()
    print('')
    print()
    print('#############')
    print('Now I additional calculate with Baxter')
    time.sleep(0.1)
    time_baxter_start = time.perf_counter()
    baxterSn.baxterSn()
    time_baxter_ende = time.perf_counter()
    print('')
    print("##  The Baxter calculation took " "{:1.3f} seconds to complete  ##".format(time_baxter_ende - time_baxter_start))
    print()
    print('')
    print()
    print('#############')
    print('')
    print()
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
    print()
    print('')
    print()
    print('#############')
    print('')
    print()


