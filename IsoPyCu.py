#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This contains the calculation for Cu Isotopes

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

import shutil

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')
import argparse

# load individual modules
import modules.config as conf
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
import modules.plotdCustd as plotdCustd
import pickle

file_path = os.path.join("modules", "variable_data_outputfolder.pkl")
with open(file_path, 'rb') as file:
    outputfolder = pickle.load(file)
    
t = time.process_time()

# load and create dirs
infile = (conf.WorkspaceVariableInput)
available_elements = conf.available_elements
inf = infile
Folder = outfile_corrected_raw_data_Sn
start = time.perf_counter()

    # choose function depending on selected elements
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
outfile_results_Cu = outputfolder + '/results'

try:
    os.makedirs(outputfolder + '/corrected_raw_data')
except FileExistsError:
# directory already exists
    pass
try:
    os.makedirs(outputfolder + '/Baxter')
except FileExistsError:
# directory already exists
    pass
Baxter_Cu = outputfolder + '/Baxter/'                      
            

print('Processing the ' + available_elements[1] + ' Isotope System')
time.sleep(0.1)
print()
print('#############')
print()
print('Start with outlier correction')

time_outl_start = time.perf_counter()
entries = Path(inf)
#print(entries)
if __name__ == '__main__':
    process.multiprocessCu()
print()
print('#############')
print()
print('Outlier correction finished')
time_outl_ende = time.perf_counter()       
print()
print('#############')
print()
print("The outlier correction took " "{:1.5f} seconds to complete".format(time_outl_ende - time_outl_start))
print()
print('#############')
print()
"""Insert Header"""
outname = os.path.join(outfile_results_Cu, 'Cu_header.csv')
fullname = os.path.join(outfile_results_Cu, 'Cu_export.csv')
reread = pd.read_csv(fullname, sep='\t', names = ['Time', 'Filename', '60Ni', '61Ni', '62Ni', '63Cu', '64Ni', '65Cu', '66Zn', '62Ni/60Ni', '65Cu/63Cu', '65Cu/63Cu_corrected'], index_col=False)
reread.sort_values(by=['Filename'], inplace = True)
reread.reset_index(drop=True, inplace = True)
reread.drop(['Time'], axis = 1, inplace = True)
reread.to_csv(fullname, sep = '\t', header=True, index=False)

outputfolderBaxter_Cu = outputfolder + '/Baxter/'
"""Insert Header for Baxter calculation"""
baxter_file = os.path.join(outputfolderBaxter_Cu, 'Baxter.csv')
baxter_header = os.path.join(outputfolderBaxter_Cu, 'Baxter_header.csv')
baxter_header_df = pd.read_csv(baxter_header, nrows=1, sep = '\t')
column_names_baxter = baxter_header_df.columns.values.tolist()
baxter_file_df = pd.read_csv(baxter_file, sep = '\t', names = column_names_baxter, index_col = False)
baxter_file_df.to_csv(baxter_file, sep = '\t', header = True, index = False)



print('Inserted Header')
time.sleep(0.1)

time_ssb_start = time.perf_counter()
print()
print('#############')
print()
print('Calculate d65/63Cu with SSB Method')
resultname = os.path.join(outfile_results_Cu, 'Cu_delta.csv')
ssb.ssbCu()

time_ssb_ende = time.perf_counter()
print()
print('#############')
print()
print("The ssb correction took " "{:1.5f} seconds to complete".format(time_ssb_ende - time_ssb_start))
print()
print('#############')
print()
print('#############')   

print('Now I additional calculate with Baxter')
time.sleep(0.1)
time_baxter_start = time.perf_counter()
baxterCu.baxterCu()
time_baxter_ende = time.perf_counter()
print('')
print("##  The Baxter calculation took " "{:1.3f} seconds to complete  ##".format(time_baxter_ende - time_baxter_start))
print()
print('#############')
print()
print()
print('Calculate d65Cu and d61Ni of Standards')
print()
plotdCustd.dCu_Standards()

print()
print('Processing of ' + available_elements[1] + ' Isotope System finished')
print('File "'+ outname +'" contains the raw isotope values')
print('File "'+ resultname +'" contains the d65/63Cu values calculated with SSB')

print()
print('#############')
print()
print('Calculating dNi and dCu of Standards')

plotdCustd.dCu_Standards()
 

ende = time.perf_counter()
print()
print('#############')
print("The function took " "{:1.3f} seconds to complete".format(ende - start))



