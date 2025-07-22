"""
@author: Andreas Wittke
"""
import os
import pandas as pd
from csv import DictReader
import multiprocessing as mp
from pathlib import Path
import glob
import pickle
import time
import warnings
import sys
from datetime import date
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append('')

today = date.today()
today_dmy = today.strftime("%Y.%m.%d")
today_year = today.strftime("%Y")

# load individual modules
WorkspaceVariableInput = 'input/' + today_dmy 
Folder = Path(WorkspaceVariableInput)

file_path = os.path.join("modules", "variable_data.pkl")

# Variable mit pickle laden
with open(file_path, 'rb') as file:
    standardoutputpfad = pickle.load(file)

file_path2 = os.path.join("modules", "variable_data_input.pkl")
# Variable mit pickle laden
with open(file_path2, 'rb') as file2:
    standardinputpfad = pickle.load(file2)

# Konvertiere standardinputpfad zu einem Path-Objekt
standardinputpfad = Path(standardinputpfad)

def load_files(path, extension):
    return sorted([f for f in os.listdir(path) if f.endswith(extension)])

def corruptdata_Sn(file):
    
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    df = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=90, index_col=False, dtype=float)
    value = df['120Sn'].mean()
    
    for row in df.index:
        if value > 5:
            pass
        elif value < 0.0004:
            print(f'{file} contains corrupt data!')
            base = os.path.splitext(file)[0]
            os.rename(file, base + ".corrupted")
            break

def select_data_Sn(file):
    for file in standardinputpfad.iterdir():
        if file.suffix == '.TXT' and 'Blk' not in file.stem:
            corruptdata_Sn(file)

def multiprocess_corruptdata_Sn():
    data = sorted(standardinputpfad.glob('*.TXT'))
    with mp.Pool() as p:
        for res in p.imap_unordered(corruptdata_Sn, data):
            print(res)

def corruptdata_Cu(file):
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    df = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=90, index_col=False, dtype=float)
    value = df['63Cu'].mean()
    
    for row in df.index:
        if value > 5:
            pass
        elif value < 0.001:
            print(f'{file} contains corrupt data!')
            base = os.path.splitext(file)[0]
            os.rename(file, base + ".corrupted")
            break

def select_data_Cu():
    for file in standardinputpfad.iterdir():
        if file.suffix == '.TXT' and 'Blk' not in file.stem:
            corruptdata_Cu(file)

def multiprocess_corruptdata_Cu():
    data = sorted(standardinputpfad.glob('*.TXT'))
    with mp.Pool() as p:
        for res in p.imap_unordered(corruptdata_Cu, data):
            print(res)

def corruptdata_Li(file):
    col_names = DictReader(open(file, 'r'), delimiter='\t').fieldnames
    col_names.remove('Trace for Mass:')
    col_names.insert(0, 'Time')
    df = pd.read_csv(file, sep='\t', names=col_names, skiprows=6, nrows=90, index_col=False, dtype=float)
    value = df['7Li'].mean()
    
    for row in df.index:
        if value > 2:
            pass
        elif value < 0.001:
            print(f'{file} contains corrupt data!')
            base = os.path.splitext(file)[0]
            os.rename(file, base + ".corrupted")
            break

def select_data_Li():
    for file in standardinputpfad.iterdir():
        if file.suffix == '.TXT' and 'Blk' not in file.stem:
            corruptdata_Li(file)

def multiprocess_corruptdata_Li():
    data = sorted(standardinputpfad.glob('*.TXT'))
    with mp.Pool() as p:
        for res in p.imap_unordered(corruptdata_Li, data):
            print(res)
