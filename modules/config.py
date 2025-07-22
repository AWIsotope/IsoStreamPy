# -*- coding: utf-8 -*-
"""
This module contains the standard configuration

@author: Andreas Wittke
"""
# Import Modules
from datetime import date
import os
import pickle

# get current date
today = date.today()
today_dmy = today.strftime("%Y.%m.%d")
today_year = today.strftime("%Y")
# set Workspace Folders
WorkspaceVariableInput = ''
WorkspaceVariableOutput = 'output' 

##Sn
file_path = os.path.join("modules", "variable_data.pkl")  ## Gibt aus: "output/" + option + '/' + today_year + '/' + ordnername + "/results"  
file_path2 = os.path.join("modules", "variable_data_outputfolder.pkl") ## Gibt aus: "output/" + option + '/' + today_year + '/' + ordnername
# Variable mit pickle laden
with open(file_path, 'rb') as file:
    standardoutputpfad = pickle.load(file)

with open(file_path2, 'rb') as file2:
    outputfolder = pickle.load(file2)

outfile_results_Sn = standardoutputpfad# outputfolder + '/results'
outfile_plt_Sn = outputfolder + '/plots/'
outfile_corrected_raw_data_Sn = outputfolder + '/corrected_raw_data/'
Baxter_Sn = outputfolder + '/Baxter/'
exportSn = outfile_corrected_raw_data_Sn
outfile_plt_Sn_Std = outfile_plt_Sn + 'Standards'

outfile_results_Cu = outputfolder + '/results'
outfile_plt_Cu = outputfolder + '/plots/'
outfile_corrected_raw_data_Cu = outputfolder + '/corrected_raw_data/'
Baxter_Cu = outputfolder + '/Baxter/'
exportCu = outfile_corrected_raw_data_Cu
outfile_plt_Cu_Std = outfile_plt_Cu + 'Standards'

## Ag
outfile_results_Ag = WorkspaceVariableOutput + '/Ag/' + today_year + '/' + today_dmy + '/results'
outfile_plt_Ag = WorkspaceVariableOutput + '/Ag/' + today_year + '/' + today_dmy + '/plots/'
outfile_corrected_raw_data_Ag = WorkspaceVariableOutput + '/Ag/' + today_year + '/' + today_dmy + '/corrected_raw_data/'
exportAg = outfile_corrected_raw_data_Ag


## Li
outfile_results_Li = WorkspaceVariableOutput + '/Li/' + today_year + '/' + today_dmy + '/results'
outfile_plt_Li = WorkspaceVariableOutput + '/Li/' + today_year + '/' + today_dmy + '/plots/'
outfile_corrected_raw_data_Li = WorkspaceVariableOutput + '/Li/' + today_year + '/' + today_dmy + '/corrected_raw_data/'
outfile_plt_Li_Std = outfile_plt_Li + 'Standards'
exportLi = outfile_corrected_raw_data_Li
Baxter_Li = WorkspaceVariableOutput + '/Li/' + today_year + '/' + today_dmy + '/Baxter/'


##
outfile_results_Sb = outputfolder + '/results'
outfile_plt_Sb = outputfolder + '/plots/'
outfile_corrected_raw_data_Sb = outputfolder + '/corrected_raw_data/'
Baxter_Sb = outputfolder + '/Baxter/'
exportSb = outfile_corrected_raw_data_Sb
outfile_plt_Sb_Std = outfile_plt_Sb + 'Standards'



# Available Elements
available_elements = ['Ca', 'Cu', 'Sn', 'Ag', 'Li', 'Sb']

database = 'input' + '/database/' + 'masses.csv'

