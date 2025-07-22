# -*- coding: utf-8 -*-
"""
This module contains the standard configuration

@author: Andreas Wittke
"""
# Import Modules
from datetime import date
import os
# get current date
today = date.today()
today_dmy = today.strftime("%Y.%m.%d")
today_year = today.strftime("%Y")
# set Workspace Folders
WorkspaceVariableInput = 'input/' + today_dmy 
WorkspaceVariableOutput = 'output/' 

outfile_results_Cu = WorkspaceVariableOutput + '/Cu/' + today_year + '/' + today_dmy + '/results'
outfile_plt_Cu = WorkspaceVariableOutput + 'Cu/' + today_year + '/' + today_dmy + '/plots/'
outfile_corrected_raw_data_Cu = WorkspaceVariableOutput + '/Cu/' + today_year + '/' + today_dmy + '/corrected_raw_data/'
exportCu = outfile_corrected_raw_data_Cu
outfile_plt_Cu_Std = outfile_plt_Cu + 'Standards'
Baxter_Cu = WorkspaceVariableOutput + '/Cu/' + today_year + '/' + today_dmy + '/Baxter/'

outfile_results_Sn = WorkspaceVariableOutput + '/Sn/' + today_year + '/' + today_dmy + '/results'
outfile_plt_Sn = WorkspaceVariableOutput + 'Sn/' + today_year + '/' + today_dmy + '/plots/'
outfile_corrected_raw_data_Sn = WorkspaceVariableOutput + 'Sn/' + today_year + '/' + today_dmy + '/corrected_raw_data/'
Baxter_Sn = WorkspaceVariableOutput + '/Sn/' + today_year + '/' + today_dmy + '/Baxter/'
exportSn = outfile_corrected_raw_data_Sn

outfile_results_Ag = WorkspaceVariableOutput + '/Ag/' + today_year + '/' + today_dmy + '/results'
outfile_plt_Ag = WorkspaceVariableOutput + '/Ag/' + today_year + '/' + today_dmy + '/plots/'
outfile_corrected_raw_data_Ag = WorkspaceVariableOutput + '/Ag/' + today_year + '/' + today_dmy + '/corrected_raw_data/'
exportAg = outfile_corrected_raw_data_Ag

outfile_results_Li = WorkspaceVariableOutput + '/Li/' + today_year + '/' + today_dmy + '/results'
outfile_plt_Li = WorkspaceVariableOutput + '/Li/' + today_year + '/' + today_dmy + '/plots/'
outfile_corrected_raw_data_Li = WorkspaceVariableOutput + '/Li/' + today_year + '/' + today_dmy + '/corrected_raw_data/'
outfile_plt_Li_Std = outfile_plt_Li + 'Standards'
exportLi = outfile_corrected_raw_data_Li
Baxter_Li = WorkspaceVariableOutput + '/Li/' + today_year + '/' + today_dmy + '/Baxter/'


# Available Elements
available_elements = ['Ca', 'Cu', 'Sn', 'Ag', 'Li']

database = 'input' + '/database/' + 'masses.csv'