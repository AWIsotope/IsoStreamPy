"""
@author: Andreas Wittke
"""
import pandas as pd
import os
import numpy as np
import warnings
from scipy import stats
from statistics import mean
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pickle
from modules.config import outfile_results_Cu, outfile_results_Sn, outfile_results_Ag, Baxter_Sn
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from modules.config import WorkspaceVariableInput
file_path2 = os.path.join("modules", "variable_data_input.pkl")
# Variable mit pickle laden
with open(file_path2, 'rb') as file2:
    standardinputpfad = pickle.load(file2)

infile = standardinputpfad
print(infile)

file_path = os.path.join("modules", "variable_data_outputfolder.pkl") ## Gibt aus: "output/" + option + '/' + today_year + '/' + ordnername
with open(file_path, 'rb') as file2:
    outputfolder = pickle.load(file2)

pd.options.mode.chained_assignment = None
pd.set_option("display.precision", 15)
pd.options.display.float_format = '{:.15f}'.format
os.getcwd()
resultnameSn_Baxter = os.path.join(outfile_results_Sn, 'Sn_delta_Baxter.csv') 
resultnameSnExcel_Baxter = os.path.join(outfile_results_Sn, 'Sn_delta_Baxter.xlsx')

outputfolderBaxter_Sn = outputfolder + '/Baxter/'

def baxterSn():
    global resultnameSn_Baxter
    global resultnameSnExcel_Baxter
    global infile
    baxter = pd.read_csv(outputfolderBaxter_Sn + 'Baxter.csv', sep = '\t', engine = 'python', index_col = False)
    baxter = baxter.drop(baxter.columns[[0]], axis = 1)
    
    baxter.drop(list(baxter.filter(regex = '_corrected')), axis = 1, inplace = True)
    
    baxter['standard/'] = baxter['Inputfile'].str.contains('_Nis|_Nis1|_Ni0')
    baxter['sample/'] = ~baxter['standard/']
    
    baxterStandard=baxter[baxter['standard/'] == True]
    double = pd.concat([baxterStandard]*2,ignore_index=True)
    baxter = pd.concat([baxter, double], ignore_index=True)
    baxter['standard/'] = baxter['Inputfile'].str.contains('_Nis|_Nis1|_Ni0')
    baxter.sort_values('Inputfile',inplace=True)
    baxter.reset_index(inplace=True)
    baxter = baxter.iloc[: , 1:]
    baxterStandard.sort_values('Inputfile',inplace=True)
    baxterStandard.reset_index(inplace=True)
    baxterStandardall = baxter[baxter['standard/'] == True]
    baxterSample = baxter[baxter['sample/'] == True]
    baxterSample['Inputfile'] = baxterSample['Inputfile'].map(lambda x: x.rstrip('_123'))

    """from Lee et al. 1995"""
    R_X_RM_116Sn_120Sn = 0.44600182836668
    R_X_RM_117Sn_120Sn = 0.235331652217662
    R_X_RM_118Sn_120Sn = 0.742936553222036
    R_X_RM_119Sn_120Sn = 0.263413646487143
    R_X_RM_122Sn_116Sn = 0.318574818585136
    R_X_RM_124Sn_116Sn = 0.398177253499329
    R_X_RM_122Sn_120Sn = 0.142084951560554
    R_X_RM_124Sn_120Sn = 0.177587783074724
    R_X_RM_117Sn_119Sn = 0.893392029441222
    
    
    U_X_RM_116Sn_120Sn = R_X_RM_116Sn_120Sn*np.sqrt((0.0003/0.145385)**2+(0.0006/0.325974)**2)/2 # 0.0006166
    U_X_RM_117Sn_120Sn = R_X_RM_117Sn_120Sn*np.sqrt((0.0006/0.076712)**2+(0.0006/0.325974)**2)/2
    U_X_RM_118Sn_120Sn = R_X_RM_118Sn_120Sn*np.sqrt((0.0024/0.242178)**2+(0.0006/0.325974)**2)/2
    U_X_RM_119Sn_120Sn = R_X_RM_119Sn_120Sn*np.sqrt((0.0003/0.085866)**2+(0.0006/0.325974)**2)/2
    U_X_RM_122Sn_116Sn = R_X_RM_122Sn_116Sn*np.sqrt((0.0004/0.046316)**2+(0.0003/0.145385)**2)/2
    U_X_RM_124Sn_116Sn = R_X_RM_124Sn_116Sn*np.sqrt((0.0017/0.057889)**2+(0.0003/0.145385)**2)/2
    U_X_RM_122Sn_120Sn = R_X_RM_122Sn_120Sn*np.sqrt((0.0004/0.046316)**2+(0.0006/0.325974)**2)/2
    U_X_RM_124Sn_120Sn = R_X_RM_124Sn_120Sn*np.sqrt((0.0017/0.057889)**2+(0.0006/0.325974)**2)/2
    U_X_RM_117Sn_119Sn = R_X_RM_117Sn_119Sn*np.sqrt((0.0006/0.076712)**2+(0.0003/0.085866)**2)/2
    """" End of insertion of Lee et al. 1995 """

    masses = pd.DataFrame()
    masses.loc[0, '121Sb'] = 120.903810584
    masses.loc[0, '123Sb'] = 122.904211755
    masses.loc[0, '112Sn'] = 111.904821807
    masses.loc[0, '114Sn'] = 113.902780398
    masses.loc[0, '115Sn'] = 114.903344699
    masses.loc[0, '116Sn'] = 115.901742835
    masses.loc[0, '117Sn'] = 116.902954032
    masses.loc[0, '118Sn'] = 117.901606625
    masses.loc[0, '119Sn'] = 118.90331124
    masses.loc[0, '120Sn'] = 119.902202063
    masses.loc[0, '122Sn'] = 121.903442607
    masses.loc[0, '124Sn'] = 123.905273581
    
    abundances = pd.DataFrame()
    abundances.loc[0, '121Sb'] = 0.5721
    abundances.loc[0, '121Sb_error'] = 0.00005
    abundances.loc[0, '123Sb'] = 0.4279
    abundances.loc[0, '123Sb_error'] = 0.00005
    abundances.loc[0, '112Sn'] = 0.009718
    abundances.loc[0, '112Sn_error'] = 0.0001
    abundances.loc[0, '114Sn'] = 0.006583
    abundances.loc[0, '114Sn_error'] = 0.00001
    abundances.loc[0, '115Sn'] = 0.003379
    abundances.loc[0, '115Sn_error'] = 0.00001
    abundances.loc[0, '116Sn'] = 0.145385
    abundances.loc[0, '116Sn_error'] = 0.00009
    abundances.loc[0, '117Sn'] = 0.076712
    abundances.loc[0, '117Sn_error'] = 0.00007
    abundances.loc[0, '118Sn'] = 0.242178
    abundances.loc[0, '118Sn_error'] = 0.00009
    abundances.loc[0, '119Sn'] = 0.085866
    abundances.loc[0, '119Sn_error'] = 0.00004
    abundances.loc[0, '120Sn'] = 0.325974
    abundances.loc[0, '120Sn_error'] = 0.00009
    abundances.loc[0, '122Sn'] = 0.046316
    abundances.loc[0, '122Sn_error'] = 0.00003
    abundances.loc[0, '124Sn'] = 0.057889
    abundances.loc[0, '124Sn_error'] = 0.00005
    
    R_is_rm = abundances['123Sb'] / abundances['121Sb']
    
    n = 0
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['ln_r_is'] = np.log(baxterStandardall['123Sb/121Sb'])
        baxterStandardall['ln_r_x_rm_116Sn_120Sn'] = np.log(baxterStandardall['116Sn/120Sn'])
        baxterStandardall['ln_r_x_rm_117Sn_120Sn'] = np.log(baxterStandardall['117Sn/120Sn'])
        baxterStandardall['ln_r_x_rm_118Sn_120Sn'] = np.log(baxterStandardall['118Sn/120Sn'])
        baxterStandardall['ln_r_x_rm_119Sn_120Sn'] = np.log(baxterStandardall['119Sn/120Sn'])
        baxterStandardall['ln_r_x_rm_122Sn_116Sn'] = np.log(baxterStandardall['122Sn/116Sn'])
        baxterStandardall['ln_r_x_rm_124Sn_116Sn'] = np.log(baxterStandardall['124Sn/116Sn'])
        baxterStandardall['ln_r_x_rm_122Sn_120Sn'] = np.log(baxterStandardall['122Sn/120Sn'])
        baxterStandardall['ln_r_x_rm_124Sn_120Sn'] = np.log(baxterStandardall['124Sn/120Sn'])
        baxterStandardall['ln_r_x_rm_117Sn_119Sn'] = np.log(baxterStandardall['117Sn/119Sn'])

    
    nRM_Standard = len(baxterStandardall.index)
    
    m_is_block_Standard = nRM_Standard
    m_x_block_Standard = nRM_Standard
    
    
    ##X_mean // remains the same for all masses! Küzel bedeuten "Zelle xxx in Baxter-Spreadsheet 116-120"
    x_mean_sum_ln_123Sb_121Sb_Standard = sum(baxterStandardall['ln_r_is'])  ## = AE8
    ln_r_is_123Sb_121Sb_Standard = x_mean_sum_ln_123Sb_121Sb_Standard / nRM_Standard   ## AE4
    p_is_mean_123Sb_121Sb_Standard = np.exp(ln_r_is_123Sb_121Sb_Standard) ## = AE5
    
    
    ## Y_mean // changes for every mass ratio
    y_mean_sum__ln116Sn_120Sn_Standard = sum(baxterStandardall['ln_r_x_rm_116Sn_120Sn']) ## = AF8
    y_mean_sum__ln117Sn_120Sn_Standard = sum(baxterStandardall['ln_r_x_rm_117Sn_120Sn'])
    y_mean_sum__ln118Sn_120Sn_Standard = sum(baxterStandardall['ln_r_x_rm_118Sn_120Sn'])
    y_mean_sum__ln119Sn_120Sn_Standard = sum(baxterStandardall['ln_r_x_rm_119Sn_120Sn'])
    y_mean_sum__ln122Sn_116Sn_Standard = sum(baxterStandardall['ln_r_x_rm_122Sn_116Sn'])
    y_mean_sum__ln124Sn_116Sn_Standard = sum(baxterStandardall['ln_r_x_rm_124Sn_116Sn'])
    y_mean_sum__ln122Sn_120Sn_Standard = sum(baxterStandardall['ln_r_x_rm_122Sn_120Sn'])
    y_mean_sum__ln124Sn_120Sn_Standard = sum(baxterStandardall['ln_r_x_rm_124Sn_120Sn'])
    y_mean_sum__ln117Sn_119Sn_Standard = sum(baxterStandardall['ln_r_x_rm_117Sn_119Sn'])
    
    y_mean_stabw_Standard = np.std(baxterStandardall['123Sb/121Sb']) ## AF5
    y_mean_stabw_1SE_Standard= np.std(baxterStandardall['123Sb/121Sb_1SE']) ## AF6
    
    
    ln_r_x_rm_116Sn_120Sn_Standard = y_mean_sum__ln116Sn_120Sn_Standard / nRM_Standard   ## AF4
    ln_r_x_rm_117Sn_120Sn_Standard = y_mean_sum__ln117Sn_120Sn_Standard / nRM_Standard
    ln_r_x_rm_118Sn_120Sn_Standard = y_mean_sum__ln118Sn_120Sn_Standard / nRM_Standard
    ln_r_x_rm_119Sn_120Sn_Standard = y_mean_sum__ln119Sn_120Sn_Standard / nRM_Standard
    ln_r_x_rm_122Sn_116Sn_Standard = y_mean_sum__ln122Sn_116Sn_Standard / nRM_Standard
    ln_r_x_rm_124Sn_116Sn_Standard = y_mean_sum__ln124Sn_116Sn_Standard / nRM_Standard
    ln_r_x_rm_122Sn_120Sn_Standard = y_mean_sum__ln122Sn_120Sn_Standard / nRM_Standard
    ln_r_x_rm_124Sn_120Sn_Standard = y_mean_sum__ln124Sn_120Sn_Standard / nRM_Standard
    ln_r_x_rm_117Sn_119Sn_Standard = y_mean_sum__ln117Sn_119Sn_Standard / nRM_Standard
    
    
    
    ### X_mean
    p_x_rm_mean_116Sn_120Sn_Standard = np.exp(ln_r_x_rm_116Sn_120Sn_Standard) ## = AE6
    p_x_rm_mean_117Sn_120Sn_Standard = np.exp(ln_r_x_rm_117Sn_120Sn_Standard) 
    p_x_rm_mean_118Sn_120Sn_Standard = np.exp(ln_r_x_rm_118Sn_120Sn_Standard) 
    p_x_rm_mean_119Sn_120Sn_Standard = np.exp(ln_r_x_rm_119Sn_120Sn_Standard) 
    p_x_rm_mean_122Sn_116Sn_Standard = np.exp(ln_r_x_rm_122Sn_116Sn_Standard) 
    p_x_rm_mean_124Sn_116Sn_Standard = np.exp(ln_r_x_rm_124Sn_116Sn_Standard) 
    p_x_rm_mean_122Sn_120Sn_Standard = np.exp(ln_r_x_rm_122Sn_120Sn_Standard) 
    p_x_rm_mean_124Sn_120Sn_Standard = np.exp(ln_r_x_rm_124Sn_120Sn_Standard) 
    p_x_rm_mean_117Sn_119Sn_Standard = np.exp(ln_r_x_rm_117Sn_119Sn_Standard) 
    
    
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['Xi-Xmean_116Sn_120Sn'] = baxterStandardall['ln_r_is'] - ln_r_is_123Sb_121Sb_Standard  ## column AH from row 12 down
        baxterStandardall['Yi-Ymean_116Sn_120Sn'] = baxterStandardall['ln_r_x_rm_116Sn_120Sn'] - ln_r_x_rm_116Sn_120Sn_Standard ## column AI from row 12 down
        baxterStandardall['Xi-Xmean_117Sn_120Sn'] = baxterStandardall['ln_r_is'] - ln_r_is_123Sb_121Sb_Standard
        baxterStandardall['Yi-Ymean_117Sn_120Sn'] = baxterStandardall['ln_r_x_rm_117Sn_120Sn'] - ln_r_x_rm_117Sn_120Sn_Standard 
        baxterStandardall['Xi-Xmean_118Sn_120Sn'] = baxterStandardall['ln_r_is'] - ln_r_is_123Sb_121Sb_Standard
        baxterStandardall['Yi-Ymean_118Sn_120Sn'] = baxterStandardall['ln_r_x_rm_118Sn_120Sn'] - ln_r_x_rm_118Sn_120Sn_Standard 
        baxterStandardall['Xi-Xmean_119Sn_120Sn'] = baxterStandardall['ln_r_is'] - ln_r_is_123Sb_121Sb_Standard
        baxterStandardall['Yi-Ymean_119Sn_120Sn'] = baxterStandardall['ln_r_x_rm_119Sn_120Sn'] - ln_r_x_rm_119Sn_120Sn_Standard 
        baxterStandardall['Xi-Xmean_122Sn_116Sn'] = baxterStandardall['ln_r_is'] - ln_r_is_123Sb_121Sb_Standard
        baxterStandardall['Yi-Ymean_122Sn_116Sn'] = baxterStandardall['ln_r_x_rm_122Sn_116Sn'] - ln_r_x_rm_122Sn_116Sn_Standard 
        baxterStandardall['Xi-Xmean_124Sn_116Sn'] = baxterStandardall['ln_r_is'] - ln_r_is_123Sb_121Sb_Standard
        baxterStandardall['Yi-Ymean_124Sn_116Sn'] = baxterStandardall['ln_r_x_rm_124Sn_116Sn'] - ln_r_x_rm_124Sn_116Sn_Standard
        baxterStandardall['Xi-Xmean_122Sn_120Sn'] = baxterStandardall['ln_r_is'] - ln_r_is_123Sb_121Sb_Standard
        baxterStandardall['Yi-Ymean_122Sn_120Sn'] = baxterStandardall['ln_r_x_rm_122Sn_120Sn'] - ln_r_x_rm_122Sn_120Sn_Standard 
        baxterStandardall['Xi-Xmean_124Sn_120Sn'] = baxterStandardall['ln_r_is'] - ln_r_is_123Sb_121Sb_Standard
        baxterStandardall['Yi-Ymean_124Sn_120Sn'] = baxterStandardall['ln_r_x_rm_124Sn_120Sn'] - ln_r_x_rm_124Sn_120Sn_Standard  
        baxterStandardall['Xi-Xmean_117Sn_119Sn'] = baxterStandardall['ln_r_is'] - ln_r_is_123Sb_121Sb_Standard
        baxterStandardall['Yi-Ymean_117Sn_119Sn'] = baxterStandardall['ln_r_x_rm_117Sn_119Sn'] - ln_r_x_rm_117Sn_119Sn_Standard 
    
        baxterStandardall['xiyi_116Sn_120Sn'] = baxterStandardall['Xi-Xmean_116Sn_120Sn'] * baxterStandardall['Yi-Ymean_116Sn_120Sn']  # column AL
        baxterStandardall['xiyi_117Sn_120Sn'] = baxterStandardall['Xi-Xmean_117Sn_120Sn'] * baxterStandardall['Yi-Ymean_117Sn_120Sn']
        baxterStandardall['xiyi_118Sn_120Sn'] = baxterStandardall['Xi-Xmean_118Sn_120Sn'] * baxterStandardall['Yi-Ymean_118Sn_120Sn']
        baxterStandardall['xiyi_119Sn_120Sn'] = baxterStandardall['Xi-Xmean_119Sn_120Sn'] * baxterStandardall['Yi-Ymean_119Sn_120Sn']
        baxterStandardall['xiyi_122Sn_116Sn'] = baxterStandardall['Xi-Xmean_122Sn_116Sn'] * baxterStandardall['Yi-Ymean_122Sn_116Sn']
        baxterStandardall['xiyi_124Sn_116Sn'] = baxterStandardall['Xi-Xmean_124Sn_116Sn'] * baxterStandardall['Yi-Ymean_124Sn_116Sn']
        baxterStandardall['xiyi_122Sn_120Sn'] = baxterStandardall['Xi-Xmean_122Sn_120Sn'] * baxterStandardall['Yi-Ymean_122Sn_120Sn']
        baxterStandardall['xiyi_124Sn_120Sn'] = baxterStandardall['Xi-Xmean_124Sn_120Sn'] * baxterStandardall['Yi-Ymean_124Sn_120Sn']
        baxterStandardall['xiyi_117Sn_119Sn'] = baxterStandardall['Xi-Xmean_117Sn_119Sn'] * baxterStandardall['Yi-Ymean_117Sn_119Sn']
    
        baxterStandardall['xi^2_116Sn_120Sn'] = baxterStandardall['Xi-Xmean_116Sn_120Sn'] * baxterStandardall['Xi-Xmean_116Sn_120Sn'] # column AM
        baxterStandardall['xi^2_117Sn_120Sn'] = baxterStandardall['Xi-Xmean_117Sn_120Sn'] * baxterStandardall['Xi-Xmean_117Sn_120Sn']
        baxterStandardall['xi^2_118Sn_120Sn'] = baxterStandardall['Xi-Xmean_118Sn_120Sn'] * baxterStandardall['Xi-Xmean_118Sn_120Sn']
        baxterStandardall['xi^2_119Sn_120Sn'] = baxterStandardall['Xi-Xmean_119Sn_120Sn'] * baxterStandardall['Xi-Xmean_119Sn_120Sn']
        baxterStandardall['xi^2_122Sn_116Sn'] = baxterStandardall['Xi-Xmean_122Sn_116Sn'] * baxterStandardall['Xi-Xmean_122Sn_116Sn']
        baxterStandardall['xi^2_124Sn_116Sn'] = baxterStandardall['Xi-Xmean_124Sn_116Sn'] * baxterStandardall['Xi-Xmean_124Sn_116Sn']
        baxterStandardall['xi^2_122Sn_120Sn'] = baxterStandardall['Xi-Xmean_122Sn_120Sn'] * baxterStandardall['Xi-Xmean_122Sn_120Sn']
        baxterStandardall['xi^2_124Sn_120Sn'] = baxterStandardall['Xi-Xmean_124Sn_120Sn'] * baxterStandardall['Xi-Xmean_124Sn_120Sn']
        baxterStandardall['xi^2_117Sn_119Sn'] = baxterStandardall['Xi-Xmean_117Sn_119Sn'] * baxterStandardall['Xi-Xmean_117Sn_119Sn']
    
        baxterStandardall['yi^2_116Sn_120Sn'] = baxterStandardall['Yi-Ymean_116Sn_120Sn'] * baxterStandardall['Yi-Ymean_116Sn_120Sn'] # Column AN
        baxterStandardall['yi^2_117Sn_120Sn'] = baxterStandardall['Yi-Ymean_117Sn_120Sn'] * baxterStandardall['Yi-Ymean_117Sn_120Sn']
        baxterStandardall['yi^2_118Sn_120Sn'] = baxterStandardall['Yi-Ymean_118Sn_120Sn'] * baxterStandardall['Yi-Ymean_118Sn_120Sn']
        baxterStandardall['yi^2_119Sn_120Sn'] = baxterStandardall['Yi-Ymean_119Sn_120Sn'] * baxterStandardall['Yi-Ymean_119Sn_120Sn']
        baxterStandardall['yi^2_122Sn_116Sn'] = baxterStandardall['Yi-Ymean_122Sn_116Sn'] * baxterStandardall['Yi-Ymean_122Sn_116Sn']
        baxterStandardall['yi^2_124Sn_116Sn'] = baxterStandardall['Yi-Ymean_124Sn_116Sn'] * baxterStandardall['Yi-Ymean_124Sn_116Sn']
        baxterStandardall['yi^2_122Sn_120Sn'] = baxterStandardall['Yi-Ymean_122Sn_120Sn'] * baxterStandardall['Yi-Ymean_122Sn_120Sn']
        baxterStandardall['yi^2_124Sn_120Sn'] = baxterStandardall['Yi-Ymean_124Sn_120Sn'] * baxterStandardall['Yi-Ymean_124Sn_120Sn']
        baxterStandardall['yi^2_117Sn_119Sn'] = baxterStandardall['Yi-Ymean_117Sn_119Sn'] * baxterStandardall['Yi-Ymean_117Sn_119Sn']
    
    sum_xiyi_116Sn_120Sn = sum(baxterStandardall['xiyi_116Sn_120Sn']) # AL8
    sum_xiyi_117Sn_120Sn = sum(baxterStandardall['xiyi_117Sn_120Sn'])
    sum_xiyi_118Sn_120Sn = sum(baxterStandardall['xiyi_118Sn_120Sn'])
    sum_xiyi_119Sn_120Sn = sum(baxterStandardall['xiyi_119Sn_120Sn'])
    sum_xiyi_122Sn_116Sn = sum(baxterStandardall['xiyi_122Sn_116Sn'])
    sum_xiyi_124Sn_116Sn = sum(baxterStandardall['xiyi_124Sn_116Sn'])
    sum_xiyi_122Sn_120Sn = sum(baxterStandardall['xiyi_122Sn_120Sn'])
    sum_xiyi_124Sn_120Sn = sum(baxterStandardall['xiyi_124Sn_120Sn'])
    sum_xiyi_117Sn_119Sn = sum(baxterStandardall['xiyi_117Sn_119Sn'])
    
    sum_xi_exp_2_116Sn_120Sn = sum(baxterStandardall['xi^2_116Sn_120Sn']) # AM8
    sum_xi_exp_2_117Sn_120Sn = sum(baxterStandardall['xi^2_117Sn_120Sn'])
    sum_xi_exp_2_118Sn_120Sn = sum(baxterStandardall['xi^2_118Sn_120Sn'])
    sum_xi_exp_2_119Sn_120Sn = sum(baxterStandardall['xi^2_119Sn_120Sn'])
    sum_xi_exp_2_122Sn_116Sn = sum(baxterStandardall['xi^2_122Sn_116Sn'])
    sum_xi_exp_2_124Sn_116Sn = sum(baxterStandardall['xi^2_124Sn_116Sn'])
    sum_xi_exp_2_122Sn_120Sn = sum(baxterStandardall['xi^2_122Sn_120Sn'])
    sum_xi_exp_2_124Sn_120Sn = sum(baxterStandardall['xi^2_124Sn_120Sn'])
    sum_xi_exp_2_117Sn_119Sn = sum(baxterStandardall['xi^2_117Sn_119Sn'])
    
    sum_yi_exp_2_116Sn_120Sn = sum(baxterStandardall['yi^2_116Sn_120Sn']) # AN8
    sum_yi_exp_2_117Sn_120Sn = sum(baxterStandardall['yi^2_117Sn_120Sn'])
    sum_yi_exp_2_118Sn_120Sn = sum(baxterStandardall['yi^2_118Sn_120Sn'])
    sum_yi_exp_2_119Sn_120Sn = sum(baxterStandardall['yi^2_119Sn_120Sn'])
    sum_yi_exp_2_122Sn_116Sn = sum(baxterStandardall['yi^2_122Sn_116Sn'])
    sum_yi_exp_2_124Sn_116Sn = sum(baxterStandardall['yi^2_124Sn_116Sn'])
    sum_yi_exp_2_122Sn_120Sn = sum(baxterStandardall['yi^2_122Sn_120Sn'])
    sum_yi_exp_2_124Sn_120Sn = sum(baxterStandardall['yi^2_124Sn_120Sn'])
    sum_yi_exp_2_117Sn_119Sn = sum(baxterStandardall['yi^2_117Sn_119Sn'])
    
    
    
    b_hat_116Sn_120Sn = sum_xiyi_116Sn_120Sn / sum_xi_exp_2_116Sn_120Sn # AM4
    b_hat_117Sn_120Sn = sum_xiyi_117Sn_120Sn / sum_xi_exp_2_117Sn_120Sn
    b_hat_118Sn_120Sn = sum_xiyi_118Sn_120Sn / sum_xi_exp_2_118Sn_120Sn
    b_hat_119Sn_120Sn = sum_xiyi_119Sn_120Sn / sum_xi_exp_2_119Sn_120Sn
    b_hat_122Sn_116Sn = sum_xiyi_122Sn_116Sn / sum_xi_exp_2_122Sn_116Sn
    b_hat_124Sn_116Sn = sum_xiyi_124Sn_116Sn / sum_xi_exp_2_124Sn_116Sn
    b_hat_122Sn_120Sn = sum_xiyi_122Sn_120Sn / sum_xi_exp_2_122Sn_120Sn
    b_hat_124Sn_120Sn = sum_xiyi_124Sn_120Sn / sum_xi_exp_2_124Sn_120Sn
    b_hat_117Sn_119Sn = sum_xiyi_117Sn_119Sn / sum_xi_exp_2_117Sn_119Sn
   
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['yi_hat_116Sn_120Sn'] = baxterStandardall['Xi-Xmean_116Sn_120Sn'] * b_hat_116Sn_120Sn # column AO
        baxterStandardall['yi_hat_117Sn_120Sn'] = baxterStandardall['Xi-Xmean_117Sn_120Sn'] * b_hat_117Sn_120Sn
        baxterStandardall['yi_hat_118Sn_120Sn'] = baxterStandardall['Xi-Xmean_118Sn_120Sn'] * b_hat_118Sn_120Sn
        baxterStandardall['yi_hat_119Sn_120Sn'] = baxterStandardall['Xi-Xmean_119Sn_120Sn'] * b_hat_119Sn_120Sn
        baxterStandardall['yi_hat_122Sn_116Sn'] = baxterStandardall['Xi-Xmean_122Sn_116Sn'] * b_hat_122Sn_116Sn
        baxterStandardall['yi_hat_124Sn_116Sn'] = baxterStandardall['Xi-Xmean_124Sn_116Sn'] * b_hat_124Sn_116Sn
        baxterStandardall['yi_hat_122Sn_120Sn'] = baxterStandardall['Xi-Xmean_122Sn_120Sn'] * b_hat_122Sn_120Sn
        baxterStandardall['yi_hat_124Sn_120Sn'] = baxterStandardall['Xi-Xmean_124Sn_120Sn'] * b_hat_124Sn_120Sn
        baxterStandardall['yi_hat_117Sn_119Sn'] = baxterStandardall['Xi-Xmean_117Sn_119Sn'] * b_hat_117Sn_119Sn 
    
        baxterStandardall['yi-yi_hat^2_116Sn_120Sn'] = (baxterStandardall['Yi-Ymean_116Sn_120Sn'] - baxterStandardall['yi_hat_116Sn_120Sn'])**2
        baxterStandardall['yi-yi_hat^2_117Sn_120Sn'] = (baxterStandardall['Yi-Ymean_117Sn_120Sn'] - baxterStandardall['yi_hat_117Sn_120Sn'])**2
        baxterStandardall['yi-yi_hat^2_118Sn_120Sn'] = (baxterStandardall['Yi-Ymean_118Sn_120Sn'] - baxterStandardall['yi_hat_118Sn_120Sn'])**2
        baxterStandardall['yi-yi_hat^2_119Sn_120Sn'] = (baxterStandardall['Yi-Ymean_119Sn_120Sn'] - baxterStandardall['yi_hat_119Sn_120Sn'])**2
        baxterStandardall['yi-yi_hat^2_122Sn_116Sn'] = (baxterStandardall['Yi-Ymean_122Sn_116Sn'] - baxterStandardall['yi_hat_122Sn_116Sn'])**2
        baxterStandardall['yi-yi_hat^2_124Sn_116Sn'] = (baxterStandardall['Yi-Ymean_124Sn_116Sn'] - baxterStandardall['yi_hat_124Sn_116Sn'])**2
        baxterStandardall['yi-yi_hat^2_122Sn_120Sn'] = (baxterStandardall['Yi-Ymean_122Sn_120Sn'] - baxterStandardall['yi_hat_122Sn_120Sn'])**2
        baxterStandardall['yi-yi_hat^2_124Sn_120Sn'] = (baxterStandardall['Yi-Ymean_124Sn_120Sn'] - baxterStandardall['yi_hat_124Sn_120Sn'])**2
        baxterStandardall['yi-yi_hat^2_117Sn_119Sn'] = (baxterStandardall['Yi-Ymean_117Sn_119Sn'] - baxterStandardall['yi_hat_117Sn_119Sn'])**2
    
    
    sum_yi_yi_hat_square_116Sn_120Sn = sum(baxterStandardall['yi-yi_hat^2_116Sn_120Sn']) # AP8
    sum_yi_yi_hat_square_117Sn_120Sn = sum(baxterStandardall['yi-yi_hat^2_117Sn_120Sn'])
    sum_yi_yi_hat_square_118Sn_120Sn = sum(baxterStandardall['yi-yi_hat^2_118Sn_120Sn'])
    sum_yi_yi_hat_square_119Sn_120Sn = sum(baxterStandardall['yi-yi_hat^2_119Sn_120Sn'])
    sum_yi_yi_hat_square_122Sn_116Sn = sum(baxterStandardall['yi-yi_hat^2_122Sn_116Sn'])
    sum_yi_yi_hat_square_124Sn_116Sn = sum(baxterStandardall['yi-yi_hat^2_124Sn_116Sn'])
    sum_yi_yi_hat_square_122Sn_120Sn = sum(baxterStandardall['yi-yi_hat^2_122Sn_120Sn'])
    sum_yi_yi_hat_square_124Sn_120Sn = sum(baxterStandardall['yi-yi_hat^2_124Sn_120Sn'])
    sum_yi_yi_hat_square_117Sn_119Sn = sum(baxterStandardall['yi-yi_hat^2_117Sn_119Sn'])
    
    S_y_x_116Sn_120Sn = np.sqrt(sum_yi_yi_hat_square_116Sn_120Sn / (nRM_Standard - 1))  ## AM2
    S_y_x_117Sn_120Sn = np.sqrt(sum_yi_yi_hat_square_117Sn_120Sn / (nRM_Standard - 1))
    S_y_x_118Sn_120Sn = np.sqrt(sum_yi_yi_hat_square_118Sn_120Sn / (nRM_Standard - 1))
    S_y_x_119Sn_120Sn = np.sqrt(sum_yi_yi_hat_square_119Sn_120Sn / (nRM_Standard - 1))
    S_y_x_122Sn_116Sn = np.sqrt(sum_yi_yi_hat_square_122Sn_116Sn / (nRM_Standard - 1))
    S_y_x_124Sn_116Sn = np.sqrt(sum_yi_yi_hat_square_124Sn_116Sn / (nRM_Standard - 1))
    S_y_x_122Sn_120Sn = np.sqrt(sum_yi_yi_hat_square_122Sn_120Sn / (nRM_Standard - 1))
    S_y_x_124Sn_120Sn = np.sqrt(sum_yi_yi_hat_square_124Sn_120Sn / (nRM_Standard - 1))
    S_y_x_117Sn_119Sn = np.sqrt(sum_yi_yi_hat_square_117Sn_119Sn / (nRM_Standard - 1))
    
    sum_xi_xmean_square_116Sn_120Sn = sum_xi_exp_2_116Sn_120Sn - sum(baxterStandardall['Xi-Xmean_116Sn_120Sn'])**2 / nRM_Standard # AM3
    sum_xi_xmean_square_117Sn_120Sn = sum_xi_exp_2_117Sn_120Sn - sum(baxterStandardall['Xi-Xmean_117Sn_120Sn'])**2 / nRM_Standard
    sum_xi_xmean_square_118Sn_120Sn = sum_xi_exp_2_118Sn_120Sn - sum(baxterStandardall['Xi-Xmean_118Sn_120Sn'])**2 / nRM_Standard
    sum_xi_xmean_square_119Sn_120Sn = sum_xi_exp_2_119Sn_120Sn - sum(baxterStandardall['Xi-Xmean_119Sn_120Sn'])**2 / nRM_Standard
    sum_xi_xmean_square_122Sn_116Sn = sum_xi_exp_2_122Sn_116Sn - sum(baxterStandardall['Xi-Xmean_122Sn_116Sn'])**2 / nRM_Standard
    sum_xi_xmean_square_124Sn_116Sn = sum_xi_exp_2_124Sn_116Sn - sum(baxterStandardall['Xi-Xmean_124Sn_116Sn'])**2 / nRM_Standard
    sum_xi_xmean_square_122Sn_120Sn = sum_xi_exp_2_122Sn_120Sn - sum(baxterStandardall['Xi-Xmean_122Sn_120Sn'])**2 / nRM_Standard
    sum_xi_xmean_square_124Sn_120Sn = sum_xi_exp_2_124Sn_120Sn - sum(baxterStandardall['Xi-Xmean_124Sn_120Sn'])**2 / nRM_Standard
    sum_xi_xmean_square_117Sn_119Sn = sum_xi_exp_2_117Sn_119Sn - sum(baxterStandardall['Xi-Xmean_117Sn_119Sn'])**2 / nRM_Standard
    
    sm_b_hat_116Sn_120Sn = np.sqrt((S_y_x_116Sn_120Sn**2) / sum_xi_xmean_square_116Sn_120Sn)  # AM5
    sm_b_hat_117Sn_120Sn = np.sqrt((S_y_x_117Sn_120Sn**2) / sum_xi_xmean_square_117Sn_120Sn)
    sm_b_hat_118Sn_120Sn = np.sqrt((S_y_x_118Sn_120Sn**2) / sum_xi_xmean_square_118Sn_120Sn)
    sm_b_hat_119Sn_120Sn = np.sqrt((S_y_x_119Sn_120Sn**2) / sum_xi_xmean_square_119Sn_120Sn)
    sm_b_hat_122Sn_116Sn = np.sqrt((S_y_x_122Sn_116Sn**2) / sum_xi_xmean_square_122Sn_116Sn)
    sm_b_hat_124Sn_116Sn = np.sqrt((S_y_x_124Sn_116Sn**2) / sum_xi_xmean_square_124Sn_116Sn)
    sm_b_hat_122Sn_120Sn = np.sqrt((S_y_x_122Sn_120Sn**2) / sum_xi_xmean_square_122Sn_120Sn)
    sm_b_hat_124Sn_120Sn = np.sqrt((S_y_x_124Sn_120Sn**2) / sum_xi_xmean_square_124Sn_120Sn)
    sm_b_hat_117Sn_119Sn = np.sqrt((S_y_x_117Sn_119Sn**2) / sum_xi_xmean_square_117Sn_119Sn)
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['R_X_corr_116Sn_120Sn'] = baxterStandardall['116Sn/120Sn'] / p_x_rm_mean_116Sn_120Sn_Standard * (R_X_RM_116Sn_120Sn/((pow((baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_116Sn_120Sn))))
        baxterStandardall['R_X_corr_117Sn_120Sn'] = baxterStandardall['117Sn/120Sn'] / p_x_rm_mean_117Sn_120Sn_Standard * (R_X_RM_117Sn_120Sn/((pow((baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_117Sn_120Sn))))
        baxterStandardall['R_X_corr_118Sn_120Sn'] = baxterStandardall['118Sn/120Sn'] / p_x_rm_mean_118Sn_120Sn_Standard * (R_X_RM_118Sn_120Sn/((pow((baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_118Sn_120Sn))))
        baxterStandardall['R_X_corr_119Sn_120Sn'] = baxterStandardall['119Sn/120Sn'] / p_x_rm_mean_119Sn_120Sn_Standard * (R_X_RM_119Sn_120Sn/((pow((baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_119Sn_120Sn))))
        baxterStandardall['R_X_corr_122Sn_116Sn'] = baxterStandardall['122Sn/116Sn'] / p_x_rm_mean_122Sn_116Sn_Standard * (R_X_RM_122Sn_116Sn/((pow((baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_122Sn_116Sn))))
        baxterStandardall['R_X_corr_124Sn_116Sn'] = baxterStandardall['124Sn/116Sn'] / p_x_rm_mean_124Sn_116Sn_Standard * (R_X_RM_124Sn_116Sn/((pow((baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_124Sn_116Sn))))
        baxterStandardall['R_X_corr_122Sn_120Sn'] = baxterStandardall['122Sn/120Sn'] / p_x_rm_mean_122Sn_120Sn_Standard * (R_X_RM_122Sn_120Sn/((pow((baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_122Sn_120Sn))))
        baxterStandardall['R_X_corr_124Sn_120Sn'] = baxterStandardall['124Sn/120Sn'] / p_x_rm_mean_124Sn_120Sn_Standard * (R_X_RM_124Sn_120Sn/((pow((baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_124Sn_120Sn))))
        baxterStandardall['R_X_corr_117Sn_119Sn'] = baxterStandardall['117Sn/119Sn'] / p_x_rm_mean_117Sn_119Sn_Standard * (R_X_RM_117Sn_119Sn/((pow((baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_117Sn_119Sn))))
    
        baxterStandardall['u^2_(rx_RM)_116Sn_120Sn'] = (baxterStandardall['116Sn/120Sn_1SE'] / baxterStandardall['116Sn/120Sn'])**2 * baxterStandardall['R_X_corr_116Sn_120Sn']**2
        baxterStandardall['u^2_(rx_RM)_117Sn_120Sn'] = (baxterStandardall['117Sn/120Sn_1SE'] / baxterStandardall['117Sn/120Sn'])**2 * baxterStandardall['R_X_corr_117Sn_120Sn']**2
        baxterStandardall['u^2_(rx_RM)_118Sn_120Sn'] = (baxterStandardall['118Sn/120Sn_1SE'] / baxterStandardall['118Sn/120Sn'])**2 * baxterStandardall['R_X_corr_118Sn_120Sn']**2
        baxterStandardall['u^2_(rx_RM)_119Sn_120Sn'] = (baxterStandardall['119Sn/120Sn_1SE'] / baxterStandardall['119Sn/120Sn'])**2 * baxterStandardall['R_X_corr_119Sn_120Sn']**2
        baxterStandardall['u^2_(rx_RM)_122Sn_116Sn'] = (baxterStandardall['122Sn/116Sn_1SE'] / baxterStandardall['122Sn/116Sn'])**2 * baxterStandardall['R_X_corr_122Sn_116Sn']**2
        baxterStandardall['u^2_(rx_RM)_124Sn_116Sn'] = (baxterStandardall['124Sn/116Sn_1SE'] / baxterStandardall['124Sn/116Sn'])**2 * baxterStandardall['R_X_corr_124Sn_116Sn']**2
        baxterStandardall['u^2_(rx_RM)_122Sn_120Sn'] = (baxterStandardall['122Sn/120Sn_1SE'] / baxterStandardall['122Sn/120Sn'])**2 * baxterStandardall['R_X_corr_122Sn_120Sn']**2
        baxterStandardall['u^2_(rx_RM)_124Sn_120Sn'] = (baxterStandardall['124Sn/120Sn_1SE'] / baxterStandardall['124Sn/120Sn'])**2 * baxterStandardall['R_X_corr_124Sn_120Sn']**2
        baxterStandardall['u^2_(rx_RM)_117Sn_119Sn'] = (baxterStandardall['117Sn/119Sn_1SE'] / baxterStandardall['117Sn/120Sn'])**2 * baxterStandardall['R_X_corr_117Sn_120Sn']**2
    
    
    R_X_RM_corr_mean_116Sn_120Sn = np.mean(baxterStandardall['R_X_corr_116Sn_120Sn'])
    R_X_RM_corr_mean_117Sn_120Sn = np.mean(baxterStandardall['R_X_corr_117Sn_120Sn'])
    R_X_RM_corr_mean_118Sn_120Sn = np.mean(baxterStandardall['R_X_corr_118Sn_120Sn'])
    R_X_RM_corr_mean_119Sn_120Sn = np.mean(baxterStandardall['R_X_corr_119Sn_120Sn'])
    R_X_RM_corr_mean_122Sn_116Sn = np.mean(baxterStandardall['R_X_corr_122Sn_116Sn'])
    R_X_RM_corr_mean_124Sn_116Sn = np.mean(baxterStandardall['R_X_corr_124Sn_116Sn'])
    R_X_RM_corr_mean_122Sn_120Sn = np.mean(baxterStandardall['R_X_corr_122Sn_120Sn'])
    R_X_RM_corr_mean_124Sn_120Sn = np.mean(baxterStandardall['R_X_corr_124Sn_120Sn'])
    R_X_RM_corr_mean_117Sn_119Sn = np.mean(baxterStandardall['R_X_corr_117Sn_119Sn'])
    
    massbias = pd.DataFrame()
    massbias['123Sb/121Sb'] = baxterStandardall['123Sb/121Sb'].copy()
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['u^2_(b_hat)_116Sn_120Sn'] = (sm_b_hat_116Sn_120Sn * np.log(baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_116Sn_120Sn)**2
        baxterStandardall['u^2_(b_hat)_117Sn_120Sn'] = (sm_b_hat_117Sn_120Sn * np.log(baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_117Sn_120Sn)**2
        baxterStandardall['u^2_(b_hat)_118Sn_120Sn'] = (sm_b_hat_118Sn_120Sn * np.log(baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_118Sn_120Sn)**2
        baxterStandardall['u^2_(b_hat)_119Sn_120Sn'] = (sm_b_hat_119Sn_120Sn * np.log(baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_119Sn_120Sn)**2
        baxterStandardall['u^2_(b_hat)_122Sn_116Sn'] = (sm_b_hat_122Sn_116Sn * np.log(baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_122Sn_116Sn)**2
        baxterStandardall['u^2_(b_hat)_124Sn_116Sn'] = (sm_b_hat_124Sn_116Sn * np.log(baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_124Sn_116Sn)**2
        baxterStandardall['u^2_(b_hat)_122Sn_120Sn'] = (sm_b_hat_122Sn_120Sn * np.log(baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_122Sn_120Sn)**2
        baxterStandardall['u^2_(b_hat)_124Sn_120Sn'] = (sm_b_hat_124Sn_120Sn * np.log(baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_124Sn_120Sn)**2
        baxterStandardall['u^2_(b_hat)_117Sn_119Sn'] = (sm_b_hat_117Sn_119Sn * np.log(baxterStandardall['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_117Sn_119Sn)**2
    
    
        baxterStandardall['u^2_(r_is)_116Sn_120Sn'] = (baxterStandardall['123Sb/121Sb_1SE'] / baxterStandardall['123Sb/121Sb'] * b_hat_116Sn_120Sn * baxterStandardall['R_X_corr_116Sn_120Sn'])**2
        baxterStandardall['u^2_(r_is)_117Sn_120Sn'] = (baxterStandardall['123Sb/121Sb_1SE'] / baxterStandardall['123Sb/121Sb'] * b_hat_117Sn_120Sn * baxterStandardall['R_X_corr_117Sn_120Sn'])**2
        baxterStandardall['u^2_(r_is)_118Sn_120Sn'] = (baxterStandardall['123Sb/121Sb_1SE'] / baxterStandardall['123Sb/121Sb'] * b_hat_118Sn_120Sn * baxterStandardall['R_X_corr_118Sn_120Sn'])**2
        baxterStandardall['u^2_(r_is)_119Sn_120Sn'] = (baxterStandardall['123Sb/121Sb_1SE'] / baxterStandardall['123Sb/121Sb'] * b_hat_119Sn_120Sn * baxterStandardall['R_X_corr_119Sn_120Sn'])**2
        baxterStandardall['u^2_(r_is)_122Sn_116Sn'] = (baxterStandardall['123Sb/121Sb_1SE'] / baxterStandardall['123Sb/121Sb'] * b_hat_122Sn_116Sn * baxterStandardall['R_X_corr_122Sn_116Sn'])**2
        baxterStandardall['u^2_(r_is)_124Sn_116Sn'] = (baxterStandardall['123Sb/121Sb_1SE'] / baxterStandardall['123Sb/121Sb'] * b_hat_124Sn_116Sn * baxterStandardall['R_X_corr_124Sn_116Sn'])**2
        baxterStandardall['u^2_(r_is)_122Sn_120Sn'] = (baxterStandardall['123Sb/121Sb_1SE'] / baxterStandardall['123Sb/121Sb'] * b_hat_122Sn_120Sn * baxterStandardall['R_X_corr_122Sn_120Sn'])**2
        baxterStandardall['u^2_(r_is)_124Sn_120Sn'] = (baxterStandardall['123Sb/121Sb_1SE'] / baxterStandardall['123Sb/121Sb'] * b_hat_124Sn_120Sn * baxterStandardall['R_X_corr_124Sn_120Sn'])**2
        baxterStandardall['u^2_(r_is)_117Sn_119Sn'] = (baxterStandardall['123Sb/121Sb_1SE'] / baxterStandardall['123Sb/121Sb'] * b_hat_117Sn_119Sn * baxterStandardall['R_X_corr_117Sn_119Sn'])**2
    
        baxterStandardall['u^2_(Rx_RM)_116Sn_120Sn'] = (U_X_RM_116Sn_120Sn / R_X_RM_116Sn_120Sn)**2 / 3 * (baxterStandardall['R_X_corr_116Sn_120Sn'] )**2
        baxterStandardall['u^2_(Rx_RM)_117Sn_120Sn'] = (U_X_RM_117Sn_120Sn / R_X_RM_117Sn_120Sn)**2 / 3 * (baxterStandardall['R_X_corr_117Sn_120Sn'] )**2
        baxterStandardall['u^2_(Rx_RM)_118Sn_120Sn'] = (U_X_RM_118Sn_120Sn / R_X_RM_118Sn_120Sn)**2 / 3 * (baxterStandardall['R_X_corr_118Sn_120Sn'] )**2
        baxterStandardall['u^2_(Rx_RM)_119Sn_120Sn'] = (U_X_RM_119Sn_120Sn / R_X_RM_119Sn_120Sn)**2 / 3 * (baxterStandardall['R_X_corr_119Sn_120Sn'] )**2
        baxterStandardall['u^2_(Rx_RM)_122Sn_116Sn'] = (U_X_RM_122Sn_116Sn / R_X_RM_122Sn_116Sn)**2 / 3 * (baxterStandardall['R_X_corr_122Sn_116Sn'] )**2
        baxterStandardall['u^2_(Rx_RM)_124Sn_116Sn'] = (U_X_RM_124Sn_116Sn / R_X_RM_124Sn_116Sn)**2 / 3 * (baxterStandardall['R_X_corr_124Sn_116Sn'] )**2
        baxterStandardall['u^2_(Rx_RM)_122Sn_120Sn'] = (U_X_RM_122Sn_120Sn / R_X_RM_122Sn_120Sn)**2 / 3 * (baxterStandardall['R_X_corr_122Sn_120Sn'] )**2
        baxterStandardall['u^2_(Rx_RM)_124Sn_120Sn'] = (U_X_RM_124Sn_120Sn / R_X_RM_124Sn_120Sn)**2 / 3 * (baxterStandardall['R_X_corr_124Sn_120Sn'] )**2
        baxterStandardall['u^2_(Rx_RM)_117Sn_119Sn'] = (U_X_RM_117Sn_119Sn / R_X_RM_117Sn_119Sn)**2 / 3 * (baxterStandardall['R_X_corr_117Sn_119Sn'] )**2
    
        baxterStandardall['mean_for_t_test_116S_120Sn'] = baxterStandardall[['u^2_(Rx_RM)_116Sn_120Sn', 'u^2_(b_hat)_116Sn_120Sn', 'u^2_(r_is)_116Sn_120Sn', 'u^2_(Rx_RM)_116Sn_120Sn']].mean(1)
        baxterStandardall['mean_for_t_test_117S_120Sn'] = baxterStandardall[['u^2_(Rx_RM)_117Sn_120Sn', 'u^2_(b_hat)_117Sn_120Sn', 'u^2_(r_is)_117Sn_120Sn', 'u^2_(Rx_RM)_117Sn_120Sn']].mean(1)
        baxterStandardall['mean_for_t_test_118S_120Sn'] = baxterStandardall[['u^2_(Rx_RM)_118Sn_120Sn', 'u^2_(b_hat)_118Sn_120Sn', 'u^2_(r_is)_118Sn_120Sn', 'u^2_(Rx_RM)_118Sn_120Sn']].mean(1)
        baxterStandardall['mean_for_t_test_119S_120Sn'] = baxterStandardall[['u^2_(Rx_RM)_119Sn_120Sn', 'u^2_(b_hat)_119Sn_120Sn', 'u^2_(r_is)_119Sn_120Sn', 'u^2_(Rx_RM)_119Sn_120Sn']].mean(1)
        baxterStandardall['mean_for_t_test_122S_116Sn'] = baxterStandardall[['u^2_(Rx_RM)_122Sn_116Sn', 'u^2_(b_hat)_122Sn_116Sn', 'u^2_(r_is)_122Sn_116Sn', 'u^2_(Rx_RM)_122Sn_116Sn']].mean(1)
        baxterStandardall['mean_for_t_test_124S_116Sn'] = baxterStandardall[['u^2_(Rx_RM)_124Sn_116Sn', 'u^2_(b_hat)_124Sn_116Sn', 'u^2_(r_is)_124Sn_116Sn', 'u^2_(Rx_RM)_124Sn_116Sn']].mean(1)
        baxterStandardall['mean_for_t_test_122S_120Sn'] = baxterStandardall[['u^2_(Rx_RM)_122Sn_120Sn', 'u^2_(b_hat)_122Sn_120Sn', 'u^2_(r_is)_122Sn_120Sn', 'u^2_(Rx_RM)_122Sn_120Sn']].mean(1)
        baxterStandardall['mean_for_t_test_124S_120Sn'] = baxterStandardall[['u^2_(Rx_RM)_124Sn_120Sn', 'u^2_(b_hat)_124Sn_120Sn', 'u^2_(r_is)_124Sn_120Sn', 'u^2_(Rx_RM)_124Sn_120Sn']].mean(1)
        baxterStandardall['mean_for_t_test_117S_119Sn'] = baxterStandardall[['u^2_(Rx_RM)_117Sn_119Sn', 'u^2_(b_hat)_117Sn_119Sn', 'u^2_(r_is)_117Sn_119Sn', 'u^2_(Rx_RM)_117Sn_119Sn']].mean(1)
    
        baxterStandardall['u_c(R_x_corr)_116Sn_120Sn'] = np.sqrt(baxterStandardall['mean_for_t_test_116S_120Sn'])
        baxterStandardall['u_c(R_x_corr)_117Sn_120Sn'] = np.sqrt(baxterStandardall['mean_for_t_test_117S_120Sn'])
        baxterStandardall['u_c(R_x_corr)_118Sn_120Sn'] = np.sqrt(baxterStandardall['mean_for_t_test_118S_120Sn'])
        baxterStandardall['u_c(R_x_corr)_119Sn_120Sn'] = np.sqrt(baxterStandardall['mean_for_t_test_119S_120Sn'])
        baxterStandardall['u_c(R_x_corr)_122Sn_116Sn'] = np.sqrt(baxterStandardall['mean_for_t_test_122S_116Sn'])
        baxterStandardall['u_c(R_x_corr)_124Sn_116Sn'] = np.sqrt(baxterStandardall['mean_for_t_test_124S_116Sn'])
        baxterStandardall['u_c(R_x_corr)_122Sn_120Sn'] = np.sqrt(baxterStandardall['mean_for_t_test_122S_120Sn'])
        baxterStandardall['u_c(R_x_corr)_124Sn_120Sn'] = np.sqrt(baxterStandardall['mean_for_t_test_124S_120Sn'])
        baxterStandardall['u_c(R_x_corr)_117Sn_119Sn'] = np.sqrt(baxterStandardall['mean_for_t_test_117S_119Sn'])
    
    
        baxterStandardall['v_eff_116Sn_120Sn'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_116Sn_120Sn'] -3)  #### richtig??
        baxterStandardall['v_eff_117Sn_120Sn'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_117Sn_120Sn'] -3)
        baxterStandardall['v_eff_118Sn_120Sn'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_118Sn_120Sn'] -3)
        baxterStandardall['v_eff_119Sn_120Sn'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_119Sn_120Sn'] -3)
        baxterStandardall['v_eff_122Sn_116Sn'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_122Sn_116Sn'] -3)
        baxterStandardall['v_eff_124Sn_116Sn'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_124Sn_116Sn'] -3)
        baxterStandardall['v_eff_122Sn_120Sn'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_122Sn_120Sn'] -3)
        baxterStandardall['v_eff_124Sn_120Sn'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_124Sn_120Sn'] -3)
        baxterStandardall['v_eff_117Sn_119Sn'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_117Sn_119Sn'] -3)
    
        baxterStandardall['t_95_Cl_116Sn_120Sn'] = baxterStandardall['u_c(R_x_corr)_116Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_116Sn_120Sn']) #
        baxterStandardall['t_95_Cl_117Sn_120Sn'] = baxterStandardall['u_c(R_x_corr)_117Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_117Sn_120Sn'])   
        baxterStandardall['t_95_Cl_118Sn_120Sn'] = baxterStandardall['u_c(R_x_corr)_118Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_118Sn_120Sn'])
        baxterStandardall['t_95_Cl_119Sn_120Sn'] = baxterStandardall['u_c(R_x_corr)_119Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_119Sn_120Sn'])
        baxterStandardall['t_95_Cl_122Sn_116Sn'] = baxterStandardall['u_c(R_x_corr)_122Sn_116Sn'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_122Sn_116Sn'])
        baxterStandardall['t_95_Cl_124Sn_116Sn'] = baxterStandardall['u_c(R_x_corr)_124Sn_116Sn'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_124Sn_116Sn'])
        baxterStandardall['t_95_Cl_122Sn_120Sn'] = baxterStandardall['u_c(R_x_corr)_122Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_122Sn_120Sn'])
        baxterStandardall['t_95_Cl_124Sn_120Sn'] = baxterStandardall['u_c(R_x_corr)_124Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_124Sn_120Sn'])
        baxterStandardall['t_95_Cl_117Sn_119Sn'] = baxterStandardall['u_c(R_x_corr)_117Sn_119Sn'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_117Sn_119Sn'])
    

    if sm_b_hat_116Sn_120Sn / b_hat_116Sn_120Sn < 0.15:
        pass
    
    else:
        print("More data points are required to adequately define the slope!")
        exit() 
    
    if sm_b_hat_117Sn_120Sn / b_hat_117Sn_120Sn < 0.15:
        pass
    
    else:
        print("More data points are required to adequately define the slope!")
        exit() 
    
    if sm_b_hat_118Sn_120Sn / b_hat_118Sn_120Sn < 0.15:
        pass
    
    else:
        print("More data points are required to adequately define the slope!")
        exit() 
    
    if sm_b_hat_119Sn_120Sn / b_hat_119Sn_120Sn < 0.15:
        pass
    
    else:
        print("More data points are required to adequately define the slope!")
        exit() 
    
    if sm_b_hat_122Sn_116Sn / b_hat_122Sn_116Sn < 0.15:
        pass
    
    else:
        print("More data points are required to adequately define the slope!")
        exit() 
    
    if sm_b_hat_124Sn_116Sn / b_hat_124Sn_116Sn < 0.15:
        pass
    
    else:
        print("More data points are required to adequately define the slope!")
        exit() 
    
    if sm_b_hat_122Sn_120Sn / b_hat_122Sn_120Sn < 0.15:
        pass
    
    else:
        print("More data points are required to adequately define the slope!")
        exit() 
    
    if sm_b_hat_124Sn_120Sn / b_hat_124Sn_120Sn < 0.15:
        pass
    
    else:
        print("More data points are required to adequately define the slope!")
        exit() 
    
    if sm_b_hat_117Sn_119Sn / b_hat_117Sn_119Sn < 0.15:
        pass
    
    else:
        print("More data points are required to adequately define the slope!")
        exit() 
    
    
    bx_bis_116Sn_120Sn =  b_hat_116Sn_120Sn * (np.log(masses['123Sb'] / masses['121Sb'])) / (np.log(masses['116Sn'] / masses['120Sn']))
    bx_bis_117Sn_120Sn =  b_hat_117Sn_120Sn * (np.log(masses['123Sb'] / masses['121Sb'])) / (np.log(masses['117Sn'] / masses['120Sn']))
    bx_bis_118Sn_120Sn =  b_hat_118Sn_120Sn * (np.log(masses['123Sb'] / masses['121Sb'])) / (np.log(masses['118Sn'] / masses['120Sn']))
    bx_bis_119Sn_120Sn =  b_hat_119Sn_120Sn * (np.log(masses['123Sb'] / masses['121Sb'])) / (np.log(masses['119Sn'] / masses['120Sn']))
    bx_bis_122Sn_116Sn =  b_hat_122Sn_116Sn * (np.log(masses['123Sb'] / masses['121Sb'])) / (np.log(masses['122Sn'] / masses['116Sn']))
    bx_bis_124Sn_116Sn =  b_hat_124Sn_116Sn * (np.log(masses['123Sb'] / masses['121Sb'])) / (np.log(masses['124Sn'] / masses['116Sn']))
    bx_bis_122Sn_120Sn =  b_hat_122Sn_120Sn * (np.log(masses['123Sb'] / masses['121Sb'])) / (np.log(masses['122Sn'] / masses['120Sn']))
    bx_bis_124Sn_120Sn =  b_hat_124Sn_120Sn * (np.log(masses['123Sb'] / masses['121Sb'])) / (np.log(masses['124Sn'] / masses['120Sn']))
    bx_bis_117Sn_119Sn =  b_hat_117Sn_119Sn * (np.log(masses['123Sb'] / masses['121Sb'])) / (np.log(masses['117Sn'] / masses['119Sn']))
    
    
    
    
    ###################################### Sample
    
    
    
    n = 0
    for SampleIndex in baxterSample.index:
        baxterSample['ln_r_is'] = np.log(baxterSample['123Sb/121Sb'])
        baxterSample['ln_r_x_rm_116Sn_120Sn'] = np.log(baxterSample['116Sn/120Sn'])
        baxterSample['ln_r_x_rm_117Sn_120Sn'] = np.log(baxterSample['117Sn/120Sn'])
        baxterSample['ln_r_x_rm_118Sn_120Sn'] = np.log(baxterSample['118Sn/120Sn'])
        baxterSample['ln_r_x_rm_119Sn_120Sn'] = np.log(baxterSample['119Sn/120Sn'])
        baxterSample['ln_r_x_rm_122Sn_116Sn'] = np.log(baxterSample['122Sn/116Sn'])
        baxterSample['ln_r_x_rm_124Sn_116Sn'] = np.log(baxterSample['124Sn/116Sn'])
        baxterSample['ln_r_x_rm_122Sn_120Sn'] = np.log(baxterSample['122Sn/120Sn'])
        baxterSample['ln_r_x_rm_124Sn_120Sn'] = np.log(baxterSample['124Sn/120Sn'])
        baxterSample['ln_r_x_rm_117Sn_119Sn'] = np.log(baxterSample['117Sn/119Sn'])
    
    
    nRM_Sample = len(baxterSample.index)
    
    m_is_block_Sample = nRM_Sample
    m_x_block_Sample = nRM_Sample
    
    
    for SampleIndex in baxterSample.index:
        baxterSample['R_X_corr_116Sn_120Sn'] = baxterSample['116Sn/120Sn'] / p_x_rm_mean_116Sn_120Sn_Standard * (R_X_RM_116Sn_120Sn/((pow((baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_116Sn_120Sn))))
        baxterSample['R_X_corr_117Sn_120Sn'] = baxterSample['117Sn/120Sn'] / p_x_rm_mean_117Sn_120Sn_Standard * (R_X_RM_117Sn_120Sn/((pow((baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_117Sn_120Sn))))
        baxterSample['R_X_corr_118Sn_120Sn'] = baxterSample['118Sn/120Sn'] / p_x_rm_mean_118Sn_120Sn_Standard * (R_X_RM_118Sn_120Sn/((pow((baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_118Sn_120Sn))))
        baxterSample['R_X_corr_119Sn_120Sn'] = baxterSample['119Sn/120Sn'] / p_x_rm_mean_119Sn_120Sn_Standard * (R_X_RM_119Sn_120Sn/((pow((baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_119Sn_120Sn))))
        baxterSample['R_X_corr_122Sn_116Sn'] = baxterSample['122Sn/116Sn'] / p_x_rm_mean_122Sn_116Sn_Standard * (R_X_RM_122Sn_116Sn/((pow((baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_122Sn_116Sn))))
        baxterSample['R_X_corr_124Sn_116Sn'] = baxterSample['124Sn/116Sn'] / p_x_rm_mean_124Sn_116Sn_Standard * (R_X_RM_124Sn_116Sn/((pow((baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_124Sn_116Sn))))
        baxterSample['R_X_corr_122Sn_120Sn'] = baxterSample['122Sn/120Sn'] / p_x_rm_mean_122Sn_120Sn_Standard * (R_X_RM_122Sn_120Sn/((pow((baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_122Sn_120Sn))))
        baxterSample['R_X_corr_124Sn_120Sn'] = baxterSample['124Sn/120Sn'] / p_x_rm_mean_124Sn_120Sn_Standard * (R_X_RM_124Sn_120Sn/((pow((baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_124Sn_120Sn))))
        baxterSample['R_X_corr_117Sn_119Sn'] = baxterSample['117Sn/119Sn'] / p_x_rm_mean_117Sn_119Sn_Standard * (R_X_RM_117Sn_119Sn/((pow((baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard), b_hat_117Sn_119Sn))))
    
        baxterSample['u^2_(rx_RM)_116Sn_120Sn'] = (baxterSample['116Sn/120Sn_1SE'] / baxterSample['116Sn/120Sn'])**2 * baxterSample['R_X_corr_116Sn_120Sn']**2
        baxterSample['u^2_(rx_RM)_117Sn_120Sn'] = (baxterSample['117Sn/120Sn_1SE'] / baxterSample['117Sn/120Sn'])**2 * baxterSample['R_X_corr_117Sn_120Sn']**2
        baxterSample['u^2_(rx_RM)_118Sn_120Sn'] = (baxterSample['118Sn/120Sn_1SE'] / baxterSample['118Sn/120Sn'])**2 * baxterSample['R_X_corr_118Sn_120Sn']**2
        baxterSample['u^2_(rx_RM)_119Sn_120Sn'] = (baxterSample['119Sn/120Sn_1SE'] / baxterSample['119Sn/120Sn'])**2 * baxterSample['R_X_corr_119Sn_120Sn']**2
        baxterSample['u^2_(rx_RM)_122Sn_116Sn'] = (baxterSample['122Sn/116Sn_1SE'] / baxterSample['122Sn/116Sn'])**2 * baxterSample['R_X_corr_122Sn_116Sn']**2
        baxterSample['u^2_(rx_RM)_124Sn_116Sn'] = (baxterSample['124Sn/116Sn_1SE'] / baxterSample['124Sn/116Sn'])**2 * baxterSample['R_X_corr_124Sn_116Sn']**2
        baxterSample['u^2_(rx_RM)_122Sn_120Sn'] = (baxterSample['122Sn/120Sn_1SE'] / baxterSample['122Sn/120Sn'])**2 * baxterSample['R_X_corr_122Sn_120Sn']**2
        baxterSample['u^2_(rx_RM)_124Sn_120Sn'] = (baxterSample['124Sn/120Sn_1SE'] / baxterSample['124Sn/120Sn'])**2 * baxterSample['R_X_corr_124Sn_120Sn']**2
        baxterSample['u^2_(rx_RM)_117Sn_119Sn'] = (baxterSample['117Sn/119Sn_1SE'] / baxterSample['117Sn/120Sn'])**2 * baxterSample['R_X_corr_117Sn_120Sn']**2
    
    
    for SampleIndex in baxterSample.index:
        baxterSample['u^2_(b_hat)_116Sn_120Sn'] = (sm_b_hat_116Sn_120Sn * np.log(baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_116Sn_120Sn)**2
        baxterSample['u^2_(b_hat)_117Sn_120Sn'] = (sm_b_hat_117Sn_120Sn * np.log(baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_117Sn_120Sn)**2
        baxterSample['u^2_(b_hat)_118Sn_120Sn'] = (sm_b_hat_118Sn_120Sn * np.log(baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_118Sn_120Sn)**2
        baxterSample['u^2_(b_hat)_119Sn_120Sn'] = (sm_b_hat_119Sn_120Sn * np.log(baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_119Sn_120Sn)**2
        baxterSample['u^2_(b_hat)_122Sn_116Sn'] = (sm_b_hat_122Sn_116Sn * np.log(baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_122Sn_116Sn)**2
        baxterSample['u^2_(b_hat)_124Sn_116Sn'] = (sm_b_hat_124Sn_116Sn * np.log(baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_124Sn_116Sn)**2
        baxterSample['u^2_(b_hat)_122Sn_120Sn'] = (sm_b_hat_122Sn_120Sn * np.log(baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_122Sn_120Sn)**2
        baxterSample['u^2_(b_hat)_124Sn_120Sn'] = (sm_b_hat_124Sn_120Sn * np.log(baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_124Sn_120Sn)**2
        baxterSample['u^2_(b_hat)_117Sn_119Sn'] = (sm_b_hat_117Sn_119Sn * np.log(baxterSample['123Sb/121Sb'] / p_is_mean_123Sb_121Sb_Standard) * R_X_RM_corr_mean_117Sn_119Sn)**2
    
    
        baxterSample['u^2_(r_is)_116Sn_120Sn'] = (baxterSample['123Sb/121Sb_1SE'] / baxterSample['123Sb/121Sb'] * b_hat_116Sn_120Sn * baxterSample['R_X_corr_116Sn_120Sn'])**2
        baxterSample['u^2_(r_is)_117Sn_120Sn'] = (baxterSample['123Sb/121Sb_1SE'] / baxterSample['123Sb/121Sb'] * b_hat_117Sn_120Sn * baxterSample['R_X_corr_117Sn_120Sn'])**2
        baxterSample['u^2_(r_is)_118Sn_120Sn'] = (baxterSample['123Sb/121Sb_1SE'] / baxterSample['123Sb/121Sb'] * b_hat_118Sn_120Sn * baxterSample['R_X_corr_118Sn_120Sn'])**2
        baxterSample['u^2_(r_is)_119Sn_120Sn'] = (baxterSample['123Sb/121Sb_1SE'] / baxterSample['123Sb/121Sb'] * b_hat_119Sn_120Sn * baxterSample['R_X_corr_119Sn_120Sn'])**2
        baxterSample['u^2_(r_is)_122Sn_116Sn'] = (baxterSample['123Sb/121Sb_1SE'] / baxterSample['123Sb/121Sb'] * b_hat_122Sn_116Sn * baxterSample['R_X_corr_122Sn_116Sn'])**2
        baxterSample['u^2_(r_is)_124Sn_116Sn'] = (baxterSample['123Sb/121Sb_1SE'] / baxterSample['123Sb/121Sb'] * b_hat_124Sn_116Sn * baxterSample['R_X_corr_124Sn_116Sn'])**2
        baxterSample['u^2_(r_is)_122Sn_120Sn'] = (baxterSample['123Sb/121Sb_1SE'] / baxterSample['123Sb/121Sb'] * b_hat_122Sn_120Sn * baxterSample['R_X_corr_122Sn_120Sn'])**2
        baxterSample['u^2_(r_is)_124Sn_120Sn'] = (baxterSample['123Sb/121Sb_1SE'] / baxterSample['123Sb/121Sb'] * b_hat_124Sn_120Sn * baxterSample['R_X_corr_124Sn_120Sn'])**2
        baxterSample['u^2_(r_is)_117Sn_119Sn'] = (baxterSample['123Sb/121Sb_1SE'] / baxterSample['123Sb/121Sb'] * b_hat_117Sn_119Sn * baxterSample['R_X_corr_117Sn_119Sn'])**2
    
        baxterSample['u^2_(Rx_RM)_116Sn_120Sn'] = (U_X_RM_116Sn_120Sn / R_X_RM_116Sn_120Sn)**2 / 3 * (baxterSample['R_X_corr_116Sn_120Sn'] )**2
        baxterSample['u^2_(Rx_RM)_117Sn_120Sn'] = (U_X_RM_117Sn_120Sn / R_X_RM_117Sn_120Sn)**2 / 3 * (baxterSample['R_X_corr_117Sn_120Sn'] )**2
        baxterSample['u^2_(Rx_RM)_118Sn_120Sn'] = (U_X_RM_118Sn_120Sn / R_X_RM_118Sn_120Sn)**2 / 3 * (baxterSample['R_X_corr_118Sn_120Sn'] )**2
        baxterSample['u^2_(Rx_RM)_119Sn_120Sn'] = (U_X_RM_119Sn_120Sn / R_X_RM_119Sn_120Sn)**2 / 3 * (baxterSample['R_X_corr_119Sn_120Sn'] )**2
        baxterSample['u^2_(Rx_RM)_122Sn_116Sn'] = (U_X_RM_122Sn_116Sn / R_X_RM_122Sn_116Sn)**2 / 3 * (baxterSample['R_X_corr_122Sn_116Sn'] )**2
        baxterSample['u^2_(Rx_RM)_124Sn_116Sn'] = (U_X_RM_124Sn_116Sn / R_X_RM_124Sn_116Sn)**2 / 3 * (baxterSample['R_X_corr_124Sn_116Sn'] )**2
        baxterSample['u^2_(Rx_RM)_122Sn_120Sn'] = (U_X_RM_122Sn_120Sn / R_X_RM_122Sn_120Sn)**2 / 3 * (baxterSample['R_X_corr_122Sn_120Sn'] )**2
        baxterSample['u^2_(Rx_RM)_124Sn_120Sn'] = (U_X_RM_124Sn_120Sn / R_X_RM_124Sn_120Sn)**2 / 3 * (baxterSample['R_X_corr_124Sn_120Sn'] )**2
        baxterSample['u^2_(Rx_RM)_117Sn_119Sn'] = (U_X_RM_117Sn_119Sn / R_X_RM_117Sn_119Sn)**2 / 3 * (baxterSample['R_X_corr_117Sn_119Sn'] )**2
    
        baxterSample['mean_for_t_test_116S_120Sn'] = baxterSample[['u^2_(Rx_RM)_116Sn_120Sn', 'u^2_(b_hat)_116Sn_120Sn', 'u^2_(r_is)_116Sn_120Sn', 'u^2_(Rx_RM)_116Sn_120Sn']].mean(1)
        baxterSample['mean_for_t_test_117S_120Sn'] = baxterSample[['u^2_(Rx_RM)_117Sn_120Sn', 'u^2_(b_hat)_117Sn_120Sn', 'u^2_(r_is)_117Sn_120Sn', 'u^2_(Rx_RM)_117Sn_120Sn']].mean(1)
        baxterSample['mean_for_t_test_118S_120Sn'] = baxterSample[['u^2_(Rx_RM)_118Sn_120Sn', 'u^2_(b_hat)_118Sn_120Sn', 'u^2_(r_is)_118Sn_120Sn', 'u^2_(Rx_RM)_118Sn_120Sn']].mean(1)
        baxterSample['mean_for_t_test_119S_120Sn'] = baxterSample[['u^2_(Rx_RM)_119Sn_120Sn', 'u^2_(b_hat)_119Sn_120Sn', 'u^2_(r_is)_119Sn_120Sn', 'u^2_(Rx_RM)_119Sn_120Sn']].mean(1)
        baxterSample['mean_for_t_test_122S_116Sn'] = baxterSample[['u^2_(Rx_RM)_122Sn_116Sn', 'u^2_(b_hat)_122Sn_116Sn', 'u^2_(r_is)_122Sn_116Sn', 'u^2_(Rx_RM)_122Sn_116Sn']].mean(1)
        baxterSample['mean_for_t_test_124S_116Sn'] = baxterSample[['u^2_(Rx_RM)_124Sn_116Sn', 'u^2_(b_hat)_124Sn_116Sn', 'u^2_(r_is)_124Sn_116Sn', 'u^2_(Rx_RM)_124Sn_116Sn']].mean(1)
        baxterSample['mean_for_t_test_122S_120Sn'] = baxterSample[['u^2_(Rx_RM)_122Sn_120Sn', 'u^2_(b_hat)_122Sn_120Sn', 'u^2_(r_is)_122Sn_120Sn', 'u^2_(Rx_RM)_122Sn_120Sn']].mean(1)
        baxterSample['mean_for_t_test_124S_120Sn'] = baxterSample[['u^2_(Rx_RM)_124Sn_120Sn', 'u^2_(b_hat)_124Sn_120Sn', 'u^2_(r_is)_124Sn_120Sn', 'u^2_(Rx_RM)_124Sn_120Sn']].mean(1)
        baxterSample['mean_for_t_test_117S_119Sn'] = baxterSample[['u^2_(Rx_RM)_117Sn_119Sn', 'u^2_(b_hat)_117Sn_119Sn', 'u^2_(r_is)_117Sn_119Sn', 'u^2_(Rx_RM)_117Sn_119Sn']].mean(1)
    
        baxterSample['u_c(R_x_corr)_116Sn_120Sn'] = np.sqrt(baxterSample['mean_for_t_test_116S_120Sn'])
        baxterSample['u_c(R_x_corr)_117Sn_120Sn'] = np.sqrt(baxterSample['mean_for_t_test_117S_120Sn'])
        baxterSample['u_c(R_x_corr)_118Sn_120Sn'] = np.sqrt(baxterSample['mean_for_t_test_118S_120Sn'])
        baxterSample['u_c(R_x_corr)_119Sn_120Sn'] = np.sqrt(baxterSample['mean_for_t_test_119S_120Sn'])
        baxterSample['u_c(R_x_corr)_122Sn_116Sn'] = np.sqrt(baxterSample['mean_for_t_test_122S_116Sn'])
        baxterSample['u_c(R_x_corr)_124Sn_116Sn'] = np.sqrt(baxterSample['mean_for_t_test_124S_116Sn'])
        baxterSample['u_c(R_x_corr)_122Sn_120Sn'] = np.sqrt(baxterSample['mean_for_t_test_122S_120Sn'])
        baxterSample['u_c(R_x_corr)_124Sn_120Sn'] = np.sqrt(baxterSample['mean_for_t_test_124S_120Sn'])
        baxterSample['u_c(R_x_corr)_117Sn_119Sn'] = np.sqrt(baxterSample['mean_for_t_test_117S_119Sn'])
    
    
        baxterSample['v_eff_116Sn_120Sn'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_116Sn_120Sn'] -3)  #### richtig??
        baxterSample['v_eff_117Sn_120Sn'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_117Sn_120Sn'] -3)
        baxterSample['v_eff_118Sn_120Sn'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_118Sn_120Sn'] -3)
        baxterSample['v_eff_119Sn_120Sn'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_119Sn_120Sn'] -3)
        baxterSample['v_eff_122Sn_116Sn'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_122Sn_116Sn'] -3)
        baxterSample['v_eff_124Sn_116Sn'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_124Sn_116Sn'] -3)
        baxterSample['v_eff_122Sn_120Sn'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_122Sn_120Sn'] -3)
        baxterSample['v_eff_124Sn_120Sn'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_124Sn_120Sn'] -3)
        baxterSample['v_eff_117Sn_119Sn'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_117Sn_119Sn'] -3)
    
        baxterSample['t_95_Cl_116Sn_120Sn'] = baxterSample['u_c(R_x_corr)_116Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_116Sn_120Sn']) #
        baxterSample['t_95_Cl_117Sn_120Sn'] = baxterSample['u_c(R_x_corr)_117Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_117Sn_120Sn'])   
        baxterSample['t_95_Cl_118Sn_120Sn'] = baxterSample['u_c(R_x_corr)_118Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_118Sn_120Sn'])
        baxterSample['t_95_Cl_119Sn_120Sn'] = baxterSample['u_c(R_x_corr)_119Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_119Sn_120Sn'])
        baxterSample['t_95_Cl_122Sn_116Sn'] = baxterSample['u_c(R_x_corr)_122Sn_116Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_122Sn_116Sn'])
        baxterSample['t_95_Cl_124Sn_116Sn'] = baxterSample['u_c(R_x_corr)_124Sn_116Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_124Sn_116Sn'])
        baxterSample['t_95_Cl_122Sn_120Sn'] = baxterSample['u_c(R_x_corr)_122Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_122Sn_120Sn'])
        baxterSample['t_95_Cl_124Sn_120Sn'] = baxterSample['u_c(R_x_corr)_124Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_124Sn_120Sn'])
        baxterSample['t_95_Cl_117Sn_119Sn'] = baxterSample['u_c(R_x_corr)_117Sn_119Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_117Sn_119Sn'])
    
    
    
        baxterSample['d_116Sn_120Sn'] = (( baxterSample['116Sn/120Sn'] / p_x_rm_mean_116Sn_120Sn_Standard) * (p_is_mean_123Sb_121Sb_Standard / baxterSample['123Sb/121Sb'])**b_hat_116Sn_120Sn -1) * 1000
        baxterSample['d_117Sn_120Sn'] = (( baxterSample['117Sn/120Sn'] / p_x_rm_mean_117Sn_120Sn_Standard) * (p_is_mean_123Sb_121Sb_Standard / baxterSample['123Sb/121Sb'])**b_hat_117Sn_120Sn -1) * 1000
        baxterSample['d_118Sn_120Sn'] = (( baxterSample['118Sn/120Sn'] / p_x_rm_mean_118Sn_120Sn_Standard) * (p_is_mean_123Sb_121Sb_Standard / baxterSample['123Sb/121Sb'])**b_hat_118Sn_120Sn -1) * 1000
        baxterSample['d_119Sn_120Sn'] = (( baxterSample['119Sn/120Sn'] / p_x_rm_mean_119Sn_120Sn_Standard) * (p_is_mean_123Sb_121Sb_Standard / baxterSample['123Sb/121Sb'])**b_hat_119Sn_120Sn -1) * 1000
        baxterSample['d_122Sn_116Sn'] = (( baxterSample['122Sn/116Sn'] / p_x_rm_mean_122Sn_116Sn_Standard) * (p_is_mean_123Sb_121Sb_Standard / baxterSample['123Sb/121Sb'])**b_hat_122Sn_116Sn -1) * 1000
        baxterSample['d_124Sn_116Sn'] = (( baxterSample['124Sn/116Sn'] / p_x_rm_mean_124Sn_116Sn_Standard) * (p_is_mean_123Sb_121Sb_Standard / baxterSample['123Sb/121Sb'])**b_hat_124Sn_116Sn -1) * 1000
        baxterSample['d_122Sn_120Sn'] = (( baxterSample['122Sn/120Sn'] / p_x_rm_mean_122Sn_120Sn_Standard) * (p_is_mean_123Sb_121Sb_Standard / baxterSample['123Sb/121Sb'])**b_hat_122Sn_120Sn -1) * 1000
        baxterSample['d_124Sn_120Sn'] = (( baxterSample['124Sn/120Sn'] / p_x_rm_mean_124Sn_120Sn_Standard) * (p_is_mean_123Sb_121Sb_Standard / baxterSample['123Sb/121Sb'])**b_hat_124Sn_120Sn -1) * 1000
        baxterSample['d_117Sn_119Sn'] = (( baxterSample['117Sn/119Sn'] / p_x_rm_mean_117Sn_119Sn_Standard) * (p_is_mean_123Sb_121Sb_Standard / baxterSample['123Sb/121Sb'])**b_hat_117Sn_119Sn -1) * 1000
    
        baxterSample['u_c_d_116Sn_120Sn'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_116Sn_120Sn'] + baxterSample['u^2_(b_hat)_116Sn_120Sn'] + baxterSample['u^2_(rx_RM)_116Sn_120Sn']) / baxterSample['R_X_corr_116Sn_120Sn']**2
        baxterSample['u_c_d_117Sn_120Sn'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_117Sn_120Sn'] + baxterSample['u^2_(b_hat)_117Sn_120Sn'] + baxterSample['u^2_(rx_RM)_117Sn_120Sn']) / baxterSample['R_X_corr_117Sn_120Sn']**2
        baxterSample['u_c_d_118Sn_120Sn'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_118Sn_120Sn'] + baxterSample['u^2_(b_hat)_118Sn_120Sn'] + baxterSample['u^2_(rx_RM)_118Sn_120Sn']) / baxterSample['R_X_corr_118Sn_120Sn']**2
        baxterSample['u_c_d_119Sn_120Sn'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_119Sn_120Sn'] + baxterSample['u^2_(b_hat)_119Sn_120Sn'] + baxterSample['u^2_(rx_RM)_119Sn_120Sn']) / baxterSample['R_X_corr_119Sn_120Sn']**2
        baxterSample['u_c_d_122Sn_116Sn'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_122Sn_116Sn'] + baxterSample['u^2_(b_hat)_122Sn_116Sn'] + baxterSample['u^2_(rx_RM)_122Sn_116Sn']) / baxterSample['R_X_corr_122Sn_116Sn']**2
        baxterSample['u_c_d_124Sn_116Sn'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_124Sn_116Sn'] + baxterSample['u^2_(b_hat)_124Sn_116Sn'] + baxterSample['u^2_(rx_RM)_124Sn_116Sn']) / baxterSample['R_X_corr_124Sn_116Sn']**2
        baxterSample['u_c_d_122Sn_120Sn'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_122Sn_120Sn'] + baxterSample['u^2_(b_hat)_122Sn_120Sn'] + baxterSample['u^2_(rx_RM)_122Sn_120Sn']) / baxterSample['R_X_corr_122Sn_120Sn']**2
        baxterSample['u_c_d_124Sn_120Sn'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_124Sn_120Sn'] + baxterSample['u^2_(b_hat)_124Sn_120Sn'] + baxterSample['u^2_(rx_RM)_124Sn_120Sn']) / baxterSample['R_X_corr_124Sn_120Sn']**2
        baxterSample['u_c_d_117Sn_119Sn'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_117Sn_119Sn'] + baxterSample['u^2_(b_hat)_117Sn_119Sn'] + baxterSample['u^2_(rx_RM)_117Sn_119Sn']) / baxterSample['R_X_corr_117Sn_119Sn']**2
    
    
        baxterSample['t_95_Cl_d_116Sn_120Sn'] = baxterSample['u_c_d_116Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_116Sn_120Sn']) #
        baxterSample['t_95_Cl_d_117Sn_120Sn'] = baxterSample['u_c_d_117Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_117Sn_120Sn'])   
        baxterSample['t_95_Cl_d_118Sn_120Sn'] = baxterSample['u_c_d_118Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_118Sn_120Sn'])
        baxterSample['t_95_Cl_d_119Sn_120Sn'] = baxterSample['u_c_d_119Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_119Sn_120Sn'])
        baxterSample['t_95_Cl_d_122Sn_116Sn'] = baxterSample['u_c_d_122Sn_116Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_122Sn_116Sn'])
        baxterSample['t_95_Cl_d_124Sn_116Sn'] = baxterSample['u_c_d_124Sn_116Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_124Sn_116Sn'])
        baxterSample['t_95_Cl_d_122Sn_120Sn'] = baxterSample['u_c_d_122Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_122Sn_120Sn'])
        baxterSample['t_95_Cl_d_124Sn_120Sn'] = baxterSample['u_c_d_124Sn_120Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_124Sn_120Sn'])
        baxterSample['t_95_Cl_d_117Sn_119Sn'] = baxterSample['u_c_d_117Sn_119Sn'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_117Sn_119Sn'])
    
    
    baxterSample_delta = baxterSample[['Inputfile', 'd_116Sn_120Sn', 'd_117Sn_120Sn', 'd_118Sn_120Sn', 'd_119Sn_120Sn', 'd_122Sn_116Sn', 'd_124Sn_116Sn', 'd_122Sn_120Sn', 'd_124Sn_120Sn', 'd_117Sn_119Sn', 'u_c_d_116Sn_120Sn', \
         'u_c_d_117Sn_120Sn', 'u_c_d_118Sn_120Sn', 'u_c_d_119Sn_120Sn', 'u_c_d_122Sn_116Sn', 'u_c_d_124Sn_116Sn', 'u_c_d_122Sn_120Sn', 'u_c_d_124Sn_120Sn', 'u_c_d_117Sn_119Sn']].copy()
    
    
    baxterSample_delta.rename(columns = {'Inputfile' : 'Filename'}, inplace = True)
    
    delta_cols = [col for col in baxterSample_delta if 'd_' in col]
    deltaresults = baxterSample_delta.groupby('Filename')[delta_cols].mean()
    deltaresults.drop(list(deltaresults.filter(regex='u_c')), axis = 1, inplace = True)
    deltaresults_round = np.round(deltaresults, 3)
    
    
    
    delta2sd = baxterSample_delta.groupby('Filename')[delta_cols].std(ddof = 0) * 2
    new_names_sd = [(i,'2SD+' + i) for i in delta2sd.iloc[:, 0:].columns.values]
    delta2sd.rename(columns = dict(new_names_sd), inplace = True)
    delta2sd_round = np.round(delta2sd, 3)
    results = pd.concat([deltaresults_round, delta2sd_round], axis = 1) 
    results = results[[item for items in zip(deltaresults_round.columns, delta2sd_round.columns) for item in items]]
    
    cols = [col for col in baxterSample if '1' in col]
    tin_antimon = baxterSample[['Inputfile', '120Sn', '121Sb']].copy().round(1)
    cols = [col for col in tin_antimon if '1' in col]
    tin_antimon_mean = tin_antimon.groupby('Inputfile')[cols].mean()
    tin_antimon_mean = tin_antimon_mean.round(1)
    
    tin_antimon_mean['120Sn/121Sb'] = (tin_antimon_mean['120Sn'] / tin_antimon_mean['121Sb']).round(2)
    result_all = pd.concat([results, tin_antimon_mean], axis = 1)
    
    
    result_values = result_all.copy()
    result_values.drop(list(result_values.filter(regex='2SD')), axis = 1, inplace = True)
    result_values.drop(list(result_values.filter(regex='117')), axis = 1, inplace = True)
    result_values.drop(list(result_values.filter(regex='119')), axis = 1, inplace = True)
    
    #dSn
    df_ratio_delta = pd.DataFrame([{'116Sn/120Sn': -4,'117/120Sn': -3, '118Sn/120Sn': -2, '119Sn/120Sn': -1, '122Sn/116Sn': 6, '124Sn/116Sn': 8, '122Sn/120Sn': 2, '124Sn/120Sn': 4, '117Sn/119Sn': -2}])
    df_4 = baxterSample_delta.copy()
    df_4.drop(list(df_4.filter(regex='u_c')), axis = 1, inplace = True)
    col4_result = list(df_4.columns)
    col4_result = [t.replace('d_','') for t in col4_result]
    df_4.columns =col4_result
    df_4 = df_4.set_index('Filename')
    df_4 = df_4.astype(float)
    
    def cal(row):
        reg = np.polyfit(df_ratio_delta.values[0], row.values, 1)
        predict = np.poly1d(reg) # Slope and intercept
        trend = np.polyval(reg, df_ratio_delta)
        std = row.std() # Standard deviation
        r2 = np.round(r2_score(row.values, predict(df_ratio_delta.T)), 5) #R-squared
        df5 = pd.DataFrame([predict])
        df5.columns = ['dSn', 'Intercept']
        Slope = df5['dSn'].to_string(dtype=False)
        Slope = Slope[3:]
        Intercept = df5['Intercept'].to_string(dtype=False)
        Intercept = Intercept[3:]
        return Slope, Intercept, r2
    
    
    df_4[['dSn', 'intercept', 'r2']] = df_4.apply(cal, axis=1, result_type='expand')
    df_4[['dSn', 'intercept']] = df_4[['dSn', 'intercept']].astype(float)
    
    df_6 = df_4[['dSn', 'intercept', 'r2']]
    df_6 = df_6.reset_index(drop = True)
    result_values_2 = pd.concat([result_values, df_6], axis = 1)
    
    dSn_mean = np.round(df_4.groupby('Filename')['dSn'].mean(), 3)
    dSn_2sd = np.round(df_4.groupby('Filename')['dSn'].std(ddof = 0) * 2, 3)
    dSn_2sd = dSn_2sd.rename('2SD')
    
    result_all = pd.concat([result_all, dSn_mean], axis = 1)
    result_all = pd.concat([result_all, dSn_2sd], axis = 1)
    
    
    col_result = list(result_all.columns)
    col_result.insert(0, 'Lab Nummer')
    col_result = [l.replace('_','') for l in col_result]
    col_result = [i.split('+', 1)[0] for i in col_result]
    
    result_all.to_csv('Baxter_Export.csv', sep='\t', mode="w", header=True, index_label='Index_name')
    
    infiles = infile + '/'
    filehead = [x for x in os.listdir(infiles) if x .endswith('exp') and 'Nis01' in x]
    filehead = infiles + str(filehead)[2:-2]
        
    df3 = pd.read_csv(filehead, skiprows = 1, nrows = 1, sep=' |\t', engine='python', index_col = False)
    df3.columns = range(df3.shape[1])
    df3.drop(df3.columns[0], axis = 1, inplace = True)
    df3.drop(df3.columns[1:3], axis = 1, inplace = True)
    
    reread = pd.read_csv('Baxter_Export.csv', sep='\t', index_col=False) # names = col_result,
    reread.columns = col_result
    #print(col_result)
    reread2 = reread.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 11, 12, 9, 10, 17, 18, 19, 20, 21, 22, 23]]
    
    reread2.columns = reread2.columns.str.replace('2SD', '2SD')
    
    
    reread2['Lab Nummer'] = reread2['Lab Nummer'].str.split('_').str.get(-1)  #make Filename nice
    reread2['Lab Nummer'] = reread2['Lab Nummer'].str.split('.').str.get(0)
    reread2.insert(1,'Messdatum', '')
    reread2.insert(0, 'Objekt', '')
    reread2.insert(0, 'Orig. Nummer', '')
    reread2.insert(0, 'Ma Nummer', '')
    reread2['Messdatum'] = df3.values[0][0]
    reread2['Referenz'] = 'NIST SRM 3161a'
    reread2['Verwenden'] = ''
    reread2['Bemerkung'] ='Baxter, Sprühkammer, Ni-Cone'
    reread2.to_csv(resultnameSn_Baxter, sep = '\t', header=True, index=False)
    reread2.to_excel(resultnameSnExcel_Baxter, index=False)
