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

from modules.config import outfile_results_Li, outfile_results_Sn, outfile_results_Ag, Baxter_Li
from modules.config import WorkspaceVariableInput
infile = WorkspaceVariableInput

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

pd.options.mode.chained_assignment = None
pd.set_option("display.precision", 15)
pd.options.display.float_format = '{:.15f}'.format
os.getcwd()
resultnameLi_Baxter = os.path.join(outfile_results_Li, 'Lidelta_Baxter.csv') 
resultnameLiExcel_Baxter = os.path.join(outfile_results_Li, 'Li_delta_Baxter.xlsx')


def baxterLi():
    global resultnameLi_Baxter
    global resultnameLiExcel_Baxter
    global infile
    baxter = pd.read_csv(Baxter_Li + 'Baxter.csv', sep = '\t', engine = 'python', index_col = False)
    baxter = baxter.drop(baxter.columns[[0]], axis = 1)


  
    baxter.drop(list(baxter.filter(regex = '_corrected')), axis = 1, inplace = True)
    
    baxter['standard/'] = baxter['Inputfile'].str.contains('_s0|s1')
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
    baxterStandard = baxterStandard.drop('index', axis = 1)
    baxterStandardall = baxterStandard.copy()
    baxterSample = baxter[baxter['sample/'] == True]
    
    baxterSample['Inputfile'] = baxterSample['Inputfile'].map(lambda x: x.rstrip('_123'))
    
    
    
    #baxter_reduced.head()
    """from Lee et al. 1995"""
    R_X_RM_Ni = 1
         
    """" End of insertion of Lee et al. 1995; For Li deactivated with setting by =1 """

    masses = pd.DataFrame()
    masses.loc[0, '6Li'] = 62.9295989
    masses.loc[0, '7Li'] = 64.9277929
    
    abundances = pd.DataFrame()
    abundances.loc[0, '6Li'] = 0.6915
    abundances.loc[0, '6Li_error'] = 0.0015
    abundances.loc[0, '7Li'] = 0.3085
    abundances.loc[0, '7Li_error'] = 0.0015
    
    R_is_rm = 1
    R_X_RM_7Li_6Li = abundances['7Li'] / abundances['6Li']
    U_X_RM_7Li_6Li = R_X_RM_7Li_6Li*np.sqrt((abundances['7Li_error']/abundances['7Li'])**2+(abundances['6Li_error']/abundances['6Li'])**2)/2
    
    n = 0
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['ln_r_is'] = np.log(baxterStandardall['7Li/6Li'])
        baxterStandardall['ln_r_x_rm'] = np.log(baxterStandardall['7Li/6Li'])
        
     

    nRM_Standard = len(baxterStandardall.index)
    
    m_is_block_Standard = nRM_Standard
    m_x_block_Standard = nRM_Standard
    
    ##X_mean // remains the same for all masses! Küzel bedeuten "Zelle xxx in Baxter-Spreadsheet 116-120"
    x_mean_sum_ln_62Ni_60Ni_Standard = sum(baxterStandardall['ln_r_is'])  ## = AE8
    ln_r_is_62Ni_60Ni_Standard = x_mean_sum_ln_62Ni_60Ni_Standard / nRM_Standard   ## AE4
    p_is_mean_62Ni_60Ni_Standard = np.exp(ln_r_is_62Ni_60Ni_Standard) ## = AE5
    
    
    ## Y_mean // changes for every mass ratio
    y_mean_sum__ln7Li_6Li_Standard = sum(baxterStandardall['ln_r_x_rm']) ## = AF8
    
    ln_r_x_rm_7Li_6Li_Standard = y_mean_sum__ln7Li_6Li_Standard / nRM_Standard   ## AF4

    
    ### X_mean
    p_x_rm_mean_7Li_6Li_Standard = np.exp(ln_r_x_rm_7Li_6Li_Standard) ## = AE6
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['Xi-Xmean_7Li_6Li'] = baxterStandardall['ln_r_is'] - ln_r_is_62Ni_60Ni_Standard  ## column AH from row 12 down
        baxterStandardall['Yi-Ymean_7Li_6Li'] = baxterStandardall['ln_r_x_rm'] - ln_r_x_rm_7Li_6Li_Standard ## column AI from row 12 down
        baxterStandardall['xiyi_7Li_6Li'] = baxterStandardall['Xi-Xmean_7Li_6Li'] * baxterStandardall['Yi-Ymean_7Li_6Li']  # column AL
        baxterStandardall['xi^2_7Li_6Li'] = baxterStandardall['Xi-Xmean_7Li_6Li'] * baxterStandardall['Xi-Xmean_7Li_6Li'] # column AM
        baxterStandardall['yi^2_7Li_6Li'] = baxterStandardall['Yi-Ymean_7Li_6Li'] * baxterStandardall['Yi-Ymean_7Li_6Li'] # Column AN
    
    sum_xiyi_7Li_6Li = sum(baxterStandardall['xiyi_7Li_6Li']) # AL8
    sum_xi_exp_2_7Li_6Li = sum(baxterStandardall['xi^2_7Li_6Li']) # AM8
    sum_yi_exp_2_7Li_6Li = sum(baxterStandardall['yi^2_7Li_6Li']) # AN8

    b_hat_7Li_6Li = sum_xiyi_7Li_6Li / sum_xi_exp_2_7Li_6Li # AM4
  
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['yi_hat_7Li_6Li'] = baxterStandardall['Xi-Xmean_7Li_6Li'] * b_hat_7Li_6Li # column AO
    
        baxterStandardall['yi-yi_hat^2_7Li_6Li'] = (baxterStandardall['Yi-Ymean_7Li_6Li'] - baxterStandardall['yi_hat_7Li_6Li'])**2
    
        baxterStandardall['R_X_corr_7Li_6Li'] = baxterStandardall['7Li/6Li'] / p_x_rm_mean_7Li_6Li_Standard * (R_X_RM_7Li_6Li/((pow((baxterStandardall['7Li/6Li'] / p_is_mean_62Ni_60Ni_Standard), b_hat_7Li_6Li))))
         
        baxterStandardall['u^2_(rx_RM)_7Li_6Li'] = (baxterStandardall['7Li/6Li_1SE'] / baxterStandardall['7Li/6Li'])**2 * baxterStandardall['R_X_corr_7Li_6Li']**2
    
    sum_yi_yi_hat_square_7Li_6Li = sum(baxterStandardall['yi-yi_hat^2_7Li_6Li']) # AP8
    
    
    S_y_x_7Li_6Li = np.sqrt(sum_yi_yi_hat_square_7Li_6Li / (nRM_Standard - 1))  ## AM2
    
    
    sum_xi_xmean_square_7Li_6Li = sum_xi_exp_2_7Li_6Li - sum(baxterStandardall['Xi-Xmean_7Li_6Li'])**2 / nRM_Standard # AM3
    
    
    sm_b_hat_7Li_6Li = np.sqrt((S_y_x_7Li_6Li**2) / sum_xi_xmean_square_7Li_6Li)  # AM5
    
    R_X_RM_corr_mean_7Li_6Li = np.mean(baxterStandardall['R_X_corr_7Li_6Li'])
    
    massbias = pd.DataFrame()
    massbias['7Li/6Li'] = baxterStandardall['7Li/6Li'].copy()
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['u^2_(b_hat)_7Li_6Li'] = (sm_b_hat_7Li_6Li * np.log(baxterStandardall['7Li/6Li'] / p_is_mean_62Ni_60Ni_Standard) * R_X_RM_corr_mean_7Li_6Li)**2
       
    
        baxterStandardall['u^2_(r_is)_7Li_6Li'] = (baxterStandardall['7Li/6Li_1SE'] / baxterStandardall['7Li/6Li'] * b_hat_7Li_6Li * baxterStandardall['R_X_corr_7Li_6Li'])**2
     
        baxterStandardall['u^2_(Rx_RM)_7Li_6Li'] = (U_X_RM_7Li_6Li / R_X_RM_7Li_6Li)**2 / 3 * (baxterStandardall['R_X_corr_7Li_6Li'] )**2
      
        baxterStandardall['mean_for_t_test_7Li_6Li'] = baxterStandardall[['u^2_(Rx_RM)_7Li_6Li', 'u^2_(b_hat)_7Li_6Li', 'u^2_(r_is)_7Li_6Li', 'u^2_(Rx_RM)_7Li_6Li']].mean(1)
    
        baxterStandardall['u_c(R_x_corr)_7Li_6Li'] = np.sqrt(baxterStandardall['mean_for_t_test_7Li_6Li'])
     
    
        baxterStandardall['v_eff_7Li_6Li'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_7Li_6Li'] -3)  #### richtig??
     
        baxterStandardall['t_95_Cl_7Li_6Li'] = baxterStandardall['u_c(R_x_corr)_7Li_6Li'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_7Li_6Li']) #
    
    
    if sm_b_hat_7Li_6Li / b_hat_7Li_6Li < 0.15:
        pass
    
    else:
        print("More data points are required to adequately define the slope!")
        exit() 
    
    
    
    
    bx_bis_7Li_6Li =  b_hat_7Li_6Li * (np.log(masses['7Li'] / masses['6Li'])) / (np.log(masses['7Li'] / masses['6Li']))
    
    
    ###################################### Sample
    
    
    
    n = 0
    for SampleIndex in baxterSample.index:
        baxterSample['ln_r_is'] = np.log(baxterSample['7Li/6Li'])
        baxterSample['ln_r_x_rm_7Li_6Li'] = np.log(baxterSample['7Li/6Li'])
        
        
       
    
    
    
    nRM_Sample = len(baxterSample.index)
    
    m_is_block_Sample = nRM_Sample
    m_x_block_Sample = nRM_Sample
    
    
    for SampleIndex in baxterSample.index:
        baxterSample['R_X_corr_7Li_6Li'] = baxterSample['7Li/6Li'] / p_x_rm_mean_7Li_6Li_Standard * (R_X_RM_7Li_6Li/((pow((baxterSample['7Li/6Li'] / p_is_mean_62Ni_60Ni_Standard), b_hat_7Li_6Li))))
      
        baxterSample['u^2_(rx_RM)_7Li_6Li'] = (baxterSample['7Li/6Li_1SE'] / baxterSample['7Li/6Li'])**2 * baxterSample['R_X_corr_7Li_6Li']**2
     
    
    for SampleIndex in baxterSample.index:
        baxterSample['u^2_(b_hat)_7Li_6Li'] = (sm_b_hat_7Li_6Li * np.log(baxterSample['7Li/6Li'] / p_is_mean_62Ni_60Ni_Standard) * R_X_RM_corr_mean_7Li_6Li)**2
    
        baxterSample['u^2_(r_is)_7Li_6Li'] = (baxterSample['7Li/6Li_1SE'] / baxterSample['7Li/6Li'] * b_hat_7Li_6Li * baxterSample['R_X_corr_7Li_6Li'])**2
      
        baxterSample['u^2_(Rx_RM)_7Li_6Li'] = (U_X_RM_7Li_6Li / R_X_RM_7Li_6Li)**2 / 3 * (baxterSample['R_X_corr_7Li_6Li'] )**2
     
        baxterSample['mean_for_t_test_7Li_6Li'] = baxterSample[['u^2_(Rx_RM)_7Li_6Li', 'u^2_(b_hat)_7Li_6Li', 'u^2_(r_is)_7Li_6Li', 'u^2_(Rx_RM)_7Li_6Li']].mean(1)
     
        baxterSample['u_c(R_x_corr)_7Li_6Li'] = np.sqrt(baxterSample['mean_for_t_test_7Li_6Li'])
    
        baxterSample['v_eff_7Li_6Li'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_7Li_6Li'] -3)  #### richtig??
       
        baxterSample['t_95_Cl_7Li_6Li'] = baxterSample['u_c(R_x_corr)_7Li_6Li'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_7Li_6Li']) #
    
        baxterSample['d_7Li_6Li'] = (( baxterSample['7Li/6Li'] / p_x_rm_mean_7Li_6Li_Standard) * (p_is_mean_62Ni_60Ni_Standard / baxterSample['7Li/6Li'])**b_hat_7Li_6Li -1) * 1000
     
        baxterSample['u_c_d_7Li_6Li'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_7Li_6Li'] + baxterSample['u^2_(b_hat)_7Li_6Li'] + baxterSample['u^2_(rx_RM)_7Li_6Li']) / baxterSample['R_X_corr_7Li_6Li']**2
      
    
        baxterSample['t_95_Cl_d_7Li_6Li'] = baxterSample['u_c_d_7Li_6Li'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_7Li_6Li']) #
     
    
    baxterSample_delta = baxterSample[['Inputfile', 'd_7Li_6Li', 'u_c_d_7Li_6Li']].copy()
    
    
    baxterSample_delta.rename(columns = {'Inputfile' : 'Filename'}, inplace = True)
    
    delta_cols = [col for col in baxterSample_delta if 'd_' in col]
    deltaresults = baxterSample_delta.groupby('Filename')[delta_cols].mean()
    deltaresults_round = np.round(deltaresults, 3)
    
    delta2sd = baxterSample_delta.groupby('Filename')[delta_cols].std(ddof = 0) * 2
    new_names_sd = [(i,'2SD+' + i) for i in delta2sd.iloc[:, 0:].columns.values]
    delta2sd.rename(columns = dict(new_names_sd), inplace = True)
    delta2sd_round = np.round(delta2sd, 3)
    results = pd.concat([deltaresults_round, delta2sd_round], axis = 1) 
    results = results[[item for items in zip(deltaresults_round.columns, delta2sd_round.columns) for item in items]]

## In future releases variable names need to be changed to Lithium, and not Copper_Nickel
    
    cols = [col for col in baxterSample if '6' in col]
    copper_nickel = baxterSample[['Inputfile', '6Li', '7Li']].copy().round(1)
    cols = [col for col in copper_nickel if '6' in col]
    copper_nickel_mean = copper_nickel.groupby('Inputfile')[cols].mean()
    copper_nickel_mean = copper_nickel_mean.round(1)
    
    copper_nickel['7Li/6Li'] = (copper_nickel_mean['7Li'] / copper_nickel_mean['6Li']).round(2)
    result_all = pd.concat([results, copper_nickel_mean], axis = 1)
    result_all = result_all.drop('u_c_d_7Li_6Li', axis = 1)
    result_all = result_all.drop('2SD+u_c_d_7Li_6Li', axis = 1)
    
    col_result = list(result_all.columns)
    col_result.insert(0, 'Lab Nummer')
    col_result = [l.replace('_','') for l in col_result]
    col_result = [i.split('+', 1)[0] for i in col_result]
    
    result_all.to_csv('Baxter_Export.csv', sep='\t', mode="w", header=True, index_label='Index_name')
    
    infiles = infile + '/'
    filehead = [x for x in os.listdir(infiles) if x .endswith('exp') and 's01' in x]
    filehead = infiles + str(filehead)[2:-2]
        
    df3 = pd.read_csv(filehead, skiprows = 1, nrows = 1, sep=' |\t', engine='python', index_col = False)
    df3.columns = range(df3.shape[1])
    df3.drop(df3.columns[0], axis = 1, inplace = True)
    df3.drop(df3.columns[1:3], axis = 1, inplace = True)
    
    reread = pd.read_csv('Baxter_Export.csv', sep='\t', index_col=False) # names = col_result,
    reread.columns = col_result
   
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('_').str.get(-1)  #make Filename nice
    reread['Lab Nummer'] = reread['Lab Nummer'].str.split('.').str.get(0)
    reread.insert(1,'Messdatum', '')
    reread.insert(0, 'Objekt', '')
    reread.insert(0, 'Orig. Nummer', '')
    reread.insert(0, 'Ma Nummer', '')
    
    reread['Messdatum'] = df3.values[0][0]
    
    reread['Referenz'] = 'NIST SRM 976'
    reread['Verwenden'] = ''
    reread['Bemerkung'] ='Baxter, Sprühkammer, Ni-Cone'
    
    
    reread.to_csv(resultnameLi_Baxter, sep = '\t', header=True, index=False)
    reread.drop(reread.columns[7:9], axis=1, inplace = True)
    reread.columns = ['\u03B47/6Li' if x=='d7Li6Li' else x for x in reread.columns]
    reread.to_excel(resultnameLiExcel_Baxter, index=False)
