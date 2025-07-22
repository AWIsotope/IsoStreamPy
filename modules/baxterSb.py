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
from modules.config import outfile_results_Sb, outfile_results_Sn, outfile_results_Ag, Baxter_Sb
from modules.config import WorkspaceVariableInput

file_path2 = os.path.join("modules", "variable_data_input.pkl")
# load variable with pickle
with open(file_path2, 'rb') as file2:
    standardinputpfad = pickle.load(file2)

infile = standardinputpfad
file_path = os.path.join("modules", "variable_data_outputfolder.pkl") ## Output: "output/" + option + '/' + today_year + '/' + ordnername
with open(file_path, 'rb') as file2:
    outputfolder = pickle.load(file2)


warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

pd.options.mode.chained_assignment = None
pd.set_option("display.precision", 15)
pd.options.display.float_format = '{:.15f}'.format
os.getcwd()
resultnameSb_Baxter = os.path.join(outfile_results_Sb, 'Sbdelta_Baxter.csv') 
resultnameSbExcel_Baxter = os.path.join(outfile_results_Sb, 'Sb_delta_Baxter.xlsx')


def baxterSb():
    global resultnameSb_Baxter
    global resultnameSbExcel_Baxter
    global infile
    baxter = pd.read_csv(Baxter_Sb + 'Baxter.csv', sep = '\t', engine = 'python', index_col = False)
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

    """from Lee et al. 1995 Newer: Isotopic compositions of the elements 2013 (IUPAC Technical Report) """

    R_X_RM_Sn = 0.032593 / 0.24229
    
    
    
    """" End of insertion of Lee et al. 1995 """
    
    masses = pd.DataFrame()
    masses.loc[0, '118Sn'] = 117.901606625
    masses.loc[0, '120Sn'] = 119.902202063
    masses.loc[0, '121Sb'] = 120.903810584
    masses.loc[0, '123Sb'] = 122.904211755
    
    abundances = pd.DataFrame()
    abundances.loc[0, '118Sn'] = 0.242178
    abundances.loc[0, '118Sn_error'] = 0.00009
    abundances.loc[0, '120Sn'] = 0.325974
    abundances.loc[0, '120Sn_error'] = 0.00009
    abundances.loc[0, '121Sb'] = 0.5721
    abundances.loc[0, '121Sb_error'] = 0.00005
    abundances.loc[0, '123Sb'] = 0.4279
    abundances.loc[0, '123Sb_error'] = 0.00005
    
    
    R_is_rm = abundances['120Sn'] / abundances['118Sn']
    R_X_RM_123Sb_121Sb = abundances['123Sb'] / abundances['121Sb']
    U_X_RM_123Sb_121Sb = R_X_RM_123Sb_121Sb*np.sqrt((abundances['123Sb_error']/abundances['123Sb'])**2+(abundances['121Sb_error']/abundances['121Sb'])**2)/2
    
    n = 0
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['ln_r_is'] = np.log(baxterStandardall['120Sn/118Sn'])
        baxterStandardall['ln_r_x_rm'] = np.log(baxterStandardall['123Sb/121Sb'])
        
     
    
    
    
    nRM_Standard = len(baxterStandardall.index)
    
    m_is_block_Standard = nRM_Standard
    m_x_block_Standard = nRM_Standard
    
    
    ##X_mean // remains the same for all masses! Küzel bedeuten "Zelle xxx in Baxter-Spreadsheet 116-120"
    x_mean_sum_ln_120Sn_118Sn_Standard = sum(baxterStandardall['ln_r_is'])  ## = AE8
    ln_r_is_120Sn_118Sn_Standard = x_mean_sum_ln_120Sn_118Sn_Standard / nRM_Standard   ## AE4
    p_is_mean_120Sn_118Sn_Standard = np.exp(ln_r_is_120Sn_118Sn_Standard) ## = AE5
    
    
    ## Y_mean // changes for every mass ratio
    y_mean_sum__ln123Sb_121Sb_Standard = sum(baxterStandardall['ln_r_x_rm']) ## = AF8
    
    ln_r_x_rm_123Sb_121Sb_Standard = y_mean_sum__ln123Sb_121Sb_Standard / nRM_Standard   ## AF4

    ### X_mean
    p_x_rm_mean_123Sb_121Sb_Standard = np.exp(ln_r_x_rm_123Sb_121Sb_Standard) ## = AE6
    
    
    
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['Xi-Xmean_123Sb_121Sb'] = baxterStandardall['ln_r_is'] - ln_r_is_120Sn_118Sn_Standard  ## column AH from row 12 down
        baxterStandardall['Yi-Ymean_123Sb_121Sb'] = baxterStandardall['ln_r_x_rm'] - ln_r_x_rm_123Sb_121Sb_Standard ## column AI from row 12 down
    
    
        baxterStandardall['xiyi_123Sb_121Sb'] = baxterStandardall['Xi-Xmean_123Sb_121Sb'] * baxterStandardall['Yi-Ymean_123Sb_121Sb']  # column AL
    
    
        baxterStandardall['xi^2_123Sb_121Sb'] = baxterStandardall['Xi-Xmean_123Sb_121Sb'] * baxterStandardall['Xi-Xmean_123Sb_121Sb'] # column AM
    
    
        baxterStandardall['yi^2_123Sb_121Sb'] = baxterStandardall['Yi-Ymean_123Sb_121Sb'] * baxterStandardall['Yi-Ymean_123Sb_121Sb'] # Column AN
    
    
    sum_xiyi_123Sb_121Sb = sum(baxterStandardall['xiyi_123Sb_121Sb']) # AL8
    
    
    sum_xi_exp_2_123Sb_121Sb = sum(baxterStandardall['xi^2_123Sb_121Sb']) # AM8
    
    
    sum_yi_exp_2_123Sb_121Sb = sum(baxterStandardall['yi^2_123Sb_121Sb']) # AN8

    
    b_hat_123Sb_121Sb = sum_xiyi_123Sb_121Sb / sum_xi_exp_2_123Sb_121Sb # AM4
    
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['yi_hat_123Sb_121Sb'] = baxterStandardall['Xi-Xmean_123Sb_121Sb'] * b_hat_123Sb_121Sb # column AO
    
        baxterStandardall['yi-yi_hat^2_123Sb_121Sb'] = (baxterStandardall['Yi-Ymean_123Sb_121Sb'] - baxterStandardall['yi_hat_123Sb_121Sb'])**2
    
        baxterStandardall['R_X_corr_123Sb_121Sb'] = baxterStandardall['123Sb/121Sb'] / p_x_rm_mean_123Sb_121Sb_Standard * (R_X_RM_123Sb_121Sb/((pow((baxterStandardall['120Sn/118Sn'] / p_is_mean_120Sn_118Sn_Standard), b_hat_123Sb_121Sb))))
         
        baxterStandardall['u^2_(rx_RM)_123Sb_121Sb'] = (baxterStandardall['123Sb/121Sb_1SE'] / baxterStandardall['123Sb/121Sb'])**2 * baxterStandardall['R_X_corr_123Sb_121Sb']**2
    
    sum_yi_yi_hat_square_123Sb_121Sb = sum(baxterStandardall['yi-yi_hat^2_123Sb_121Sb']) # AP8
    
    
    S_y_x_123Sb_121Sb = np.sqrt(sum_yi_yi_hat_square_123Sb_121Sb / (nRM_Standard - 1))  ## AM2
    
    
    sum_xi_xmean_square_123Sb_121Sb = sum_xi_exp_2_123Sb_121Sb - sum(baxterStandardall['Xi-Xmean_123Sb_121Sb'])**2 / nRM_Standard # AM3
    
    
    sm_b_hat_123Sb_121Sb = np.sqrt((S_y_x_123Sb_121Sb**2) / sum_xi_xmean_square_123Sb_121Sb)  # AM5
    
    R_X_RM_corr_mean_123Sb_121Sb = np.mean(baxterStandardall['R_X_corr_123Sb_121Sb'])
    
    massbias = pd.DataFrame()
    massbias['120Sn/118Sn'] = baxterStandardall['120Sn/118Sn'].copy()
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['u^2_(b_hat)_123Sb_121Sb'] = (sm_b_hat_123Sb_121Sb * np.log(baxterStandardall['120Sn/118Sn'] / p_is_mean_120Sn_118Sn_Standard) * R_X_RM_corr_mean_123Sb_121Sb)**2
       
    
        baxterStandardall['u^2_(r_is)_123Sb_121Sb'] = (baxterStandardall['120Sn/118Sn_1SE'] / baxterStandardall['120Sn/118Sn'] * b_hat_123Sb_121Sb * baxterStandardall['R_X_corr_123Sb_121Sb'])**2
     
        baxterStandardall['u^2_(Rx_RM)_123Sb_121Sb'] = (U_X_RM_123Sb_121Sb / R_X_RM_123Sb_121Sb)**2 / 3 * (baxterStandardall['R_X_corr_123Sb_121Sb'] )**2
      
        baxterStandardall['mean_for_t_test_123Sb_121Sb'] = baxterStandardall[['u^2_(Rx_RM)_123Sb_121Sb', 'u^2_(b_hat)_123Sb_121Sb', 'u^2_(r_is)_123Sb_121Sb', 'u^2_(Rx_RM)_123Sb_121Sb']].mean(1)
    
        baxterStandardall['u_c(R_x_corr)_123Sb_121Sb'] = np.sqrt(baxterStandardall['mean_for_t_test_123Sb_121Sb'])
     
    
        baxterStandardall['v_eff_123Sb_121Sb'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_123Sb_121Sb'] -3)  #### richtig??
     
        baxterStandardall['t_95_Cl_123Sb_121Sb'] = baxterStandardall['u_c(R_x_corr)_123Sb_121Sb'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_123Sb_121Sb']) #
    

    if sm_b_hat_123Sb_121Sb / b_hat_123Sb_121Sb < 0.15:
        pass

    bx_bis_123Sb_121Sb =  b_hat_123Sb_121Sb * (np.log(masses['120Sn'] / masses['118Sn'])) / (np.log(masses['123Sb'] / masses['121Sb']))
    
    
    ###################################### Sample
    
    
    
    n = 0
    for SampleIndex in baxterSample.index:
        baxterSample['ln_r_is'] = np.log(baxterSample['120Sn/118Sn'])
        baxterSample['ln_r_x_rm_123Sb_121Sb'] = np.log(baxterSample['123Sb/121Sb'])

    nRM_Sample = len(baxterSample.index)
    
    m_is_block_Sample = nRM_Sample
    m_x_block_Sample = nRM_Sample
    
    
    for SampleIndex in baxterSample.index:
        baxterSample['R_X_corr_123Sb_121Sb'] = baxterSample['123Sb/121Sb'] / p_x_rm_mean_123Sb_121Sb_Standard * (R_X_RM_123Sb_121Sb/((pow((baxterSample['120Sn/118Sn'] / p_is_mean_120Sn_118Sn_Standard), b_hat_123Sb_121Sb))))
      
        baxterSample['u^2_(rx_RM)_123Sb_121Sb'] = (baxterSample['123Sb/121Sb_1SE'] / baxterSample['123Sb/121Sb'])**2 * baxterSample['R_X_corr_123Sb_121Sb']**2
     
    
    for SampleIndex in baxterSample.index:
        baxterSample['u^2_(b_hat)_123Sb_121Sb'] = (sm_b_hat_123Sb_121Sb * np.log(baxterSample['120Sn/118Sn'] / p_is_mean_120Sn_118Sn_Standard) * R_X_RM_corr_mean_123Sb_121Sb)**2
      
    
        baxterSample['u^2_(r_is)_123Sb_121Sb'] = (baxterSample['120Sn/118Sn_1SE'] / baxterSample['120Sn/118Sn'] * b_hat_123Sb_121Sb * baxterSample['R_X_corr_123Sb_121Sb'])**2
      
        baxterSample['u^2_(Rx_RM)_123Sb_121Sb'] = (U_X_RM_123Sb_121Sb / R_X_RM_123Sb_121Sb)**2 / 3 * (baxterSample['R_X_corr_123Sb_121Sb'] )**2
     
        baxterSample['mean_for_t_test_123Sb_121Sb'] = baxterSample[['u^2_(Rx_RM)_123Sb_121Sb', 'u^2_(b_hat)_123Sb_121Sb', 'u^2_(r_is)_123Sb_121Sb', 'u^2_(Rx_RM)_123Sb_121Sb']].mean(1)
     
        baxterSample['u_c(R_x_corr)_123Sb_121Sb'] = np.sqrt(baxterSample['mean_for_t_test_123Sb_121Sb'])
      
    
        baxterSample['v_eff_123Sb_121Sb'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_123Sb_121Sb'] -3)  #### richtig??
       
        baxterSample['t_95_Cl_123Sb_121Sb'] = baxterSample['u_c(R_x_corr)_123Sb_121Sb'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_123Sb_121Sb']) #
    
    
    
        baxterSample['d_123Sb_121Sb'] = (( baxterSample['123Sb/121Sb'] / p_x_rm_mean_123Sb_121Sb_Standard) * (p_is_mean_120Sn_118Sn_Standard / baxterSample['120Sn/118Sn'])**b_hat_123Sb_121Sb -1) * 1000
     
        baxterSample['u_c_d_123Sb_121Sb'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_123Sb_121Sb'] + baxterSample['u^2_(b_hat)_123Sb_121Sb'] + baxterSample['u^2_(rx_RM)_123Sb_121Sb']) / baxterSample['R_X_corr_123Sb_121Sb']**2
      
    
        baxterSample['t_95_Cl_d_123Sb_121Sb'] = baxterSample['u_c_d_123Sb_121Sb'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_123Sb_121Sb']) #
     
    
    baxterSample_delta = baxterSample[['Inputfile', 'd_123Sb_121Sb', 'u_c_d_123Sb_121Sb']].copy()
    
    
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
    
## In future releases variable names need to be changed to Tin and Antimony, and not Copper_Nickel
    
    cols = [col for col in baxterSample if '6' in col]
    copper_nickel = baxterSample[['Inputfile', '121Sb', '118Sn']].copy().round(1)
    cols = [col for col in copper_nickel if '6' in col]
    copper_nickel_mean = copper_nickel.groupby('Inputfile')[cols].mean()
    copper_nickel_mean = copper_nickel_mean.round(1)
    
    copper_nickel['121Sb/118Sn'] = (copper_nickel_mean['121Sb'] / copper_nickel_mean['118Sn']).round(2)
    result_all = pd.concat([results, copper_nickel_mean], axis = 1)
    result_all = result_all.drop('u_c_d_123Sb_121Sb', axis = 1)
    result_all = result_all.drop('2SD+u_c_d_123Sb_121Sb', axis = 1)
    
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

    reread.to_csv(resultnameSb_Baxter, sep = '\t', header=True, index=False)
    reread.drop(reread.columns[7:9], axis=1, inplace = True)
    reread.columns = ['\u03B465/121Sb' if x=='d123Sb121Sb' else x for x in reread.columns]
    reread.to_excel(resultnameSbExcel_Baxter, index=False)
