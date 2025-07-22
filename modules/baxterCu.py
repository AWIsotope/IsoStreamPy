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
from modules.config import outfile_results_Cu, outfile_results_Sn, outfile_results_Ag, Baxter_Cu
from modules.config import WorkspaceVariableInput
# infile = WorkspaceVariableInput

file_path2 = os.path.join("modules", "variable_data_input.pkl")
# Variable mit pickle laden
with open(file_path2, 'rb') as file2:
    standardinputpfad = pickle.load(file2)

infile = standardinputpfad


warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

pd.options.mode.chained_assignment = None
pd.set_option("display.precision", 15)
pd.options.display.float_format = '{:.15f}'.format
os.getcwd()
resultnameCu_Baxter = os.path.join(outfile_results_Cu, 'Cudelta_Baxter.csv') 
resultnameCuExcel_Baxter = os.path.join(outfile_results_Cu, 'Cu_delta_Baxter.xlsx')


def baxterCu():
    global resultnameCu_Baxter
    global resultnameCuExcel_Baxter
    global infile
    baxter = pd.read_csv(Baxter_Cu + 'Baxter.csv', sep = '\t', engine = 'python', index_col = False)
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
    
    
    
    """from Lee et al. 1995"""
    R_X_RM_Ni = 0.036345 / 0.262231
            
    """" End of insertion of Lee et al. 1995 """
    
    masses = pd.DataFrame()
    masses.loc[0, '60Ni'] = 57.9353462
    masses.loc[0, '62Ni'] = 61.9283461
    masses.loc[0, '63Cu'] = 62.9295989
    masses.loc[0, '65Cu'] = 64.9277929
    
    abundances = pd.DataFrame()
    abundances.loc[0, '60Ni'] = 0.262231
    abundances.loc[0, '60Ni_error'] = 0.000077
    abundances.loc[0, '62Ni'] = 0.036345
    abundances.loc[0, '62Ni_error'] = 0.000017
    abundances.loc[0, '63Cu'] = 0.6915
    abundances.loc[0, '63Cu_error'] = 0.0015
    abundances.loc[0, '65Cu'] = 0.3085
    abundances.loc[0, '65Cu_error'] = 0.0015
    
    
    R_is_rm = 0.036345 / 0.262231
    R_X_RM_65Cu_63Cu = abundances['65Cu'] / abundances['63Cu']
    U_X_RM_65Cu_63Cu = R_X_RM_65Cu_63Cu*np.sqrt((abundances['65Cu_error']/abundances['65Cu'])**2+(abundances['63Cu_error']/abundances['63Cu'])**2)/2
    
    n = 0
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['ln_r_is'] = np.log(baxterStandardall['62Ni/60Ni'])
        baxterStandardall['ln_r_x_rm'] = np.log(baxterStandardall['65Cu/63Cu'])
        
     
    
    
    
    nRM_Standard = len(baxterStandardall.index)
    
    m_is_block_Standard = nRM_Standard
    m_x_block_Standard = nRM_Standard
    
    
    ##X_mean // remains the same for all masses! Küzel bedeuten "Zelle xxx in Baxter-Spreadsheet 116-120"
    x_mean_sum_ln_62Ni_60Ni_Standard = sum(baxterStandardall['ln_r_is'])  ## = AE8
    ln_r_is_62Ni_60Ni_Standard = x_mean_sum_ln_62Ni_60Ni_Standard / nRM_Standard   ## AE4
    p_is_mean_62Ni_60Ni_Standard = np.exp(ln_r_is_62Ni_60Ni_Standard) ## = AE5
    
    
    ## Y_mean // changes for every mass ratio
    y_mean_sum__ln65Cu_63Cu_Standard = sum(baxterStandardall['ln_r_x_rm']) ## = AF8
    
       
    ln_r_x_rm_65Cu_63Cu_Standard = y_mean_sum__ln65Cu_63Cu_Standard / nRM_Standard   ## AF4
    
         
    ### X_mean
    p_x_rm_mean_65Cu_63Cu_Standard = np.exp(ln_r_x_rm_65Cu_63Cu_Standard) ## = AE6
    
        
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['Xi-Xmean_65Cu_63Cu'] = baxterStandardall['ln_r_is'] - ln_r_is_62Ni_60Ni_Standard  ## column AH from row 12 down
        baxterStandardall['Yi-Ymean_65Cu_63Cu'] = baxterStandardall['ln_r_x_rm'] - ln_r_x_rm_65Cu_63Cu_Standard ## column AI from row 12 down
    
    
        baxterStandardall['xiyi_65Cu_63Cu'] = baxterStandardall['Xi-Xmean_65Cu_63Cu'] * baxterStandardall['Yi-Ymean_65Cu_63Cu']  # column AL
    
    
        baxterStandardall['xi^2_65Cu_63Cu'] = baxterStandardall['Xi-Xmean_65Cu_63Cu'] * baxterStandardall['Xi-Xmean_65Cu_63Cu'] # column AM
    
    
        baxterStandardall['yi^2_65Cu_63Cu'] = baxterStandardall['Yi-Ymean_65Cu_63Cu'] * baxterStandardall['Yi-Ymean_65Cu_63Cu'] # Column AN
    
    
    sum_xiyi_65Cu_63Cu = sum(baxterStandardall['xiyi_65Cu_63Cu']) # AL8
    
    
    sum_xi_exp_2_65Cu_63Cu = sum(baxterStandardall['xi^2_65Cu_63Cu']) # AM8
    
    
    sum_yi_exp_2_65Cu_63Cu = sum(baxterStandardall['yi^2_65Cu_63Cu']) # AN8
    
    
    
    
    b_hat_65Cu_63Cu = sum_xiyi_65Cu_63Cu / sum_xi_exp_2_65Cu_63Cu # AM4
    
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['yi_hat_65Cu_63Cu'] = baxterStandardall['Xi-Xmean_65Cu_63Cu'] * b_hat_65Cu_63Cu # column AO
    
        baxterStandardall['yi-yi_hat^2_65Cu_63Cu'] = (baxterStandardall['Yi-Ymean_65Cu_63Cu'] - baxterStandardall['yi_hat_65Cu_63Cu'])**2
    
        baxterStandardall['R_X_corr_65Cu_63Cu'] = baxterStandardall['65Cu/63Cu'] / p_x_rm_mean_65Cu_63Cu_Standard * (R_X_RM_65Cu_63Cu/((pow((baxterStandardall['62Ni/60Ni'] / p_is_mean_62Ni_60Ni_Standard), b_hat_65Cu_63Cu))))
         
        baxterStandardall['u^2_(rx_RM)_65Cu_63Cu'] = (baxterStandardall['65Cu/63Cu_1SE'] / baxterStandardall['65Cu/63Cu'])**2 * baxterStandardall['R_X_corr_65Cu_63Cu']**2
    
    sum_yi_yi_hat_square_65Cu_63Cu = sum(baxterStandardall['yi-yi_hat^2_65Cu_63Cu']) # AP8
    
    
    S_y_x_65Cu_63Cu = np.sqrt(sum_yi_yi_hat_square_65Cu_63Cu / (nRM_Standard - 1))  ## AM2
    
    
    sum_xi_xmean_square_65Cu_63Cu = sum_xi_exp_2_65Cu_63Cu - sum(baxterStandardall['Xi-Xmean_65Cu_63Cu'])**2 / nRM_Standard # AM3
    
    
    sm_b_hat_65Cu_63Cu = np.sqrt((S_y_x_65Cu_63Cu**2) / sum_xi_xmean_square_65Cu_63Cu)  # AM5

    
    R_X_RM_corr_mean_65Cu_63Cu = np.mean(baxterStandardall['R_X_corr_65Cu_63Cu'])
    
    massbias = pd.DataFrame()
    massbias['62Ni/60Ni'] = baxterStandardall['62Ni/60Ni'].copy()
    
    for StandardIndex in baxterStandardall.index:
        baxterStandardall['u^2_(b_hat)_65Cu_63Cu'] = (sm_b_hat_65Cu_63Cu * np.log(baxterStandardall['62Ni/60Ni'] / p_is_mean_62Ni_60Ni_Standard) * R_X_RM_corr_mean_65Cu_63Cu)**2
       
    
        baxterStandardall['u^2_(r_is)_65Cu_63Cu'] = (baxterStandardall['62Ni/60Ni_1SE'] / baxterStandardall['62Ni/60Ni'] * b_hat_65Cu_63Cu * baxterStandardall['R_X_corr_65Cu_63Cu'])**2
     
        baxterStandardall['u^2_(Rx_RM)_65Cu_63Cu'] = (U_X_RM_65Cu_63Cu / R_X_RM_65Cu_63Cu)**2 / 3 * (baxterStandardall['R_X_corr_65Cu_63Cu'] )**2
      
        baxterStandardall['mean_for_t_test_65Cu_63Cu'] = baxterStandardall[['u^2_(Rx_RM)_65Cu_63Cu', 'u^2_(b_hat)_65Cu_63Cu', 'u^2_(r_is)_65Cu_63Cu', 'u^2_(Rx_RM)_65Cu_63Cu']].mean(1)
    
        baxterStandardall['u_c(R_x_corr)_65Cu_63Cu'] = np.sqrt(baxterStandardall['mean_for_t_test_65Cu_63Cu'])
     
    
        baxterStandardall['v_eff_65Cu_63Cu'] = round(nRM_Standard + nRM_Standard + baxterStandardall['u^2_(b_hat)_65Cu_63Cu'] -3)  #### richtig??
     
        baxterStandardall['t_95_Cl_65Cu_63Cu'] = baxterStandardall['u_c(R_x_corr)_65Cu_63Cu'] * stats.t.ppf(1-0.05/2, baxterStandardall['v_eff_65Cu_63Cu']) #
    

    
    if sm_b_hat_65Cu_63Cu / b_hat_65Cu_63Cu < 0.15:
        pass
    

    
    
    bx_bis_65Cu_63Cu =  b_hat_65Cu_63Cu * (np.log(masses['62Ni'] / masses['60Ni'])) / (np.log(masses['65Cu'] / masses['63Cu']))
    
    
    ###################################### Sample
    
    
    
    n = 0
    for SampleIndex in baxterSample.index:
        baxterSample['ln_r_is'] = np.log(baxterSample['62Ni/60Ni'])
        baxterSample['ln_r_x_rm_65Cu_63Cu'] = np.log(baxterSample['65Cu/63Cu'])
        
        
 
    nRM_Sample = len(baxterSample.index)
    
    m_is_block_Sample = nRM_Sample
    m_x_block_Sample = nRM_Sample
    
    
    for SampleIndex in baxterSample.index:
        baxterSample['R_X_corr_65Cu_63Cu'] = baxterSample['65Cu/63Cu'] / p_x_rm_mean_65Cu_63Cu_Standard * (R_X_RM_65Cu_63Cu/((pow((baxterSample['62Ni/60Ni'] / p_is_mean_62Ni_60Ni_Standard), b_hat_65Cu_63Cu))))
      
        baxterSample['u^2_(rx_RM)_65Cu_63Cu'] = (baxterSample['65Cu/63Cu_1SE'] / baxterSample['65Cu/63Cu'])**2 * baxterSample['R_X_corr_65Cu_63Cu']**2
     
    
    for SampleIndex in baxterSample.index:
        baxterSample['u^2_(b_hat)_65Cu_63Cu'] = (sm_b_hat_65Cu_63Cu * np.log(baxterSample['62Ni/60Ni'] / p_is_mean_62Ni_60Ni_Standard) * R_X_RM_corr_mean_65Cu_63Cu)**2
      
    
        baxterSample['u^2_(r_is)_65Cu_63Cu'] = (baxterSample['62Ni/60Ni_1SE'] / baxterSample['62Ni/60Ni'] * b_hat_65Cu_63Cu * baxterSample['R_X_corr_65Cu_63Cu'])**2
      
        baxterSample['u^2_(Rx_RM)_65Cu_63Cu'] = (U_X_RM_65Cu_63Cu / R_X_RM_65Cu_63Cu)**2 / 3 * (baxterSample['R_X_corr_65Cu_63Cu'] )**2
     
        baxterSample['mean_for_t_test_65Cu_63Cu'] = baxterSample[['u^2_(Rx_RM)_65Cu_63Cu', 'u^2_(b_hat)_65Cu_63Cu', 'u^2_(r_is)_65Cu_63Cu', 'u^2_(Rx_RM)_65Cu_63Cu']].mean(1)
     
        baxterSample['u_c(R_x_corr)_65Cu_63Cu'] = np.sqrt(baxterSample['mean_for_t_test_65Cu_63Cu'])
      
    
        baxterSample['v_eff_65Cu_63Cu'] = round(nRM_Sample + nRM_Sample + baxterSample['u^2_(b_hat)_65Cu_63Cu'] -3)  #### richtig??
       
        baxterSample['t_95_Cl_65Cu_63Cu'] = baxterSample['u_c(R_x_corr)_65Cu_63Cu'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_65Cu_63Cu']) #
    
    
    
        baxterSample['d_65Cu_63Cu'] = (( baxterSample['65Cu/63Cu'] / p_x_rm_mean_65Cu_63Cu_Standard) * (p_is_mean_62Ni_60Ni_Standard / baxterSample['62Ni/60Ni'])**b_hat_65Cu_63Cu -1) * 1000
     
        baxterSample['u_c_d_65Cu_63Cu'] = 1000 * np.sqrt(baxterSample['u^2_(r_is)_65Cu_63Cu'] + baxterSample['u^2_(b_hat)_65Cu_63Cu'] + baxterSample['u^2_(rx_RM)_65Cu_63Cu']) / baxterSample['R_X_corr_65Cu_63Cu']**2
      
    
        baxterSample['t_95_Cl_d_65Cu_63Cu'] = baxterSample['u_c_d_65Cu_63Cu'] * stats.t.ppf(1-0.05/2, baxterSample['v_eff_65Cu_63Cu']) #
     
    
    baxterSample_delta = baxterSample[['Inputfile', 'd_65Cu_63Cu', 'u_c_d_65Cu_63Cu']].copy()
    
    
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
    
    cols = [col for col in baxterSample if '6' in col]
    copper_nickel = baxterSample[['Inputfile', '63Cu', '60Ni']].copy().round(1)
    cols = [col for col in copper_nickel if '6' in col]
    copper_nickel_mean = copper_nickel.groupby('Inputfile')[cols].mean()
    copper_nickel_mean = copper_nickel_mean.round(1)
    
    copper_nickel['63Cu/60Ni'] = (copper_nickel_mean['63Cu'] / copper_nickel_mean['60Ni']).round(2)
    result_all = pd.concat([results, copper_nickel_mean], axis = 1)
    result_all = result_all.drop('u_c_d_65Cu_63Cu', axis = 1)
    result_all = result_all.drop('2SD+u_c_d_65Cu_63Cu', axis = 1)
    
    
    
    
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
    
    
    reread.to_csv(resultnameCu_Baxter, sep = '\t', header=True, index=False)
    reread.drop(reread.columns[7:9], axis=1, inplace = True)
    reread.columns = ['\u03B465/63Cu' if x=='d65Cu63Cu' else x for x in reread.columns]
    reread.to_excel(resultnameCuExcel_Baxter, index=False)


