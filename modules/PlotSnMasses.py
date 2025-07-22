# -*- coding: utf-8 -*-
"""

@author: Andreas Wittke
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
from sklearn.metrics import mean_squared_error, r2_score
from modules.config import outfile_results_Sn, outfile_plt_Sn
from modules.config import WorkspaceVariableInput
infile = WorkspaceVariableInput
fullnameSn = os.path.join(outfile_results_Sn, 'Sn_export.csv')
resultnameSn = os.path.join(outfile_results_Sn, 'Sn_delta.csv')
plotSn = os.path.join(outfile_results_Sn, 'Sn_plots.csv')
baxternameSn = os.path.join(outfile_results_Sn, 'Sn_delta_Baxter.csv')

def PlotSnMasses():
    global fullnameSn
    global resultnameSn
    global plotSn
    
    Sn_delta = pd.read_csv(resultnameSn, sep = '\t')
    
    fig, ax = plt.subplots(nrows =3, ncols = 2, figsize = (40, 30), label='Inline label', sharex = False)
    fig.suptitle('\u03B4 Plot of all Sn samples')
    fig.tight_layout()
    plt.rcParams.update({'font.size': 14})
    plt.subplots_adjust(left=0.1,
                bottom=0.1, 
                right=0.9, 
                top=0.9, 
                wspace=0.4, 
                hspace=0.4)
    
    # Plot d124Sn/120Sn vs d116Sn/120Sn
    ax[0, 0].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], color='r', marker='o')
    ax[0, 0].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.5'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    reg_ax0 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 1)
    predict_ax0 = np.poly1d(reg_ax0) # Slope and intercept
    trend_ax0 = np.polyval(reg_ax0, Sn_delta['d116Sn/120Sn'])
    std_ax0 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax0 = np.round(r2_score(Sn_delta['d124Sn/120Sn'], predict_ax0(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax0 = str(r2_ax0)
    
    ax[0, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax0, 'k--')
    ax[0, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax0 - std_ax0, 'c--')
    ax[0, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax0 + std_ax0, 'c--')
    ax[0, 0].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[0, 0].set_ylabel('\u03B4$^{124}$Sn/$^{120}$Sn')
    ax[0, 0].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax0)
    
    ax[0, 0].grid(True, which='both')
    ax[0, 0].axhline(y=0, color='k')
    ax[0, 0].axvline(x=0, color='k')
    ax[0, 0].set_xticks(ax[0, 0].get_xticks()[::1])
    ax[0, 0].legend(loc=1)
    
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d116Sn/120Sn']*1.02, Sn_delta['d124Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[0, 0])
    ax[0, 0].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[0, 0].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    
    # Plot d122Sn/120Sn vs d116Sn/120Sn
    ax[0, 1].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], color='r', marker='o')
    ax[0, 1].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.6'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    reg_ax1 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], 1)
    predict_ax1 = np.poly1d(reg_ax1) # Slope and intercept
    trend_ax1 = np.polyval(reg_ax1, Sn_delta['d116Sn/120Sn'])
    std_ax1 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax1 = np.round(r2_score(Sn_delta['d122Sn/120Sn'], predict_ax1(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax1 = str(r2_ax1)
    
    ax[0, 1].plot(Sn_delta['d116Sn/120Sn'], trend_ax1, 'k--')
    ax[0, 1].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 - std_ax1, 'c--')
    ax[0, 1].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 + std_ax1, 'c--')
    ax[0, 1].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[0, 1].set_ylabel('\u03B4$^{122}$Sn/$^{120}$Sn')
    ax[0, 1].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax1)
    
    ax[0, 1].grid(True, which='both')
    ax[0, 1].axhline(y=0, color='k')
    ax[0, 1].axvline(x=0, color='k')
    ax[0, 1].set_xticks(ax[0, 0].get_xticks()[::1])
    ax[0, 1].legend(loc=1)
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d116Sn/120Sn']*1.02, Sn_delta['d122Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[0, 1])
    
    ax[0, 1].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[0, 1].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    
    # Plot d124Sn/120Sn vs d118Sn/120Sn
    ax[1, 0].scatter(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], color='r', marker='o')
    ax[1, 0].errorbar(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], xerr=Sn_delta['2SD.2'], yerr=Sn_delta['2SD.5'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    reg_ax2 = np.polyfit(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 1)
    predict_ax2 = np.poly1d(reg_ax2) # Slope and intercept
    trend_ax2 = np.polyval(reg_ax2, Sn_delta['d118Sn/120Sn'])
    std_ax2 = Sn_delta['d118Sn/120Sn'].std() # Standard deviation
    r2_ax2 = np.round(r2_score(Sn_delta['d124Sn/120Sn'], predict_ax2(Sn_delta['d118Sn/120Sn'])), 5) #R-squared
    r2string_ax2 = str(r2_ax2)
    
    ax[1, 0].plot(Sn_delta['d118Sn/120Sn'], trend_ax2, 'k--')
    ax[1, 0].plot(Sn_delta['d118Sn/120Sn'], trend_ax2 - std_ax2, 'c--')
    ax[1, 0].plot(Sn_delta['d118Sn/120Sn'], trend_ax2 + std_ax2, 'c--')
    ax[1, 0].set_xlabel('\u03B4$^{118}$Sn/$^{120}$Sn')
    ax[1, 0].set_ylabel('\u03B4$^{124}$Sn/$^{120}$Sn')
    ax[1, 0].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax2)
    
    ax[1, 0].grid(True, which='both')
    ax[1, 0].axhline(y=0, color='k')
    ax[1, 0].axvline(x=0, color='k')
    ax[1, 0].set_xticks(ax[1, 0].get_xticks()[::1])
    ax[1, 0].legend(loc=1)
    ax[1, 0].set_xlim(-0.4, 0.1)
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d118Sn/120Sn']*1.02, Sn_delta['d124Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[1, 0])
    ax[1, 0].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[1, 0].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    # Plot d124Sn/120Sn vs d122Sn/120Sn
    ax[1, 1].scatter(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], color='r', marker='o')
    ax[1, 1].errorbar(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], xerr=Sn_delta['2SD.2'], yerr=Sn_delta['2SD.5'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    
    reg_ax1 = np.polyfit(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 1)
    predict_ax1 = np.poly1d(reg_ax1) # Slope and intercept
    trend_ax1 = np.polyval(reg_ax1, Sn_delta['d122Sn/120Sn'])
    std_ax1 = Sn_delta['d122Sn/120Sn'].std() # Standard deviation
    r2_ax1 = np.round(r2_score(Sn_delta['d124Sn/120Sn'], predict_ax1(Sn_delta['d122Sn/120Sn'])), 5) #R-squared
    r2string_ax1 = str(r2_ax1)
    
    ax[1, 1].plot(Sn_delta['d122Sn/120Sn'], trend_ax1, 'k--')
    ax[1, 1].plot(Sn_delta['d122Sn/120Sn'], trend_ax1 - std_ax1, 'c--')
    ax[1, 1].plot(Sn_delta['d122Sn/120Sn'], trend_ax1 + std_ax1, 'c--')
    ax[1, 1].set_xlabel('\u03B4$^{122}$Sn/$^{120}$Sn')
    ax[1, 1].set_ylabel('\u03B4$^{124}$Sn/$^{120}$Sn')
    ax[1, 1].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax1)
    
    ax[1, 1].grid(True, which='both')
    ax[1, 1].axhline(y=0, color='k')
    ax[1, 1].axvline(x=0, color='k')
    ax[1, 1].set_xlim(-0.1, 0.4)
    ax[1, 1].set_xticks(ax[1, 1].get_xticks()[::1])
    ax[1, 1].legend(loc=1)
    
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d122Sn/120Sn']*1.02, Sn_delta['d124Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[1, 1])
    
    
    ax[1, 1].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[1, 1].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)

    
    # Plot d118Sn/120Sn and d117Sn/119Sn vs d116Sn/120Sn
    ax[2, 0].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d118Sn/120Sn'], color='r', marker='o')
    ax[2, 0].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d118Sn/120Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.2'], linestyle="none", color = 'r', zorder = 0,  capsize=4)
    ax[2, 0].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d117Sn/119Sn'], color='b', marker='o')
    ax[2, 0].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d117Sn/119Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.8'], linestyle="none", color = 'b', zorder = 0,  capsize=4)
    
    
    reg_ax1 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d118Sn/120Sn'], 1)
    predict_ax1 = np.poly1d(reg_ax1) # Slope and intercept
    trend_ax1 = np.polyval(reg_ax1, Sn_delta['d116Sn/120Sn'])
    std_ax1 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax1 = np.round(r2_score(Sn_delta['d118Sn/120Sn'], predict_ax1(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax1 = str(r2_ax1)
    
    reg_ax2 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d117Sn/119Sn'], 1)
    predict_ax2 = np.poly1d(reg_ax2) # Slope and intercept
    trend_ax2 = np.polyval(reg_ax2, Sn_delta['d116Sn/120Sn'])
    std_ax2 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax2 = np.round(r2_score(Sn_delta['d117Sn/119Sn'], predict_ax2(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax2 = str(r2_ax2)
    
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax1, 'k--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 - std_ax1, 'c--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 + std_ax1, 'c--')
    ax[2, 0].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[2, 0].set_ylabel('\u03B4$^{118}$Sn/$^{120}$Sn / \u03B4$^{117}$Sn/$^{119}$Sn')
    ax[2, 0].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax1)
    
    ax[2, 0].grid(True, which='both')
    ax[2, 0].axhline(y=0, color='k')
    ax[2, 0].axvline(x=0, color='k')
    ax[2, 0].set_xticks(ax[2, 0].get_xticks()[::1])
    ax[2, 0].legend(loc=1)
    
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d116Sn/120Sn']*1.02, Sn_delta['d118Sn/120Sn']-0.1, Sn_delta['Lab Nummer'], ax[2, 0])
    
    
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax2, 'k--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax2 - std_ax2, 'c--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax2 + std_ax2, 'c--')
    ax[2, 0].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[2, 0].plot([], [], ' ', label='R$^{2}$ of d117Sn/119Sn = '+r2string_ax2)

    ax[2, 0].set_ylim(-0.4, 0.1)

    ax[2, 0].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[2, 0].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    
    ax[2,1].set_visible(False)
    
    fig.savefig(outfile_plt_Sn + "PlotSnMasses.pdf", dpi=200)
    
    plt.close()
    
    
    
def PlotSnBaxterMasses():
    global fullnameSn
    global resultnameSn
    global plotSn
    global baxternameSn
    
    Sn_delta = pd.read_csv(baxternameSn, sep = '\t')
    
    fig, ax = plt.subplots(nrows =3, ncols = 2, figsize = (40, 30), label='Inline label', sharex = False)
    fig.suptitle('\u03B4 Plot of all Sn samples')
    fig.tight_layout()
    plt.rcParams.update({'font.size': 14})
    plt.subplots_adjust(left=0.1,
                bottom=0.1, 
                right=0.9, 
                top=0.9, 
                wspace=0.4, 
                hspace=0.4)
    
    # Plot d124Sn/120Sn vs d116Sn/120Sn
    ax[0, 0].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], color='r', marker='o')
    ax[0, 0].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.5'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    reg_ax0 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 1)
    predict_ax0 = np.poly1d(reg_ax0) # Slope and intercept
    trend_ax0 = np.polyval(reg_ax0, Sn_delta['d116Sn/120Sn'])
    std_ax0 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax0 = np.round(r2_score(Sn_delta['d124Sn/120Sn'], predict_ax0(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax0 = str(r2_ax0)
    
    ax[0, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax0, 'k--')
    ax[0, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax0 - std_ax0, 'c--')
    ax[0, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax0 + std_ax0, 'c--')
    ax[0, 0].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[0, 0].set_ylabel('\u03B4$^{124}$Sn/$^{120}$Sn')
    ax[0, 0].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax0)
    
    ax[0, 0].grid(True, which='both')
    ax[0, 0].axhline(y=0, color='k')
    ax[0, 0].axvline(x=0, color='k')
    ax[0, 0].set_xticks(ax[0, 0].get_xticks()[::1])
    ax[0, 0].legend(loc=1)
    
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d116Sn/120Sn']*1.02, Sn_delta['d124Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[0, 0])
    ax[0, 0].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[0, 0].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    
    # Plot d122Sn/120Sn vs d116Sn/120Sn
    ax[0, 1].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], color='r', marker='o')
    ax[0, 1].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.6'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    reg_ax1 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], 1)
    predict_ax1 = np.poly1d(reg_ax1) # Slope and intercept
    trend_ax1 = np.polyval(reg_ax1, Sn_delta['d116Sn/120Sn'])
    std_ax1 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax1 = np.round(r2_score(Sn_delta['d122Sn/120Sn'], predict_ax1(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax1 = str(r2_ax1)
    
    ax[0, 1].plot(Sn_delta['d116Sn/120Sn'], trend_ax1, 'k--')
    ax[0, 1].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 - std_ax1, 'c--')
    ax[0, 1].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 + std_ax1, 'c--')
    ax[0, 1].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[0, 1].set_ylabel('\u03B4$^{122}$Sn/$^{120}$Sn')
    ax[0, 1].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax1)
    
    ax[0, 1].grid(True, which='both')
    ax[0, 1].axhline(y=0, color='k')
    ax[0, 1].axvline(x=0, color='k')
    ax[0, 1].set_xticks(ax[0, 0].get_xticks()[::1])
    ax[0, 1].legend(loc=1)
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d116Sn/120Sn']*1.02, Sn_delta['d122Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[0, 1])
    
    ax[0, 1].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[0, 1].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    
    # Plot d124Sn/120Sn vs d118Sn/120Sn
    ax[1, 0].scatter(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], color='r', marker='o')
    ax[1, 0].errorbar(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], xerr=Sn_delta['2SD.2'], yerr=Sn_delta['2SD.5'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    reg_ax2 = np.polyfit(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 1)
    predict_ax2 = np.poly1d(reg_ax2) # Slope and intercept
    trend_ax2 = np.polyval(reg_ax2, Sn_delta['d118Sn/120Sn'])
    std_ax2 = Sn_delta['d118Sn/120Sn'].std() # Standard deviation
    r2_ax2 = np.round(r2_score(Sn_delta['d124Sn/120Sn'], predict_ax2(Sn_delta['d118Sn/120Sn'])), 5) #R-squared
    r2string_ax2 = str(r2_ax2)
    
    ax[1, 0].plot(Sn_delta['d118Sn/120Sn'], trend_ax2, 'k--')
    ax[1, 0].plot(Sn_delta['d118Sn/120Sn'], trend_ax2 - std_ax2, 'c--')
    ax[1, 0].plot(Sn_delta['d118Sn/120Sn'], trend_ax2 + std_ax2, 'c--')
    ax[1, 0].set_xlabel('\u03B4$^{118}$Sn/$^{120}$Sn')
    ax[1, 0].set_ylabel('\u03B4$^{124}$Sn/$^{120}$Sn')
    ax[1, 0].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax2)
    
    ax[1, 0].grid(True, which='both')
    ax[1, 0].axhline(y=0, color='k')
    ax[1, 0].axvline(x=0, color='k')
    ax[1, 0].set_xticks(ax[1, 0].get_xticks()[::1])
    ax[1, 0].legend(loc=1)
    ax[1, 0].set_xlim(-0.4, 0.1)
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d118Sn/120Sn']*1.02, Sn_delta['d124Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[1, 0])
    ax[1, 0].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[1, 0].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    
    # Plot d124Sn/120Sn vs d122Sn/120Sn
    ax[1, 1].scatter(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], color='r', marker='o')
    ax[1, 1].errorbar(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], xerr=Sn_delta['2SD.2'], yerr=Sn_delta['2SD.5'], linestyle="none", color = 'k', zorder = 0,  capsize=4)
    
    reg_ax1 = np.polyfit(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 1)
    predict_ax1 = np.poly1d(reg_ax1) # Slope and intercept
    trend_ax1 = np.polyval(reg_ax1, Sn_delta['d122Sn/120Sn'])
    std_ax1 = Sn_delta['d122Sn/120Sn'].std() # Standard deviation
    r2_ax1 = np.round(r2_score(Sn_delta['d124Sn/120Sn'], predict_ax1(Sn_delta['d122Sn/120Sn'])), 5) #R-squared
    r2string_ax1 = str(r2_ax1)
    
    ax[1, 1].plot(Sn_delta['d122Sn/120Sn'], trend_ax1, 'k--')
    ax[1, 1].plot(Sn_delta['d122Sn/120Sn'], trend_ax1 - std_ax1, 'c--')
    ax[1, 1].plot(Sn_delta['d122Sn/120Sn'], trend_ax1 + std_ax1, 'c--')
    ax[1, 1].set_xlabel('\u03B4$^{122}$Sn/$^{120}$Sn')
    ax[1, 1].set_ylabel('\u03B4$^{124}$Sn/$^{120}$Sn')
    ax[1, 1].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax1)
    
    ax[1, 1].grid(True, which='both')
    ax[1, 1].axhline(y=0, color='k')
    ax[1, 1].axvline(x=0, color='k')
    ax[1, 1].set_xlim(-0.1, 0.4)
    ax[1, 1].set_xticks(ax[1, 1].get_xticks()[::1])
    ax[1, 1].legend(loc=1)
    
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d122Sn/120Sn']*1.02, Sn_delta['d124Sn/120Sn']*1.2, Sn_delta['Lab Nummer'], ax[1, 1])
    
    
    ax[1, 1].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[1, 1].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)

    
    # Plot d118Sn/120Sn and d117Sn/119Sn vs d116Sn/120Sn
    ax[2, 0].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d118Sn/120Sn'], color='r', marker='o')
    ax[2, 0].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d118Sn/120Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.2'], linestyle="none", color = 'r', zorder = 0,  capsize=4)
    ax[2, 0].scatter(Sn_delta['d116Sn/120Sn'], Sn_delta['d117Sn/119Sn'], color='b', marker='o')
    ax[2, 0].errorbar(Sn_delta['d116Sn/120Sn'], Sn_delta['d117Sn/119Sn'], xerr=Sn_delta['2SD'], yerr=Sn_delta['2SD.8'], linestyle="none", color = 'b', zorder = 0,  capsize=4)
    
    
    reg_ax1 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d118Sn/120Sn'], 1)
    predict_ax1 = np.poly1d(reg_ax1) # Slope and intercept
    trend_ax1 = np.polyval(reg_ax1, Sn_delta['d116Sn/120Sn'])
    std_ax1 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax1 = np.round(r2_score(Sn_delta['d118Sn/120Sn'], predict_ax1(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax1 = str(r2_ax1)
    
    reg_ax2 = np.polyfit(Sn_delta['d116Sn/120Sn'], Sn_delta['d117Sn/119Sn'], 1)
    predict_ax2 = np.poly1d(reg_ax2) # Slope and intercept
    trend_ax2 = np.polyval(reg_ax2, Sn_delta['d116Sn/120Sn'])
    std_ax2 = Sn_delta['d116Sn/120Sn'].std() # Standard deviation
    r2_ax2 = np.round(r2_score(Sn_delta['d117Sn/119Sn'], predict_ax2(Sn_delta['d116Sn/120Sn'])), 5) #R-squared
    r2string_ax2 = str(r2_ax2)
    
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax1, 'k--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 - std_ax1, 'c--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax1 + std_ax1, 'c--')
    ax[2, 0].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[2, 0].set_ylabel('\u03B4$^{118}$Sn/$^{120}$Sn / \u03B4$^{117}$Sn/$^{119}$Sn')
    ax[2, 0].plot([], [], ' ', label='R$^{2}$ = '+r2string_ax1)
    
    ax[2, 0].grid(True, which='both')
    ax[2, 0].axhline(y=0, color='k')
    ax[2, 0].axvline(x=0, color='k')
    ax[2, 0].set_xticks(ax[2, 0].get_xticks()[::1])
    ax[2, 0].legend(loc=1)
    
    
    def label_point(x, y, val, ax):
        a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
        for i, point in a.iterrows():
            ax.text(point['x'], point['y'], str(point['val']))
    label_point(Sn_delta['d116Sn/120Sn']*1.02, Sn_delta['d118Sn/120Sn']-0.1, Sn_delta['Lab Nummer'], ax[2, 0])
    
    
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax2, 'k--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax2 - std_ax2, 'c--')
    ax[2, 0].plot(Sn_delta['d116Sn/120Sn'], trend_ax2 + std_ax2, 'c--')
    ax[2, 0].set_xlabel('\u03B4$^{116}$Sn/$^{120}$Sn')
    ax[2, 0].plot([], [], ' ', label='R$^{2}$ of d117Sn/119Sn = '+r2string_ax2)

    ax[2, 0].set_ylim(-0.4, 0.1)

    ax[2, 0].axhspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    ax[2, 0].axvspan(-0.02, 0.02, facecolor='grey', alpha=0.1)
    

    
    ax[2,1].set_visible(False)
    
    fig.savefig(outfile_plt_Sn + "PlotSnMasses.pdf", dpi=200)
    
    plt.close()