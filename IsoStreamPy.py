# -*- coding: utf-8 -*-
"""
This is the main program of IsoStreamPy 
Run with "streamlit run IsoStreamPy.py"
@author: Dr. Andreas Wittke
"""

import streamlit as st
import time
import os
import pandas as pd
from csv import DictReader
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sb
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from statistics import mean
import modules.config as config
import modules.plotCu as plotCu
import modules.plotSb as plotSb
from datetime import date
import requests
import pickle
unique_key = str(time.time())
import sys
import signal
import subprocess
from modules.config import outfile_plt_Cu, outfile_corrected_raw_data_Cu, outfile_results_Cu, outfile_plt_Cu_Std
from modules.config import outfile_plt_Sn, outfile_corrected_raw_data_Sn, outfile_results_Sn, outfile_plt_Sn_Std
from modules import outlierSn
from modules.plotCu import plot_Cu_outlier_Blk, load_files, Voltages_Cu, plot_Cu_outlier_Std, plot_Cu_outlier_Sample
from modules.plotSn import plot_Sn_outlier_Blk, load_files, Voltages_Sn, plot_Sn_outlier_Std, plot_Sn_outlier_Sample, load_data_PlotSn, plot_Sn_Masses, PlotSnBaxterMasses
from modules.plot_Sn_Standards import load_data, plot_all, load_data_Baxter, plot_all_Baxter
from modules.plotSb import plot_Sb_outlier_Blk, load_files, Voltages_Sb, plot_Sb_outlier_Std, plot_Sb_outlier_Sample


st.set_page_config(layout="wide")

def kill_process_on_port(port):
    # Finde den Prozess, der den Port verwendet
    result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
    pids = result.stdout.strip().split("\n")

    # Beende jeden gefundenen Prozess
    for pid in pids:
        if pid:
            os.kill(int(pid), signal.SIGKILL)
            print(f"Prozess {pid} auf Port {port} wurde beendet.")

def restart_streamlit():
    # Beende alle Prozesse auf Port 8501
    # Starte die aktuelle Datei in einem neuen Prozess neu
    subprocess.Popen(['streamlit', 'run'] + sys.argv)  # `sys.argv` enthält das aktuelle Skript
    os._exit(0)  # Beende den aktuellen Prozess

def offer_download_button(file_path, label):
    if os.path.exists(file_path):
        # Datei im Binärmodus öffnen
        with open(file_path, "rb") as file:
            file_bytes = file.read()
            # Erstelle Download-Button
            st.download_button(
                label=label,
                data=file_bytes,
                file_name=os.path.basename(file_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.warning(f"Die Datei {os.path.basename(file_path)} existiert nicht.")


today = date.today()
today_dmy = today.strftime("%Y.%m.%d")
today_year = today.strftime("%Y")


# Erstellen Sie eine SessionState-Instanz

calculation_done = False


# Streamlit-Layout
st.title('IsoStreamPy - The Isotope Data Reduction and Visualization Dashboard')
st.header('More informations: https://github.com/AWIsotope/IsoStreamPy')

# Seitenleiste
st.sidebar.write('Main Panel')


    
with st.sidebar:
    if st.button("IsoPy neu starten"):
        st.success("IsoPy wird neu gestartet...")
        restart_streamlit()
    option = st.radio('Which Isotope System do you want to calculate?', ('Sn', 'Cu', 'Sb'))

    # Pfad basierend auf der Auswahl des Isotopensystems
    standardpfad = os.path.join("input", option, today_year)

    # Liste der vorhandenen Ordner im standardpfad
    vorhandene_ordner = [f for f in os.listdir(standardpfad) if os.path.isdir(os.path.join(standardpfad, f))]
    
    # Dropdown-Menü zur Auswahl des Ordners
    ordnername = st.selectbox('Select a folder', vorhandene_ordner, key='ordner_selector')

    # Den vollständigen Pfad des ausgewählten Ordners
    standardinputpfad = os.path.join(standardpfad, ordnername)
    
    outputfolder = "output/" + option + '/' + today_year + '/' + ordnername
    standardoutputpfad = "output/" + option + '/' + today_year + '/' + ordnername + "/results"

        
    file_path = os.path.join('modules', "variable_data.pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(standardoutputpfad, file)
        
    file_path2 = os.path.join('modules', "variable_data_input.pkl")
    with open(file_path2, 'wb') as file2:
        pickle.dump(standardinputpfad, file2)
        
    file_path3 = os.path.join('modules', "variable_data_outputfolder.pkl")
    with open(file_path3, 'wb') as file3:
        pickle.dump(outputfolder, file3)
    
    st.write("You chose: ", option)
    if option == 'Sn':
        if st.button('Press to run calculation'):
            with st.spinner("running... Please wait!"):
                time_start = time.perf_counter()
                exec(open('IsoPySn.py').read())
                st.write('Done!')
                time_end = time.perf_counter()
                st.write("The function took " "{:1.1f} seconds to complete  ".format(time_end - time_start))
        else:
            st.write("")
            
    elif option == 'Cu':
        if st.button('Press to run calculation'):
            with st.spinner("running... Please wait!"):
                time_start = time.perf_counter()
                exec(open('IsoPyCu.py').read())
                st.write('Done!')
                time_end = time.perf_counter()
                st.write("The function took " "{:1.1f} seconds to complete  ".format(time_end - time_start))
    
    elif option == 'Sb':
        if st.button('Press to run calculation'):
            with st.spinner("running... Please wait!"):
                time_start = time.perf_counter()
                exec(open('IsoPySb.py').read())
                st.write('Done!')
                time_end = time.perf_counter()
                st.write("The function took " "{:1.1f} seconds to complete  ".format(time_end - time_start))
    else:
        st.error("Ungültiges Isotopensystem. Bitte wählen Sie 'Cu', 'Sb' oder 'Sn'.")
        st.stop()
    
    # Beispiel: Für Sn
    if option == "Sn":
        file_sn = os.path.join(standardoutputpfad, "Sn_delta.xlsx")
        file_sn_baxter = os.path.join(standardoutputpfad, "Sn_delta_Baxter.xlsx")
        st.subheader("Sn Dateien")
        offer_download_button(file_sn, "Download Sn_delta.xlsx")
        offer_download_button(file_sn_baxter, "Download Sn_delta_Baxter.xlsx")
    
    # Für Cu
    elif option == "Cu":
        file_cu = os.path.join(standardoutputpfad, "Cu_delta.xlsx")
        file_cu_baxter = os.path.join(standardoutputpfad, "Cu_delta_Baxter.xlsx")
        st.subheader("Cu Dateien")
        offer_download_button(file_cu, "Download Cu_delta.xlsx")
        offer_download_button(file_cu_baxter, "Download Cu_delta_Baxter.xlsx")
    
    # Für Sb
    elif option == "Sb":
        file_sb = os.path.join(standardoutputpfad, "Sb_delta.xlsx")
        st.subheader("Sb Dateien")
        offer_download_button(file_sb, "Download Sb_delta.xlsx")


def load_files(path, extension):
    return sorted([f for f in os.listdir(path) if f.endswith(extension)])


if option == 'Cu':
    tabs = st.tabs(["Outlier Correction and Voltages", "Voltages", "Results"])

    with tabs[0]:
        def main():
            st.title("Outlier Detection")

            folder_path = standardinputpfad
            if folder_path:
                files = sorted([f for f in os.listdir(folder_path) if f.endswith('.TXT')])
                
                    
                selected_file = st.selectbox('Wählen Sie eine Datei aus', files, key='outliercorrectiontab')#, index=0)
                if selected_file is not None:
                    if selected_file.endswith('Blk.TXT') or selected_file.endswith('BLK_.TXT'):
                        plot_Cu_outlier_Blk(os.path.join(folder_path, selected_file))
                    elif selected_file.endswith('.TXT') and 's' in selected_file:
                        plot_Cu_outlier_Std(os.path.join(folder_path, selected_file))
                    else:
                        plot_Cu_outlier_Sample(os.path.join(folder_path, selected_file))
                    
                                        
        if __name__ == "__main__":
            main()

    with tabs[1]:

        def main():
            st.title("Voltages")

            folder_path = standardinputpfad

            if folder_path:
                files = load_files(folder_path, '.TXT')
                selected_file2 = st.selectbox('Select File for plotting of Voltages', files, key='file_selector_tab2', index=1)

                if selected_file2 is not None:
                    Voltages_Cu(os.path.join(folder_path, selected_file2))
        main()

    with tabs[2]:
        
        subtabs = st.tabs(["Blk", "Std", "Sample", "ln-Std Total", "d65Cu of Standards"])
        with subtabs[0]:
            files = load_files(standardinputpfad, 'Blk.TXT') + load_files(standardinputpfad, 'BLK_.TXT')
            selected_file = st.selectbox('Wählen Sie eine Blk-Datei aus', files, key='subtab_blk', index=0)
            if selected_file is not None:
                plotCu.plot_Cu_blk(os.path.join(standardinputpfad, selected_file))
        with subtabs[1]:
            files = [f for f in load_files(standardinputpfad, '.TXT') if 's' in f]
            selected_file = st.selectbox('Wählen Sie eine Standard-Datei aus', files, key='subtab_std', index=0)
            if selected_file is not None:
                plotCu.plot_Cu_std(os.path.join(standardinputpfad, selected_file))
        with subtabs[2]:
            files = [f for f in load_files(standardinputpfad, '.TXT') if not f.endswith('Blk.TXT') and not f.endswith('BLK_.TXT') and not 's' in f]
            selected_file = st.selectbox('Wählen Sie eine Sample-Datei aus', files, key='subtab_sample', index=0)
            if selected_file is not None:
                plotCu.plot_Cu_sample(os.path.join(standardinputpfad, selected_file))
        with subtabs[3]:
            fullname = os.path.join(standardoutputpfad, "Cu_export.csv")
            if os.path.exists(fullname):
                plotCu.plot_Cu_Standards()
            else:
                st.warning("Die benötigten Daten sind nicht vorhanden. Bitte klicken Sie auf den Button, um die Daten zu generieren.")

        with subtabs[4]:
            plotCu.plot_dCu_Standards()


if option == 'Sn':
    tabs = st.tabs(["Voltages", "Outlier Detection and ln Plots"])
            
    with tabs[0]:

        def main():
            st.title("Voltages")

            folder_path = standardinputpfad

            if folder_path:
                files = load_files(folder_path, '.TXT')
                selected_file2 = st.selectbox('Select File for plotting of Voltages', files, key='file_selector_tab2', index=1)

                if selected_file2 is not None:
                    Voltages_Sn(os.path.join(folder_path, selected_file2))
        main()

    with tabs[1]:
        
        subtabs = st.tabs(["Blk", "Std", "Sample", "ln-Std Total", "Sn Masses"])
        with subtabs[0]:
            files = load_files(standardinputpfad, 'Blk.TXT') + load_files(standardinputpfad, 'BLK_.TXT')
            selected_file = st.selectbox('Wählen Sie eine Blk-Datei aus', files, key='subtab_blk', index=0)
            if selected_file is not None:
                plot_Sn_outlier_Blk(os.path.join(standardinputpfad, selected_file))
        with subtabs[1]:
            files = [f for f in load_files(standardinputpfad, '.TXT') if 's' in f]
            selected_file = st.selectbox('Wählen Sie eine Standard-Datei aus', files, key='subtab_std', index=0)
            if selected_file is not None:
                plot_Sn_outlier_Std(os.path.join(standardinputpfad, selected_file))
        with subtabs[2]:
            files = [f for f in load_files(standardinputpfad, '.TXT') if not f.endswith('Blk.TXT') and not f.endswith('BLK_.TXT') and not 's' in f]
            selected_file = st.selectbox('Wählen Sie eine Sample-Datei aus', files, key='subtab_sample', index=0)
            if selected_file is not None:
                plot_Sn_outlier_Sample(os.path.join(standardinputpfad, selected_file))
                
        with subtabs[3]:
            try:
                # Versuche, die Daten zu laden und die Plots mit den Standardfunktionen zu generieren
                Sn_delta = load_data()  # Lade die Daten hier

                # Auswahlbox für den Plot
                plot_selection = st.selectbox("Wähle den Plot aus:", 
                                            ["d124Sn/120Sn vs d116Sn/120Sn",
                                            "d122Sn/120Sn vs d116Sn/120Sn",
                                            "d124Sn/120Sn vs d118Sn/120Sn",
                                            "d124Sn/120Sn vs d122Sn/120Sn",
                                            "d118Sn/120Sn und d117Sn/119Sn vs d116Sn/120Sn"])

                # Plots generieren
                fig = plot_all(Sn_delta, plot_selection)

            except FileNotFoundError as e:
                # Wenn die Datei nicht gefunden wird, alternative Funktionen verwenden
                st.warning(f"Sn_Delta not found: {str(e)}. Using Baxter instead...")
                Sn_delta = load_data_Baxter()  # Lade die Daten mit der Baxter-Methode

                # Auswahlbox für den Plot
                plot_selection = st.selectbox("Wähle den Plot aus:", 
                                            ["d124Sn120Sn vs d116Sn120Sn",
                                            "d122Sn120Sn vs d116Sn120Sn",
                                            "d124Sn120Sn vs d118Sn120Sn",
                                            "d124Sn120Sn vs d122Sn120Sn",
                                            "d118Sn120Sn und d117Sn119Sn vs d116Sn120Sn"])

                # Plots generieren mit der Baxter-Methode
                fig = plot_all_Baxter(Sn_delta, plot_selection)

            # Zeige das Diagramm an, wenn es erstellt wurde
            if fig:
                st.plotly_chart(fig)
                
        with subtabs[4]:
            try:
                # Versuche, die Daten zu laden und plot_Sn_Masses auszuführen
                Sn_export = load_data_PlotSn()
                fig3 = plot_Sn_Masses(Sn_export)
                st.plotly_chart(fig3)
            except FileNotFoundError as e:
                # Wenn die Datei nicht gefunden wird, PlotSnBaxterMasses aufrufen
                st.warning(f"Datei nicht gefunden: {str(e)}. Verwende stattdessen PlotSnBaxterMasses.")
                fig3 = PlotSnBaxterMasses()  # Ohne Parameter, wenn keine Daten erforderlich
                st.plotly_chart(fig3)
                
                
if option == 'Sb':
    tabs = st.tabs(["Outlier Correction and Voltages", "Voltages", "Results"])
    with tabs[0]:
        def main():
            st.title("Outlier Detection")

            folder_path = standardinputpfad
            if folder_path:
                files = sorted([f for f in os.listdir(folder_path) if f.endswith('.TXT')])
                
                    
                selected_file = st.selectbox('Wählen Sie eine Datei aus', files, key='outliercorrectiontab')#, index=0)
                if selected_file is not None:
                    if selected_file.endswith('Blk.TXT') or selected_file.endswith('BLK_.TXT'):
                        plot_Sb_outlier_Blk(os.path.join(folder_path, selected_file))
                    elif selected_file.endswith('.TXT') and 's' in selected_file:
                        plot_Sb_outlier_Std(os.path.join(folder_path, selected_file))
                    else:
                        plot_Sb_outlier_Sample(os.path.join(folder_path, selected_file))
                    
                                        
        if __name__ == "__main__":
            main()

    with tabs[1]:

        def main():
            st.title("Voltages")

            folder_path = standardinputpfad

            if folder_path:
                files = load_files(folder_path, '.TXT')
                selected_file2 = st.selectbox('Select File for plotting of Voltages', files, key='file_selector_tab2', index=1)

                if selected_file2 is not None:
                    Voltages_Sb(os.path.join(folder_path, selected_file2))
        main()

    with tabs[2]:
        
        subtabs = st.tabs(["Blk", "Std", "Sample", "ln-Std Total", "d65Cu of Standards"])
        with subtabs[0]:
            files = load_files(standardinputpfad, 'Blk.TXT') + load_files(standardinputpfad, 'BLK_.TXT')
            selected_file = st.selectbox('Wählen Sie eine Blk-Datei aus', files, key='subtab_blk', index=0)
            if selected_file is not None:
                plotSb.plot_Sb_blk(os.path.join(standardinputpfad, selected_file))
        with subtabs[1]:
            files = [f for f in load_files(standardinputpfad, '.TXT') if 'Nis' in f]
            selected_file = st.selectbox('Wählen Sie eine Standard-Datei aus', files, key='subtab_std', index=0)
            if selected_file is not None:
                plotSb.plot_Sb_std(os.path.join(standardinputpfad, selected_file))
        with subtabs[2]:
            files = [f for f in load_files(standardinputpfad, '.TXT') if not f.endswith('Blk.TXT') and not f.endswith('BLK_.TXT') and not 's' in f]
            selected_file = st.selectbox('Wählen Sie eine Sample-Datei aus', files, key='subtab_sample', index=0)
            if selected_file is not None:
                plotSb.plot_Sb_sample(os.path.join(standardinputpfad, selected_file))
        with subtabs[3]:
            fullname = os.path.join(standardoutputpfad, "Sb_export.csv")
            if os.path.exists(fullname):
                plotSb.plot_Sb_Standards()
            else:
                st.warning("Die benötigten Daten sind nicht vorhanden. Bitte klicken Sie auf den Button, um die Daten zu generieren.")

        with subtabs[4]:
            plotSb.plot_dSb_Standards()
