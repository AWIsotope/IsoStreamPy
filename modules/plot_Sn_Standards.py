"""
@author: Andreas Wittke
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pickle

def load_data():
    # Datei mit Pickle laden
    file_path = os.path.join("modules", "variable_data.pkl")
    with open(file_path, 'rb') as file:
        data_directory = pickle.load(file)  # Lade den Pfad
    # Jetzt lade die CSV-Datei mit dem geladenen Pfad
    csv_path = os.path.join(data_directory, "Sn_delta.csv")
    Sn_delta = pd.read_csv(csv_path, sep='\t')  # Passe den Separator an
    return Sn_delta

def plot_Sn_Standards(x, y, x_label, y_label, xerr, yerr, data_label):
    # Plotly Scatter-Plot
    fig = go.Figure()

    # Scatter-Punkte hinzufügen
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=data_label,
        textposition='top center',
        name='Datenpunkte'
    ))

    # Fehlerbalken hinzufügend
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color='black', width=0),
        error_x=dict(type='data', array=xerr, visible=True, color='black'),
        error_y=dict(type='data', array=yerr, visible=True, color='black'),
        name='Fehlerbalken'
    ))

    # Trendlinie berechnen
    reg = np.polyfit(x, y, 1)
    trend_x = np.linspace(x.min() - 0.1, x.max() + 0.1, 500)
    trend_y = np.polyval(reg, trend_x)
    
    # Trendlinie hinzufügen
    fig.add_trace(go.Scatter(
        x=trend_x,
        y=trend_y,
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Trendlinie'
    ))

    # Achsen schneiden
    fig.add_shape(type='line',
                  x0=0, y0=y.min() - 0.1, x1=0, y1=y.max() + 0.1,
                  line=dict(color='black', width=1))
    fig.add_shape(type='line',
                  x0=x.min() - 0.1, y0=0, x1=x.max() + 0.1, y1=0,
                  line=dict(color='black', width=1))

    fig.add_shape(type='rect',
                  x0=x.min() - 0.1, y0=-0.02, x1=x.max() + 0.1, y1=0.02,  # Von -1.5 bis 1.5 entlang der x-Achse
                  line=dict(color='grey', width=0),
                  fillcolor='grey', opacity=0.1)
    
    # Graue Bereiche für die Y-Achse
    fig.add_shape(type='rect',
                  x0=-0.02, y0=y.min() - 0.1, x1=0.02, y1=y.max() + 0.1,  # Von -1.5 bis 1.5 entlang der y-Achse
                  line=dict(color='grey', width=0),
                  fillcolor='grey', opacity=0.1)
    # Layout anpassen
    fig.update_layout(
        title=f'{x_label} vs {y_label}',
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=800,  # Höhe des Plots erhöhen
        showlegend=True
    )

    return fig

def plot_all(Sn_delta, plot_selection):
    # Aufruf der Plot-Funktion mit den entsprechenden Argumenten
    if plot_selection == "d124Sn/120Sn vs d116Sn/120Sn":
        return plot_Sn_Standards(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 
                                  '\u03B4$^{116}$Sn/$^{120}$Sn', '\u03B4$^{124}$Sn/$^{120}$Sn', 
                                  Sn_delta['2SD'], Sn_delta['2SD.5'], Sn_delta['Lab Nummer'])

    elif plot_selection == "d122Sn/120Sn vs d116Sn/120Sn":
        return plot_Sn_Standards(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], 
                                  '\u03B4$^{116}$Sn/$^{120}$Sn', '\u03B4$^{122}$Sn/$^{120}$Sn', 
                                  Sn_delta['2SD'], Sn_delta['2SD.6'], Sn_delta['Lab Nummer'])

    elif plot_selection == "d124Sn/120Sn vs d118Sn/120Sn":
        return plot_Sn_Standards(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 
                                  '\u03B4$^{118}$Sn/$^{120}$Sn', '\u03B4$^{124}$Sn/$^{120}$Sn', 
                                  Sn_delta['2SD.2'], Sn_delta['2SD.5'], Sn_delta['Lab Nummer'])

    elif plot_selection == "d124Sn/120Sn vs d122Sn/120Sn":
        return plot_Sn_Standards(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 
                                  '\u03B4$^{122}$Sn/$^{120}$Sn', '\u03B4$^{124}$Sn/$^{120}$Sn', 
                                  Sn_delta['2SD.2'], Sn_delta['2SD.5'], Sn_delta['Lab Nummer'])

    elif plot_selection == "d118Sn/120Sn und d117Sn/119Sn vs d116Sn/120Sn":
        fig = go.Figure()
        x = Sn_delta['d116Sn/120Sn']
        y1 = Sn_delta['d118Sn/120Sn']
        y2 = Sn_delta['d117Sn/119Sn']
        
        # Scatter-Plot für beide Werte
        fig.add_trace(go.Scatter(
            x=x,
            y=y1,
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=Sn_delta['Lab Nummer'],
            textposition='top center',
            name='\u03B4$^{118}$Sn/$^{120}$Sn'
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=y2,
            mode='markers+text',
            marker=dict(size=10, color='blue'),
            text=Sn_delta['Lab Nummer'],
            textposition='top center',
            name='\u03B4$^{117}$Sn/$^{119}$Sn'
        ))
        # Fehlerbalken für y1 hinzufügen
        fig.add_trace(go.Scatter(
            x=x,
            y=y1,
            mode='lines',
            line=dict(color='black', width=0),
            error_x=dict(type='data', array=Sn_delta['2SD'], visible=True, color='black'),  # Fehlerbalken für X-Achse
            error_y=dict(type='data', array=Sn_delta['2SD.2'], visible=True, color='black'),  # Fehlerbalken für Y-Achse
            name='Fehlerbalken d118Sn/120Sn'
        ))

        # Fehlerbalken für y2 hinzufügen
        fig.add_trace(go.Scatter(
            x=x,
            y=y2,
            mode='lines',
            line=dict(color='black', width=0),
            error_x=dict(type='data', array=Sn_delta['2SD'], visible=True, color='black'),  # Fehlerbalken für X-Achse
            error_y=dict(type='data', array=Sn_delta['2SD.8'], visible=True, color='black'),  # Fehlerbalken für Y-Achse
            name='Fehlerbalken d117Sn/119Sn'
        ))
        
        
        # Trendlinien berechnen
        reg1 = np.polyfit(x, y1, 1)
        reg2 = np.polyfit(x, y2, 1)
        trend_x = np.linspace(x.min() - 0.1, x.max() + 0.1, 500)
        trend_y1 = np.polyval(reg1, trend_x)
        trend_y2 = np.polyval(reg2, trend_x)

        # Trendlinien hinzufügen
        fig.add_trace(go.Scatter(
            x=trend_x,
            y=trend_y1,
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Trend d118Sn/120Sn'
        ))

        fig.add_trace(go.Scatter(
            x=trend_x,
            y=trend_y2,
            mode='lines',
            line=dict(color='green', dash='dash'),
            name='Trend d117Sn/119Sn'
        ))

        # Achsen schneiden
        fig.add_shape(type='line',
                  x0=0, y0=y1.min() - 0.1, x1=0, y1=y1.max() + 0.1,
                  line=dict(color='black', width=1))
        fig.add_shape(type='line',
                  x0=x.min() - 0.1, y0=0, x1=x.max() + 0.1, y1=0,
                  line=dict(color='black', width=1))
        
        
        fig.add_shape(type='rect',
                  x0=x.min() - 0.1, y0=-0.02, x1=x.max() + 0.1, y1=0.02,  # Von -1.5 bis 1.5 entlang der x-Achse
                  line=dict(color='grey', width=0),
                  fillcolor='grey', opacity=0.1)
    
    # Graue Bereiche für die Y-Achse
        fig.add_shape(type='rect',
                  x0=-0.02, y0=y1.min() - 0.1, x1=0.02, y1=y1.max() + 0.1,  # Von -1.5 bis 1.5 entlang der y-Achse
                  line=dict(color='grey', width=0),
                  fillcolor='grey', opacity=0.1)

        # Layout anpassen
        fig.update_layout(
            title='d118Sn/120Sn und d117Sn/119Sn vs d116Sn/120Sn',
            xaxis_title='\u03B4$^{116}$Sn/$^{120}$Sn',
            yaxis_title='\u03B4$^{118}$Sn/$^{120}$Sn / \u03B4$^{117}$Sn/$^{119}$Sn',
            height=800,  # Höhe des Plots erhöhen
            showlegend=True
        )

        return fig


def load_data_Baxter():
    # Datei mit Pickle laden
    file_path = os.path.join("modules", "variable_data.pkl")
    with open(file_path, 'rb') as file:
        data_directory = pickle.load(file)  # Lade den Pfad
    # Jetzt lade die CSV-Datei mit dem geladenen Pfad
    csv_path = os.path.join(data_directory, "Sn_delta_Baxter.csv")
    Sn_delta = pd.read_csv(csv_path, sep='\t')  # Passe den Separator an
    return Sn_delta

def plot_Sn_Standards_Baxter(x, y, x_label, y_label, xerr, yerr, data_label):
    # Plotly Scatter-Plot
    fig = go.Figure()

    # Scatter-Punkte hinzufügen
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=data_label,
        textposition='top center',
        name='Datenpunkte'
    ))

    # Fehlerbalken hinzufügend
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=dict(color='black', width=0),
        error_x=dict(type='data', array=xerr, visible=True, color='black'),
        error_y=dict(type='data', array=yerr, visible=True, color='black'),
        name='Fehlerbalken'
    ))

    # Trendlinie berechnen
    reg = np.polyfit(x, y, 1)
    trend_x = np.linspace(x.min() - 0.1, x.max() + 0.1, 500)
    trend_y = np.polyval(reg, trend_x)
    
    # Trendlinie hinzufügen
    fig.add_trace(go.Scatter(
        x=trend_x,
        y=trend_y,
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Trendlinie'
    ))

    # Achsen schneiden
    fig.add_shape(type='line',
                  x0=0, y0=y.min() - 0.1, x1=0, y1=y.max() + 0.1,
                  line=dict(color='black', width=1))
    fig.add_shape(type='line',
                  x0=x.min() - 0.1, y0=0, x1=x.max() + 0.1, y1=0,
                  line=dict(color='black', width=1))

    fig.add_shape(type='rect',
                  x0=x.min() - 0.1, y0=-0.02, x1=x.max() + 0.1, y1=0.02,  # Von -1.5 bis 1.5 entlang der x-Achse
                  line=dict(color='grey', width=0),
                  fillcolor='grey', opacity=0.1)
    
    # Graue Bereiche für die Y-Achse
    fig.add_shape(type='rect',
                  x0=-0.02, y0=y.min() - 0.1, x1=0.02, y1=y.max() + 0.1,  # Von -1.5 bis 1.5 entlang der y-Achse
                  line=dict(color='grey', width=0),
                  fillcolor='grey', opacity=0.1)
    # Layout anpassen
    fig.update_layout(
        title=f'{x_label} vs {y_label}',
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=800,  # Höhe des Plots erhöhen
        showlegend=True
    )

    return fig

def plot_all_Baxter(Sn_delta, plot_selection):
    # Aufruf der Plot-Funktion mit den entsprechenden Argumenten
    if plot_selection == "d124Sn/120Sn vs d116Sn/120Sn":
        return plot_Sn_Standards(Sn_delta['d116Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 
                                  '\u03B4$^{116}$Sn/$^{120}$Sn', '\u03B4$^{124}$Sn/$^{120}$Sn', 
                                  Sn_delta['2SD'], Sn_delta['2SD.5'], Sn_delta['Lab Nummer'])

    elif plot_selection == "d122Sn/120Sn vs d116Sn/120Sn":
        return plot_Sn_Standards(Sn_delta['d116Sn/120Sn'], Sn_delta['d122Sn/120Sn'], 
                                  '\u03B4$^{116}$Sn/$^{120}$Sn', '\u03B4$^{122}$Sn/$^{120}$Sn', 
                                  Sn_delta['2SD'], Sn_delta['2SD.6'], Sn_delta['Lab Nummer'])

    elif plot_selection == "d124Sn/120Sn vs d118Sn/120Sn":
        return plot_Sn_Standards(Sn_delta['d118Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 
                                  '\u03B4$^{118}$Sn/$^{120}$Sn', '\u03B4$^{124}$Sn/$^{120}$Sn', 
                                  Sn_delta['2SD.2'], Sn_delta['2SD.5'], Sn_delta['Lab Nummer'])

    elif plot_selection == "d124Sn/120Sn vs d122Sn/120Sn":
        return plot_Sn_Standards(Sn_delta['d122Sn/120Sn'], Sn_delta['d124Sn/120Sn'], 
                                  '\u03B4$^{122}$Sn/$^{120}$Sn', '\u03B4$^{124}$Sn/$^{120}$Sn', 
                                  Sn_delta['2SD.2'], Sn_delta['2SD.5'], Sn_delta['Lab Nummer'])

    elif plot_selection == "d118Sn/120Sn und d117Sn/119Sn vs d116Sn/120Sn":
        fig = go.Figure()
        x = Sn_delta['d116Sn/120Sn']
        y1 = Sn_delta['d118Sn/120Sn']
        y2 = Sn_delta['d117Sn/119Sn']
        
        # Scatter-Plot für beide Werte
        fig.add_trace(go.Scatter(
            x=x,
            y=y1,
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=Sn_delta['Lab Nummer'],
            textposition='top center',
            name='\u03B4$^{118}$Sn/$^{120}$Sn'
        ))

        fig.add_trace(go.Scatter(
            x=x,
            y=y2,
            mode='markers+text',
            marker=dict(size=10, color='blue'),
            text=Sn_delta['Lab Nummer'],
            textposition='top center',
            name='\u03B4$^{117}$Sn/$^{119}$Sn'
        ))
        # Fehlerbalken für y1 hinzufügen
        fig.add_trace(go.Scatter(
            x=x,
            y=y1,
            mode='lines',
            line=dict(color='black', width=0),
            error_x=dict(type='data', array=Sn_delta['2SD'], visible=True, color='black'),  # Fehlerbalken für X-Achse
            error_y=dict(type='data', array=Sn_delta['2SD.2'], visible=True, color='black'),  # Fehlerbalken für Y-Achse
            name='Fehlerbalken d118Sn/120Sn'
        ))

        # Fehlerbalken für y2 hinzufügen
        fig.add_trace(go.Scatter(
            x=x,
            y=y2,
            mode='lines',
            line=dict(color='black', width=0),
            error_x=dict(type='data', array=Sn_delta['2SD'], visible=True, color='black'),  # Fehlerbalken für X-Achse
            error_y=dict(type='data', array=Sn_delta['2SD.8'], visible=True, color='black'),  # Fehlerbalken für Y-Achse
            name='Fehlerbalken d117Sn/119Sn'
        ))
        
        
        # Trendlinien berechnen
        reg1 = np.polyfit(x, y1, 1)
        reg2 = np.polyfit(x, y2, 1)
        trend_x = np.linspace(x.min() - 0.1, x.max() + 0.1, 500)
        trend_y1 = np.polyval(reg1, trend_x)
        trend_y2 = np.polyval(reg2, trend_x)

        # Trendlinien hinzufügen
        fig.add_trace(go.Scatter(
            x=trend_x,
            y=trend_y1,
            mode='lines',
            line=dict(color='black', dash='dash'),
            name='Trend d118Sn/120Sn'
        ))

        fig.add_trace(go.Scatter(
            x=trend_x,
            y=trend_y2,
            mode='lines',
            line=dict(color='green', dash='dash'),
            name='Trend d117Sn/119Sn'
        ))

        # Achsen schneiden
        fig.add_shape(type='line',
                  x0=0, y0=y1.min() - 0.1, x1=0, y1=y1.max() + 0.1,
                  line=dict(color='black', width=1))
        fig.add_shape(type='line',
                  x0=x.min() - 0.1, y0=0, x1=x.max() + 0.1, y1=0,
                  line=dict(color='black', width=1))
        
        
        fig.add_shape(type='rect',
                  x0=x.min() - 0.1, y0=-0.02, x1=x.max() + 0.1, y1=0.02,  # Von -1.5 bis 1.5 entlang der x-Achse
                  line=dict(color='grey', width=0),
                  fillcolor='grey', opacity=0.1)
    
    # Graue Bereiche für die Y-Achse
        fig.add_shape(type='rect',
                  x0=-0.02, y0=y1.min() - 0.1, x1=0.02, y1=y1.max() + 0.1,  # Von -1.5 bis 1.5 entlang der y-Achse
                  line=dict(color='grey', width=0),
                  fillcolor='grey', opacity=0.1)

        # Layout anpassen
        fig.update_layout(
            title='d118Sn/120Sn und d117Sn/119Sn vs d116Sn/120Sn',
            xaxis_title='\u03B4$^{116}$Sn/$^{120}$Sn',
            yaxis_title='\u03B4$^{118}$Sn/$^{120}$Sn / \u03B4$^{117}$Sn/$^{119}$Sn',
            height=800,  # Höhe des Plots erhöhen
            showlegend=True
        )

        return fig
