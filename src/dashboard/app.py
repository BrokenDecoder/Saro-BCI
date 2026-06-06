"""
Saro BCI: Interactive Neural Dashboard
=======================================
A premium, dark-mode real-time dashboard visualizing human EEG data 
and AI Twin predictions using Mamba SSM.
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import threading
import time

# Adjust paths to import our modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing.physionet_fetcher import fetch_real_eeg_data
from src.ai_twin.eeg_mamba import EEGMambaTwin

# Initialize Dashboard
app = dash.Dash(__name__, title="Saro BCI - Neural Twin Dashboard")
server = app.server

# ---------------------------------------------------------
# Global State for Streaming Simulation
# ---------------------------------------------------------
class StreamingData:
    def __init__(self):
        self.raw_data = None
        self.sfreq = 160
        self.current_idx = 0
        self.window_size = 800  # 5 seconds at 160Hz
        self.ai_twin = None
        
    def load(self):
        print("Loading real EEG dataset...")
        try:
            self.raw_data, self.sfreq = fetch_real_eeg_data(subject=1, run=3, tmin=0.0, tmax=120.0)
            self.window_size = int(self.sfreq * 3) # 3 seconds window
            print("Loading AI Twin...")
            self.ai_twin = EEGMambaTwin(n_channels=self.raw_data.shape[0])
            print("Ready.")
        except Exception as e:
            print(f"Error loading data: {e}")
            # Fallback to noise if network fails
            self.sfreq = 160
            self.window_size = 160 * 3
            self.raw_data = np.random.randn(64, 160 * 120).astype(np.float32)
            self.ai_twin = EEGMambaTwin(n_channels=64)

stream = StreamingData()
# Load in a background thread so the app boots quickly
threading.Thread(target=stream.load).start()


# ---------------------------------------------------------
# App Layout (Premium Dark Mode)
# ---------------------------------------------------------
app.layout = html.Div(style={'backgroundColor': '#0b0f19', 'color': '#ffffff', 'fontFamily': 'Inter, sans-serif', 'minHeight': '100vh', 'padding': '20px'}, children=[
    html.H1("Saro BCI: Neural Twin Diagnostics", style={'textAlign': 'center', 'fontWeight': '300', 'letterSpacing': '2px', 'marginBottom': '10px'}),
    html.P("Real-time Human Brainwave Integration & Mamba SSM Generative Inference", style={'textAlign': 'center', 'color': '#718096', 'marginBottom': '40px'}),
    
    html.Div(style={'display': 'flex', 'flexDirection': 'row', 'gap': '20px', 'justifyContent': 'center'}, children=[
        html.Div(style={'width': '45%', 'backgroundColor': '#111827', 'borderRadius': '12px', 'padding': '20px', 'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.5)'}, children=[
            html.H3("Live Patient EEG (Real Data)", style={'color': '#9ca3af', 'fontWeight': '400'}),
            dcc.Graph(id='live-eeg-graph', config={'displayModeBar': False}),
        ]),
        html.Div(style={'width': '45%', 'backgroundColor': '#111827', 'borderRadius': '12px', 'padding': '20px', 'boxShadow': '0 4px 6px -1px rgba(0, 0, 0, 0.5)'}, children=[
            html.H3("AI Twin Healthy Prediction (Mamba SSM)", style={'color': '#9ca3af', 'fontWeight': '400'}),
            dcc.Graph(id='ai-twin-graph', config={'displayModeBar': False}),
        ]),
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=500, # Update every 500ms
        n_intervals=0
    )
])

# ---------------------------------------------------------
# Callbacks for Real-Time Plotting
# ---------------------------------------------------------
@app.callback(
    [Output('live-eeg-graph', 'figure'),
     Output('ai-twin-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graphs(n):
    if stream.raw_data is None:
        return go.Figure(), go.Figure()
        
    # Simulate streaming by incrementing index
    step = int(stream.sfreq * 0.5) # Advance 0.5s
    stream.current_idx = (stream.current_idx + step) % (stream.raw_data.shape[1] - stream.window_size)
    
    start = stream.current_idx
    end = start + stream.window_size
    window = stream.raw_data[:, start:end]
    
    # Generate AI Twin prediction
    healthy_pred = stream.ai_twin.predict_healthy_state(window)
    
    # We plot only a subset of channels for visual clarity (e.g., 5 channels)
    n_plot = 5
    times = np.linspace(0, stream.window_size / stream.sfreq, stream.window_size)
    
    def create_figure(data_window, color, title):
        fig = go.Figure()
        for i in range(n_plot):
            offset = i * 2.0 # Vertical offset to separate channels
            fig.add_trace(go.Scatter(
                x=times, y=data_window[i, :] + offset,
                mode='lines',
                line=dict(color=color, width=1.5),
                name=f'CH {i+1}'
            ))
        fig.update_layout(
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            font=dict(color='#9ca3af'),
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            showlegend=False
        )
        return fig

    fig_live = create_figure(window, color='#3b82f6', title="Live EEG")
    fig_ai = create_figure(healthy_pred, color='#10b981', title="AI Twin")
    
    return fig_live, fig_ai

if __name__ == '__main__':
    print("Starting Saro BCI Dashboard...")
    app.run(debug=True, port=8050)
