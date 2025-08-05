import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import base64
from scipy import signal
from io import BytesIO

def create_interactive_visualizations(audio, fs, tnr_results, nfft=16384):
    """
    Create interactive visualizations for TNR analysis using Plotly.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio data
    fs : float
        Sample rate in Hz
    tnr_results : list
        List of dictionaries with TNR analysis results
    nfft : int
        FFT size for spectral analysis
    
    Returns
    -------
    dict
        Dictionary containing plotly figure JSONs for different visualizations
    """
    # Sort results by frequency for consistent visualization
    sorted_results = sorted(tnr_results, key=lambda x: x['f0'])
    
    visualizations = {}
    
    # 1. Interactive Waveform visualization
    fig_waveform = go.Figure()
    time = np.arange(len(audio)) / fs
    fig_waveform.add_trace(go.Scatter(
        x=time, 
        y=audio,
        mode='lines',
        name='Waveform'
    ))
    fig_waveform.update_layout(
        title='Audio Waveform',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        hovermode='closest'
    )
    visualizations['waveform'] = fig_waveform.to_json()
    
    # 2. Interactive Spectrum visualization with peaks highlighted
    freqs, psd = signal.welch(audio, fs=fs, window='hann', nperseg=nfft, noverlap=nfft//2)
    
    fig_spectrum = go.Figure()
    fig_spectrum.add_trace(go.Scatter(
        x=freqs,
        y=10 * np.log10(psd + 1e-10),  # Convert to dB
        mode='lines',
        name='PSD',
        line=dict(color='blue')
    ))
    
    # Add markers for detected peaks
    peak_freqs = [result['f0'] for result in sorted_results]
    peak_psd_db = []
    
    # Find PSD values at peak frequencies
    for freq in peak_freqs:
        idx = np.argmin(np.abs(freqs - freq))
        peak_psd_db.append(10 * np.log10(psd[idx] + 1e-10))
    
    # Add peak markers with TNR values in the hover text
    hover_texts = [f"Freq: {r['f0']:.1f} Hz<br>TNR: {r['TNR_dB']:.1f} dB" for r in sorted_results]
    
    fig_spectrum.add_trace(go.Scatter(
        x=peak_freqs,
        y=peak_psd_db,
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            symbol='circle'
        ),
        name='Detected Peaks',
        text=hover_texts,
        hoverinfo='text'
    ))
    
    fig_spectrum.update_layout(
        title='Power Spectral Density with Detected Tones',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power/Frequency (dB/Hz)',
        xaxis_type='log',
        hovermode='closest'
    )
    visualizations['spectrum'] = fig_spectrum.to_json()
    
    # 3. TNR Values Comparison
    if len(sorted_results) > 1:
        fig_tnr = px.bar(
            x=[f"{r['f0']:.1f} Hz" for r in sorted_results],
            y=[r['TNR_dB'] for r in sorted_results],
            labels={'x': 'Frequency', 'y': 'TNR (dB)'},
            title='Tone-to-Noise Ratio Comparison',
            color=[r['TNR_dB'] for r in sorted_results],
            color_continuous_scale='Viridis'
        )
        fig_tnr.update_layout(coloraxis_colorbar=dict(title='TNR (dB)'))
        visualizations['tnr_comparison'] = fig_tnr.to_json()
    
    # 4. Spectrogram visualization
    f, t, Sxx = signal.spectrogram(
        audio, 
        fs=fs, 
        window='hann',
        nperseg=min(1024, len(audio)//8),
        noverlap=min(512, len(audio)//16),
        nfft=min(2048, len(audio))
    )
    
    # Convert to dB for better visualization
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    fig_spectrogram = go.Figure(data=go.Heatmap(
        z=Sxx_db,
        x=t,
        y=f,
        colorscale='Jet',
        hoverongaps=False
    ))
    
    fig_spectrogram.update_layout(
        title='Spectrogram',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        yaxis_type='log'
    )
    visualizations['spectrogram'] = fig_spectrogram.to_json()
    
    # 5. 3D Surface plot of the spectrogram
    fig_3d = go.Figure(data=[go.Surface(z=Sxx_db, x=t, y=f)])
    fig_3d.update_layout(
        title='3D Spectrogram Surface',
        scene=dict(
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            zaxis_title='Power (dB)'
        )
    )
    visualizations['spectrogram_3d'] = fig_3d.to_json()
    
    # 6. Time-Frequency-Amplitude Waterfall
    fig_waterfall = go.Figure()
    
    # Create a waterfall plot by adding multiple line traces
    skip = max(1, len(t) // 20)  # Show at most 20 time slices for clarity
    for i in range(0, len(t), skip):
        fig_waterfall.add_trace(go.Scatter3d(
            x=f,
            y=np.full_like(f, t[i]),
            z=Sxx_db[:, i],
            mode='lines',
            line=dict(color=f'rgb({int(255*i/len(t))},0,{int(255*(1-i/len(t)))})', width=2),
            name=f't={t[i]:.3f}s'
        ))
    
    fig_waterfall.update_layout(
        title='Time-Frequency Waterfall Plot',
        scene=dict(
            xaxis_title='Frequency (Hz)',
            yaxis_title='Time (s)',
            zaxis_title='Power (dB)'
        )
    )
    visualizations['waterfall'] = fig_waterfall.to_json()
    
    return visualizations

def create_html_report(visualizations):
    """
    Create a standalone HTML report with interactive visualizations
    
    Parameters
    ----------
    visualizations : dict
        Dictionary of Plotly figure JSONs
    
    Returns
    -------
    str
        HTML content as string
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TNR Analysis Interactive Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .plot-container { margin-bottom: 30px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; }
            .plot { width: 100%; height: 500px; }
        </style>
    </head>
    <body>
        <h1>TNR Analysis Interactive Report</h1>
    """
    
    for plot_name, plot_json in visualizations.items():
        div_id = f"plot_{plot_name}"
        title = plot_name.replace('_', ' ').title()
        
        html_content += f"""
        <div class="plot-container">
            <h2>{title}</h2>
            <div id="{div_id}" class="plot"></div>
            <script>
                Plotly.newPlot("{div_id}", {plot_json});
            </script>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content
