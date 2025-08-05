import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
import io
import base64

def compute_tnr(
    audio: np.ndarray,
    fs: float,
    nfft: int = 65536,
    cb_width_hz: float = None,
    peak_prominence_db: float = 6.0,
    min_freq: float = 20.0,
    max_freq: float = None,
    generate_plot: bool = False
):
    """
    Compute Tone-to-Noise Ratio (TNR) for each tonal peak in `audio`.
    
    Parameters
    ----------
    audio : 1-D numpy array
        Time-domain audio signal.
    fs : float
        Sampling rate in Hz.
    nfft : int, optional
        Number of FFT points for PSD estimation (default: 65536).
        Will be automatically adjusted if larger than audio length.
    cb_width_hz : float, optional
        Half-width of critical band around each peak (Hz). 
        If None, uses 1/3-octave band: cb_width_hz = f0*(2^(1/6) - 2^(−1/6)).
    peak_prominence_db : float, optional
        Minimum peak prominence to detect tonal components (dB).
    min_freq : float, optional
        Minimum frequency to consider for peak detection (Hz).
    max_freq : float, optional
        Maximum frequency to consider for peak detection (Hz).
        If None, uses Nyquist frequency (fs/2).
    generate_plot : bool, optional
        If True, generates and returns a plot of the PSD with peaks and critical bands.
    
    Returns
    -------
    peaks_info : list of dict
        Each dict contains:
        - 'f0' : peak frequency (Hz)
        - 'P_tone' : tone power (linear)
        - 'P_noise' : noise power in critical band (linear)
        - 'TNR_dB'  : 10*log10(P_tone/P_noise)
        - 'peak_height_db' : peak height in dB relative to nearby minimum
    plot_data : dict, optional
        Only returned if generate_plot=True. Contains:
        - 'image_base64': Base64 encoded PNG of the plot
        - 'freqs': Frequency array (for custom plotting)
        - 'psd_db': PSD in dB (for custom plotting)
    """
    # Ensure audio is 1D
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # Use first channel if multi-channel
    
    # Remove DC component
    audio = audio - np.mean(audio)
    
    # Apply window to reduce spectral leakage
    audio = audio * signal.windows.hann(len(audio))
    
    # Ensure nfft doesn't exceed audio length
    nperseg = min(nfft, len(audio))
    
    # Ensure noverlap is less than nperseg
    noverlap = int(nperseg * 0.75)  # 75% overlap for better frequency resolution
    
    # 1. PSD estimate via Welch method with improved parameters
    freqs, psd = signal.welch(audio, fs=fs, window='hann', 
                              nperseg=nperseg, noverlap=noverlap,
                              detrend='constant', scaling='density')
    
    # Convert to dB for better peak detection
    with np.errstate(divide='ignore'):
        psd_db = 10 * np.log10(psd)
    
    # Apply frequency limits
    if max_freq is None:
        max_freq = fs / 2  # Nyquist frequency
    
    # Mask for frequency range of interest
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs_masked = freqs[freq_mask]
    psd_masked = psd[freq_mask]
    psd_db_masked = psd_db[freq_mask]
    
    # 2. Detect peaks in PSD (in dB units for better prominence assessment)
    prom_db = peak_prominence_db
    # Find peaks in the masked frequency range
    peaks, props = signal.find_peaks(psd_db_masked, prominence=prom_db, distance=5)
    
    # Prepare for plotting if requested
    if generate_plot:
        plt.figure(figsize=(12, 6))
        plt.semilogx(freqs, psd_db, linewidth=1, alpha=0.7, color='#1f77b4')
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.title('Power Spectral Density with Detected Tonal Peaks')
        plt.xlim(min_freq, max_freq)
        
        # Add a reasonable y-axis limit
        y_min = max(np.min(psd_db_masked), np.median(psd_db_masked) - 40)
        y_max = min(np.max(psd_db_masked), np.median(psd_db_masked) + 40)
        plt.ylim(y_min, y_max)
    
    peaks_info = []
    
    # Prepare colors for plotting multiple peaks
    if generate_plot and len(peaks) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(peaks))))
    
    for i, idx in enumerate(peaks):
        # Convert index from masked array to original array
        original_idx = np.where(freq_mask)[0][idx]
        f0 = freqs[original_idx]
        
        # 3. Determine half-power bandwidth (–3 dB points)
        peak_power_db = psd_db[original_idx]
        half_power_db = peak_power_db - 3
        
        # Find left boundary
        left_idx = original_idx
        while left_idx > 0 and psd_db[left_idx] > half_power_db:
            left_idx -= 1
            
        # Find right boundary
        right_idx = original_idx
        while right_idx < len(psd_db) - 1 and psd_db[right_idx] > half_power_db:
            right_idx += 1
            
        # Tone power: integrate PSD over half-power bandwidth
        tone_indices = slice(left_idx, right_idx + 1)
        P_tone = np.trapz(psd[tone_indices], freqs[tone_indices])
        
        # 4. Critical-band limits
        if cb_width_hz is None:
            # 1/3-octave band: ±1/6 octave
            cb_width_hz = f0 * (2**(1/6) - 2**(-1/6))
        
        f_low = max(min_freq, f0 - cb_width_hz)
        f_high = min(max_freq, f0 + cb_width_hz)
        
        # Indices for critical band
        cb_mask = (freqs >= f_low) & (freqs <= f_high)
        
        # Total power in critical band
        P_cb = np.trapz(psd[cb_mask], freqs[cb_mask])
        
        # Noise power: remaining power in critical band after subtracting tone power
        P_noise = max(P_cb - P_tone, 1e-10)  # Avoid division by zero
        
        # 5. TNR in dB
        TNR_dB = 10 * np.log10(P_tone / P_noise)
        
        # Calculate peak height in dB relative to nearby minimum
        # This helps evaluate the significance of the peak
        nearby_range = int(len(freqs) * 0.01)  # 1% of spectrum width
        nearby_start = max(0, original_idx - nearby_range)
        nearby_end = min(len(psd_db), original_idx + nearby_range)
        nearby_min_db = np.min(psd_db[nearby_start:nearby_end])
        peak_height_db = peak_power_db - nearby_min_db
        
        peaks_info.append({
            'f0': f0,
            'P_tone': P_tone,
            'P_noise': P_noise,
            'TNR_dB': TNR_dB,
            'peak_height_db': peak_height_db,
            'bandwidth_hz': freqs[right_idx] - freqs[left_idx],
            'critical_band_width_hz': 2 * cb_width_hz
        })
        
        # Add peak visualization if requested
        if generate_plot:
            color = colors[i % len(colors)]
            plt.scatter(f0, peak_power_db, marker='x', s=100, color=color)
            plt.axvspan(f_low, f_high, alpha=0.2, color=color)
            plt.axvspan(freqs[left_idx], freqs[right_idx], alpha=0.3, color=color)
            plt.text(f0, peak_power_db, f"{f0:.1f} Hz\nTNR: {TNR_dB:.1f} dB", 
                     fontsize=8, ha='center', va='bottom')
    
    # Finalize and return plot if requested
    if generate_plot:
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Convert to base64 for embedding in web
        plot_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # Return both TNR results and plot data
        return peaks_info, {
            'image_base64': plot_base64,
            'freqs': freqs.tolist(),
            'psd_db': psd_db.tolist()
        }
    
    return peaks_info

def generate_advanced_visualizations(
    audio: np.ndarray,
    fs: float,
    nfft: int = 65536,
    peaks_info: List[Dict] = None,
    min_freq: float = 20.0,
    max_freq: float = None
) -> Dict[str, str]:
    """
    Generate a set of advanced visualizations for TNR analysis.
    
    Parameters
    ----------
    audio : 1-D numpy array
        Time-domain audio signal.
    fs : float
        Sampling rate in Hz.
    nfft : int, optional
        Number of FFT points for PSD estimation.
    peaks_info : list of dict, optional
        Peak information from compute_tnr function.
    min_freq : float, optional
        Minimum frequency to display in Hz.
    max_freq : float, optional
        Maximum frequency to display in Hz.
    
    Returns
    -------
    dict
        Dictionary containing base64 encoded PNG images of visualizations.
    """
    if max_freq is None:
        max_freq = fs / 2
    
    # Ensure audio is 1D
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    visualizations = {}
    
    # 1. Waveform visualization
    plt.figure(figsize=(12, 4))
    time_array = np.arange(len(audio)) / fs
    plt.plot(time_array, audio, color='#1f77b4', linewidth=1, alpha=0.8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    
    # Save to base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    visualizations['waveform'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # 2. Spectrogram
    plt.figure(figsize=(12, 6))
    nperseg = min(nfft, len(audio))
    noverlap = int(nperseg * 0.75)
    
    f, t, Sxx = signal.spectrogram(audio, fs, window='hann', 
                                  nperseg=nperseg, 
                                  noverlap=noverlap,
                                  scaling='density')
    
    # Convert to dB and limit frequency range
    with np.errstate(divide='ignore'):
        Sxx_db = 10 * np.log10(Sxx)
    
    freq_mask = (f >= min_freq) & (f <= max_freq)
    f_masked = f[freq_mask]
    Sxx_db_masked = Sxx_db[freq_mask, :]
    
    plt.pcolormesh(t, f_masked, Sxx_db_masked, shading='gouraud', cmap='viridis')
    plt.colorbar(label='PSD (dB/Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.yscale('log')
    plt.ylim(min_freq, max_freq)
    plt.title('Audio Spectrogram')
    
    # Save to base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    visualizations['spectrogram'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    
    # 3. If we have peaks, generate a visualization highlighting each peak
    if peaks_info and len(peaks_info) > 0:
        plt.figure(figsize=(12, 6))
        
        # Compute PSD for the visualization
        freqs, psd = signal.welch(audio, fs=fs, window='hann', 
                                  nperseg=nperseg, noverlap=noverlap,
                                  detrend='constant', scaling='density')
        
        # Convert to dB for better visualization
        with np.errstate(divide='ignore'):
            psd_db = 10 * np.log10(psd)
        
        freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
        freqs_masked = freqs[freq_mask]
        psd_db_masked = psd_db[freq_mask]
        
        # Plot overall PSD curve
        plt.semilogx(freqs_masked, psd_db_masked, linewidth=1.5, color='gray', alpha=0.7, label='PSD')
        
        # Use distinct colors for each peak
        colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(peaks_info))))
        
        # Add vertical lines for each detected peak with their TNR value
        for i, peak in enumerate(peaks_info):
            color = colors[i % len(colors)]
            f0 = peak['f0']
            tnr = peak['TNR_dB']
            
            if min_freq <= f0 <= max_freq:
                # Find closest index to f0 in freqs_masked
                idx = np.abs(freqs_masked - f0).argmin()
                y_pos = psd_db_masked[idx]
                
                plt.axvline(x=f0, color=color, linestyle='--', alpha=0.7,
                           label=f'Peak {i+1}: {f0:.1f} Hz, TNR: {tnr:.1f} dB')
                
                # Highlight the critical band
                cb_width = peak.get('critical_band_width_hz', 0)
                if cb_width > 0:
                    f_low = max(min_freq, f0 - cb_width/2)
                    f_high = min(max_freq, f0 + cb_width/2)
                    plt.axvspan(f_low, f_high, alpha=0.2, color=color)
        
        plt.grid(True, which="both", ls="--", alpha=0.7)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (dB/Hz)')
        plt.title('Tonal Components Analysis')
        plt.legend(loc='upper right', fontsize='small')
        plt.xlim(min_freq, max_freq)
        
        # Set reasonable y-limits
        y_range = np.percentile(psd_db_masked, [5, 95])
        margin = (y_range[1] - y_range[0]) * 0.1
        plt.ylim(y_range[0] - margin, y_range[1] + margin)
        
        # Save to base64
        buf = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        visualizations['tonal_analysis'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        
        # 4. TNR Distribution Bar Chart
        if len(peaks_info) >= 2:
            plt.figure(figsize=(10, 6))
            
            # Sort peaks by frequency
            sorted_peaks = sorted(peaks_info, key=lambda x: x['f0'])
            
            frequencies = [peak['f0'] for peak in sorted_peaks]
            tnr_values = [peak['TNR_dB'] for peak in sorted_peaks]
            
            # Create bar chart
            bars = plt.bar(range(len(frequencies)), tnr_values, color=colors[:len(frequencies)])
            
            # Add peak frequency labels
            plt.xticks(range(len(frequencies)), [f"{f:.0f} Hz" for f in frequencies], rotation=45)
            
            # Add TNR values above bars
            for i, v in enumerate(tnr_values):
                plt.text(i, v + 0.5, f"{v:.1f} dB", ha='center')
            
            plt.xlabel('Peak Frequency')
            plt.ylabel('TNR (dB)')
            plt.title('TNR Distribution Across Detected Peaks')
            plt.grid(axis='y', alpha=0.3)
            
            # Save to base64
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            visualizations['tnr_distribution'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()
    
    return visualizations
