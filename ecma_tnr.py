import numpy as np
from scipy import signal
from scipy.signal.windows import hann

def get_prominence_threshold(f0):
    """
    Return frequency-dependent prominence threshold (dB) per ANSI S1.13-2005.
    """
    if f0 >= 1000:
        return 9.0
    elif f0 >= 500:
        return 12.0
    elif f0 >= 400:
        return 8.0
    elif f0 >= 160:
        return 8.0
    elif f0 >= 125:
        return 18.0
    elif f0 >= 100:
        return 19.0
    else:
        return 15.0

def zwicker_critical_bandwidth(f0):
    """
    Compute Zwicker critical bandwidth (Hz).
    """
    return 25 + 75 * (1 + 1.4 * (f0 / 1000.0)**2)**0.69

def calculate_prominence_ratio(tone_power, adjacent_powers):
    """
    Calculate Prominence Ratio (PR) in dB.
    """
    avg_adj = np.mean(adjacent_powers)
    return 10 * np.log10(tone_power / max(avg_adj, 1e-10))

def calculate_tnr(tone_power, noise_power):
    """
    Calculate Tone-to-Noise Ratio (TNR) in dB.
    """
    return 10 * np.log10(tone_power / max(noise_power, 1e-10))

def apply_loudness_weighting(freqs, psd):
    """
    Apply simplified ISO 532B loudness weighting to PSD.
    """
    f2 = freqs**2
    ra = (f2 + 56.8e6)**2 * f2
    rb = (f2 + 6.3e6)**2 * (f2 + 0.38e9)
    aw = 2.0 + 20 * np.log10(12194**2 * f2 / np.sqrt(ra * rb))
    weights = 10**(aw / 20.0)
    return psd * weights

def adjust_for_background(tnr_db, bg_level_dba):
    """
    Adjust TNR threshold for higher background levels.
    """
    if bg_level_dba > 50:
        return tnr_db + 0.1 * (bg_level_dba - 40)
    return tnr_db

def predict_annoyance_probability(tnr_db, pr_db, loudness_sone, f0):
    """
    Predict complaint probability via logistic model (example coefficients).
    """
    b0, b1, b2 = -20.354, 0.294, 0.040
    aud_db = max(tnr_db, pr_db) - get_prominence_threshold(f0)
    logit = b0 + b1 * loudness_sone + b2 * aud_db
    return 1.0 / (1.0 + np.exp(-logit))

def optimized_spectral_analysis(audio, fs):
    """
    Compute PSD with high resolution and robust averaging.
    """
    nfft = min(131072, len(audio))
    freqs, psd = signal.welch(
        audio,
        fs=fs,
        nperseg=nfft,
        noverlap=nfft//2,
        window=hann(nfft, sym=False),
        scaling='density',
        average='median'
    )
    return freqs, psd

def tnr_ecma_st_improved(audio, fs, bg_level_dba=None):
    """
    Enhanced TNR/PR detection aligned with human annoyance research.

    Returns:
      tnr_vals     : array of TNR dB for each tone (clipped to 1–10 dB)
      pr_vals      : array of PR dB for each tone
      probs        : array of annoyance probabilities [0–1]
      freqs_tones  : array of detected tone frequencies (Hz)
    """
    freqs, psd = optimized_spectral_analysis(audio, fs)
    psd_db = 10 * np.log10(psd + 1e-12)
    psd_weighted = apply_loudness_weighting(freqs, psd)

    peaks, props = signal.find_peaks(psd_db, prominence=3.0, distance=5)
    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    pr_vals, tnr_vals, probs, freqs_tones = [], [], [], freqs[peaks]
    freq_res = freqs[1] - freqs[0]

    for idx, f0 in zip(peaks, freqs[peaks]):
        bw = zwicker_critical_bandwidth(f0)
        f_low, f_high = f0 - bw/2, f0 + bw/2
        band_idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]

        p_tone = psd[idx]
        p_band = np.sum(psd[band_idx]) * freq_res
        p_noise = max(p_band - p_tone, 1e-12)

        adj_lo = np.where((freqs >= f_low - bw) & (freqs < f_low))[0]
        adj_hi = np.where((freqs > f_high) & (freqs <= f_high + bw))[0]
        adj_powers = (np.concatenate((psd[adj_lo], psd[adj_hi]))
                      if adj_lo.size + adj_hi.size > 0
                      else np.array([p_noise]))

        pr_db = calculate_prominence_ratio(p_tone, adj_powers)
        raw_tnr = calculate_tnr(p_tone, p_noise)
        # Clip TNR to [1, 10] dB
        tnr_db = np.clip(raw_tnr, 1.0, 10.0)

        thr = get_prominence_threshold(f0)
        pr_vals.append(pr_db if pr_db >= thr else 0.0)
        tnr_vals.append(tnr_db)

        if bg_level_dba is not None:
            tnr_vals[-1] = adjust_for_background(tnr_vals[-1], bg_level_dba)

        loudness_sone = np.sum(psd_weighted[band_idx]) * freq_res
        probs.append(predict_annoyance_probability(
            tnr_vals[-1], pr_vals[-1], loudness_sone, f0))

    return np.array(tnr_vals), np.array(pr_vals), np.array(probs), freqs_tones

def tnr_ecma_418_2(audio: np.ndarray, fs: float, nfft: int = 16384):
    """
    Calculate Tone-to-Noise Ratio (TNR) according to ECMA 418-2 standard.
    
    Parameters
    ----------
    audio : np.ndarray
        Audio signal data (time domain)
    fs : float
        Sampling rate in Hz
    nfft : int, optional
        FFT size for spectral analysis
        
    Returns
    -------
    tuple
        (tnr_values, prominences, tonal_frequencies, critical_band_ratios)
        - tnr_values: TNR values for each detected tone
        - prominences: Prominence of each detected tone
        - tonal_frequencies: Frequencies of detected tones
        - critical_band_ratios: Signal-to-noise ratio within critical bands
    """
    # Ensure audio is properly preprocessed
    audio = audio - np.mean(audio)  # Remove DC offset
    
    # Apply window to reduce spectral leakage
    windowed_audio = audio * signal.windows.hann(len(audio))
    
    # Compute FFT with high resolution
    nperseg = min(nfft, len(audio))
    noverlap = int(nperseg * 0.75)  # 75% overlap per ECMA 418-2
    
    # Compute power spectrum according to ECMA 418-2 requirements
    freqs, psd = signal.welch(
        windowed_audio, 
        fs=fs, 
        window='hann', 
        nperseg=nperseg, 
        noverlap=noverlap,
        detrend=False,  # ECMA 418-2 doesn't specify detrending
        scaling='density'
    )
    
    # Convert to dB for analysis
    psd_db = 10 * np.log10(psd + 1e-10)
    
    # ECMA 418-2 specific parameters
    min_prominence = 6.0  # dB prominence threshold 
    min_freq = 89.1  # Hz - ECMA 418-2 defines specific frequency bands
    max_freq = 11220  # Hz - Upper limit per standard
    
    # Define frequency bands according to ECMA 418-2
    band_edges = [89.1, 112, 141, 178, 224, 282, 355, 447, 562,
                  708, 891, 1122, 1413, 1778, 2239, 2818, 3548, 
                  4467, 5623, 7079, 8913, 11220]
    
    # Apply frequency range limits
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs_masked = freqs[mask]
    psd_db_masked = psd_db[mask]
    
    # Find peaks according to ECMA 418-2 criteria
    peaks, props = signal.find_peaks(psd_db_masked, prominence=min_prominence, distance=3)
    
    # If no peaks were found, return zeros
    if len(peaks) == 0:
        return 0, 0, [], 0
    
    # Extract peak information
    peak_freqs = freqs_masked[peaks]
    peak_psd = psd_db_masked[peaks]
    prominences = props['prominences']
    
    tnr_values = []
    critical_band_ratios = []
    
    for i, idx in enumerate(peaks):
        f0 = freqs_masked[idx]
        
        # Find appropriate band for this frequency
        band_idx = np.searchsorted(band_edges, f0) - 1
        if band_idx < 0:
            band_idx = 0
        if band_idx >= len(band_edges) - 1:
            band_idx = len(band_edges) - 2
            
        # Get band edges for the critical band
        f_low = band_edges[band_idx]
        f_high = band_edges[band_idx + 1]
        
        # Calculate tone power and noise power in critical band
        # According to ECMA 418-2 methodology
        
        # Find indices for tone power calculation (within 3dB of peak)
        peak_power_db = psd_db_masked[idx]
        tone_threshold = peak_power_db - 3
        
        # Find left boundary
        left_idx = idx
        while left_idx > 0 and psd_db_masked[left_idx] > tone_threshold:
            left_idx -= 1
            
        # Find right boundary
        right_idx = idx
        while right_idx < len(psd_db_masked) - 1 and psd_db_masked[right_idx] > tone_threshold:
            right_idx += 1
            
        # Calculate tone power by integrating over the tone bandwidth
        tone_indices = slice(left_idx, right_idx + 1)
        tone_freqs = freqs_masked[tone_indices]
        tone_psd = psd[mask][tone_indices]  # Use linear units for integration
        P_tone = np.trapz(tone_psd, tone_freqs)
        
        # Calculate critical band limits based on ECMA 418-2
        cb_mask = (freqs_masked >= f_low) & (freqs_masked <= f_high)
        cb_freqs = freqs_masked[cb_mask]
        cb_psd = psd[mask][cb_mask]  # Use linear units
        
        # Total power in the critical band
        P_cb = np.trapz(cb_psd, cb_freqs)
        
        # Noise power is the difference between total and tone power
        P_noise = max(P_cb - P_tone, 1e-10)
        
        # Calculate TNR according to ECMA 418-2
        tnr_db = 10 * np.log10(P_tone / P_noise)
        tnr_values.append(tnr_db)
        
        # Calculate critical band ratio (signal-to-noise in critical band)
        # This is an additional metric required by ECMA 418-2
        cb_ratio = 10 * np.log10(P_cb / P_noise)
        critical_band_ratios.append(cb_ratio)
    
    return np.array(tnr_values), prominences, peak_freqs, np.array(critical_band_ratios)

# Alias for backward compatibility
tnr_ecma_st = tnr_ecma_st_improved
