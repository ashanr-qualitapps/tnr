import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

def compute_tnr(
    audio: np.ndarray,
    fs: float,
    nfft: int = 65536,
    cb_width_hz: float = None,
    peak_prominence_db: float = 6.0
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
    
    Returns
    -------
    peaks_info : list of dict
        Each dict contains:
        - 'f0' : peak frequency (Hz)
        - 'P_tone' : tone power (linear)
        - 'P_noise' : noise power in critical band (linear)
        - 'TNR_dB'  : 10*log10(P_tone/P_noise)
    """
    # Ensure nfft doesn't exceed audio length
    nperseg = min(nfft, len(audio))
    
    # Ensure noverlap is less than nperseg
    noverlap = nperseg // 2
    
    # 1. PSD estimate via Welch
    freqs, psd = signal.welch(audio, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    # 2. Detect peaks in PSD (in linear units; convert prominence to linear)
    prom_linear = 10**(peak_prominence_db/10)
    peaks, props = signal.find_peaks(psd, prominence=prom_linear)
    
    peaks_info = []
    for idx in peaks:
        f0 = freqs[idx]
        
        # 3. Determine half-power bandwidth (–3 dB points)
        half_power = psd[idx] / 2
        # find left boundary
        left_idx = idx
        while left_idx>0 and psd[left_idx] > half_power:
            left_idx -= 1
        # find right boundary
        right_idx = idx
        while right_idx<len(psd)-1 and psd[right_idx] > half_power:
            right_idx += 1
        
        # Tone power: integrate PSD over [left_idx, right_idx]
        P_tone = np.trapz(psd[left_idx:right_idx+1], freqs[left_idx:right_idx+1])
        
        # 4. Critical-band limits
        if cb_width_hz is None:
            # 1/3-octave band: ±1/6 octave
            # Fix the character issue in the original code
            cb_width_hz = f0 * (2**(1/6) - 2**(-1/6))
        f_low = f0 - cb_width_hz
        f_high = f0 + cb_width_hz
        # indices for critical band
        cb_mask = (freqs>=f_low) & (freqs<=f_high)
        P_cb = np.trapz(psd[cb_mask], freqs[cb_mask])
        
        # Noise power: remaining in critical band
        P_noise = max(P_cb - P_tone, 1e-20)
        
        # 5. TNR in dB
        TNR_dB = 10*np.log10(P_tone/P_noise)
        
        peaks_info.append({
            'f0': f0,
            'P_tone': P_tone,
            'P_noise': P_noise,
            'TNR_dB': TNR_dB
        })
    
    # Add example usage to the docstring
    """
    Example usage:
    --------------
    >>> import numpy as np
    >>> from scipy import signal
    >>> from tnr_calculator import compute_tnr
    >>> 
    >>> # Generate a sample signal: 1000 Hz tone + noise
    >>> fs = 44100
    >>> t = np.arange(0, 1, 1/fs)
    >>> signal = np.sin(2*np.pi*1000*t) + 0.5*np.random.randn(len(t))
    >>> 
    >>> # Compute TNR
    >>> results = compute_tnr(signal, fs)
    >>> for r in results:
    >>>     print(f"Tone at {r['f0']:.1f} Hz → TNR {r['TNR_dB']:.1f} dB")
    """
    
    return peaks_info
