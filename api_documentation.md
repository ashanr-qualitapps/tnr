# TNR Calculator API Documentation

This document provides information about the Tone-to-Noise Ratio (TNR) Calculator API endpoints and their usage.

## Endpoints

### compute_tnr

Calculates the Tone-to-Noise Ratio (TNR) for each tonal peak in an audio signal.

#### Function Signature

```python
compute_tnr(
    audio: np.ndarray,
    fs: float,
    nfft: int = 65536,
    cb_width_hz: float = None,
    peak_prominence_db: float = 6.0
) -> List[Dict]
```

#### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `audio` | `np.ndarray` | Time-domain audio signal (1-D numpy array) | Required |
| `fs` | `float` | Sampling rate in Hz | Required |
| `nfft` | `int` | Number of FFT points for PSD estimation | 65536 |
| `cb_width_hz` | `float` | Half-width of critical band around each peak (Hz). If None, uses 1/3-octave band: cb_width_hz = f0*(2^(1/6) - 2^(−1/6)) | None |
| `peak_prominence_db` | `float` | Minimum peak prominence to detect tonal components (dB) | 6.0 |

#### Returns

A list of dictionaries, where each dictionary contains information about a detected tonal peak:

| Key | Type | Description |
|-----|------|-------------|
| `f0` | `float` | Peak frequency (Hz) |
| `P_tone` | `float` | Tone power (linear) |
| `P_noise` | `float` | Noise power in critical band (linear) |
| `TNR_dB` | `float` | Tone-to-Noise Ratio in dB: 10*log10(P_tone/P_noise) |

#### Example Usage

```python
import numpy as np
from scipy import signal
from tnr_calculator import compute_tnr

# Load audio file
fs, audio = signal.wavfile.read('example.wav')
audio = audio.astype(float)
audio /= np.max(np.abs(audio))

# Compute TNR
results = compute_tnr(audio, fs)

# Display detected tones and their TNRs
for r in results:
    print(f"Tone at {r['f0']:.1f} Hz → TNR {r['TNR_dB']:.1f} dB")
```

#### Advanced Usage

```python
# With custom parameters
results = compute_tnr(
    audio, 
    fs=44100, 
    nfft=131072,  # Higher resolution FFT
    cb_width_hz=100,  # Custom critical band width
    peak_prominence_db=10.0  # More strict peak detection
)
```

## Tips & Variations

- **PSD Method**: Use `signal.periodogram` for stationary signals.
- **Windowing**: Increase `nperseg` for finer frequency resolution.
- **Critical Band**: For psychoacoustic accuracy, replace 1/3-octave with ERB-based bandwidth.
- **Peak Criteria**: Adjust `peak_prominence_db` to fine-tune detection.
- **Smoothing**: Apply median or Gaussian smoothing to the PSD before peak detection to reduce spurious peaks.
