# API Request Examples

## Calculate TNR from Mobile Audio Recording

Below are curl examples for the TNR calculation endpoint with various options.

### Basic TNR Analysis

```bash
curl -X POST http://localhost:5000/api/tnr/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "UkIGPwAA3j8JcL+/8YCKP4bYbz/rZgw/jhv5voB+vL8fh/q/CGenvtLejT+ePQJASTBBP9I0ET9M0FC/IAruvjQeCcDyN4e/q7GVP07Sor+1ZHC//ah/v8I1tb6WnFU/MbWUP28XUL9v41e/JbURwCpOAL/gCDo/0oSJvmreCMDZsEU/fEW+PcK6Cj/aogi/a5Yiv2kc9L4vqAc/qwoIP7GQor4eEcA+JO/RvqKJqj5/whO+h3iTvw==",
    "sampleRate": 44100,
    "channels": 1,
    "nfft": 4096,
    "peakProminenceDb": 6.0
  }'
```

### Advanced TNR Analysis with Visualization

```bash
curl -X POST http://localhost:5000/api/tnr/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "UkIGPwAA3j8JcL+/8YCKP4bYbz/rZgw/jhv5voB+vL8fh/q/CGenvtLejT+ePQJASTBBP9I0ET9M0FC/IAruvjQeCcDyN4e/q7GVP07Sor+1ZHC//ah/v8I1tb6WnFU/MbWUP28XUL9v41e/JbURwCpOAL/gCDo/0oSJvmreCMDZsEU/fEW+PcK6Cj/aogi/a5Yiv2kc9L4vqAc/qwoIP7GQor4eEcA+JO/RvqKJqj5/whO+h3iTvw==",
    "sampleRate": 44100,
    "channels": 1,
    "nfft": 8192,
    "criticalBandWidth": 100,
    "peakProminenceDb": 6.0,
    "minFreq": 50,
    "maxFreq": 5000,
    "generatePlot": true
  }'
```

### Analysis with Custom Frequency Range

This example focuses the analysis on the mid-frequency range (200-2000 Hz):

```bash
curl -X POST http://localhost:5000/api/tnr/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "UkIGPwAA3j8JcL+/8YCKP4bYbz/rZgw/jhv5voB+vL8fh/q/CGenvtLejT+ePQJASTBBP9I0ET9M0FC/IAruvjQeCcDyN4e/q7GVP07Sor+1ZHC//ah/v8I1tb6WnFU/MbWUP28XUL9v41e/JbURwCpOAL/gCDo/0oSJvmreCMDZsEU/fEW+PcK6Cj/aogi/a5Yiv2kc9L4vqAc/qwoIP7GQor4eEcA+JO/RvqKJqj5/whO+h3iTvw==",
    "sampleRate": 44100,
    "minFreq": 200,
    "maxFreq": 2000,
    "peakProminenceDb": 10.0
  }'
```

### Using WAV Files

You can also upload audio files directly:

```bash
curl -X POST http://localhost:5000/api/tnr/calculate \
  -F "audio=@your_audio_file.wav" \
  -F "nfft=16384" \
  -F "peakProminenceDb=8.0" \
  -F "generatePlot=true" \
  -F "minFreq=50" \
  -F "maxFreq=10000"
```

> **Note:** The `nfft` parameter will be automatically adjusted if it's larger than the audio length to ensure proper processing.

### For Testing with Real Mobile Audio:

1. **Record audio on your mobile device**
2. **Convert the audio file to base64:**

   On Linux/Mac:
   ```bash
   base64 -i your_audio_file.wav > audio_base64.txt
   ```
   
   On Windows (PowerShell):
   ```powershell
   [Convert]::ToBase64String([IO.File]::ReadAllBytes("your_audio_file.wav")) > audio_base64.txt
   ```

3. **Use the base64 content in your API request**

### Mobile App Recording Guidelines:

For best TNR analysis results when recording from a mobile app:

1. Use a consistent sampling rate (44100 Hz or 48000 Hz recommended)
2. Convert multi-channel audio to mono before sending, or specify "channels": 2
3. For float32 audio, ensure values are normalized between -1.0 and 1.0
4. For int16 audio, the API will automatically normalize by dividing by 32768.0
5. Longer audio samples (1-5 seconds) provide more accurate spectral analysis

### Example Response:

```json
{
  "status": "success",
  "timestamp": "2023-07-15T14:23:45.123456",
  "results": [
    {
      "f0": 1000.5,
      "P_tone": 0.0045,
      "P_noise": 0.0003,
      "TNR_dB": 11.76,
      "peak_height_db": 15.23,
      "bandwidth_hz": 5.2,
      "critical_band_width_hz": 200.0
    },
    {
      "f0": 2500.25,
      "P_tone": 0.0032,
      "P_noise": 0.00025,
      "TNR_dB": 11.06,
      "peak_height_db": 12.8,
      "bandwidth_hz": 6.8,
      "critical_band_width_hz": 500.0
    }
  ],
  "metadata": {
    "audio_length": 48000,
    "duration_seconds": 1.0,
    "analyzed_samples": 4096,
    "sample_rate": 44100,
    "min_frequency": 50,
    "max_frequency": 5000
  },
  "plot": {
    "image_base64": "data:image/png;base64,iVBORw0KGgo..."
  }
}
```

## Advanced TNR Visualizations

Below are examples for the advanced visualization endpoint.

### Basic Visualization Request

```bash
curl -X POST http://localhost:5000/api/tnr/visualize \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "UkIGPwAA3j8JcL+/8YCKP4bYbz/rZgw/jhv5voB+vL8fh/q/CGenvtLejT+ePQJASTBBP9I0ET9M0FC/IAruvjQeCcDyN4e/q7GVP07Sor+1ZHC//ah/v8I1tb6WnFU/MbWUP28XUL9v41e/JbURwCpOAL/gCDo/0oSJvmreCMDZsEU/fEW+PcK6Cj/aogi/a5Yiv2kc9L4vqAc/qwoIP7GQor4eEcA+JO/RvqKJqj5/whO+h3iTvw==",
    "sampleRate": 44100,
    "channels": 1,
    "nfft": 8192,
    "minFreq": 50,
    "maxFreq": 5000
  }'
```

### Visualization Request with WAV File

```bash
curl -X POST http://localhost:5000/api/tnr/visualize \
  -F "audio=@your_audio_file.wav" \
  -F "nfft=16384" \
  -F "peakProminenceDb=8.0" \
  -F "minFreq=50" \
  -F "maxFreq=10000"
```

### Example Response

```json
{
  "status": "success",
  "timestamp": "2023-07-15T15:30:45.123456",
  "metadata": {
    "audio_length": 48000,
    "duration_seconds": 1.0,
    "analyzed_samples": 8192,
    "sample_rate": 44100,
    "min_frequency": 50,
    "max_frequency": 5000
  },
  "visualizations": {
    "waveform": "data:image/png;base64,iVBOR...",
    "spectrogram": "data:image/png;base64,iVBOR...",
    "tonal_analysis": "data:image/png;base64,iVBOR...",
    "tnr_distribution": "data:image/png;base64,iVBOR..."
  },
  "tnr_results": [
    {
      "f0": 1000.5,
      "P_tone": 0.0045,
      "P_noise": 0.0003,
      "TNR_dB": 11.76,
      "peak_height_db": 15.23,
      "bandwidth_hz": 5.2,
      "critical_band_width_hz": 200.0
    },
    {
      "f0": 2500.25,
      "P_tone": 0.0032,
      "P_noise": 0.00025,
      "TNR_dB": 11.06,
      "peak_height_db": 12.8,
      "bandwidth_hz": 6.8,
      "critical_band_width_hz": 500.0
    }
  ]
}
```

### Visualization Types

The endpoint returns multiple visualization types:

1. **waveform** - Time-domain representation of the audio
2. **spectrogram** - Time-frequency analysis showing how frequency content changes over time
3. **tonal_analysis** - Frequency spectrum with detected peaks and their critical bands highlighted
4. **tnr_distribution** - Bar chart showing TNR values across detected peaks (only when multiple peaks are detected)

These visualizations can be directly embedded in web or mobile applications using the base64-encoded image data.

## Interactive TNR Visualizations

The new interactive visualization endpoint provides dynamic, interactive plots using Plotly.

### Basic Interactive Visualization Request

```bash
curl -X POST http://localhost:5000/api/tnr/interactive \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "UkIGPwAA3j8JcL+/8YCKP4bYbz/rZgw/jhv5voB+vL8fh/q/CGenvtLejT+ePQJASTBBP9I0ET9M0FC/IAruvjQeCcDyN4e/q7GVP07Sor+1ZHC//ah/v8I1tb6WnFU/MbWUP28XUL9v41e/JbURwCpOAL/gCDo/0oSJvmreCMDZsEU/fEW+PcK6Cj/aogi/a5Yiv2kc9L4vqAc/qwoIP7GQor4eEcA+JO/RvqKJqj5/whO+h3iTvw==",
    "sampleRate": 44100,
    "channels": 1,
    "nfft": 8192,
    "minFreq": 50,
    "maxFreq": 5000
  }'
```

### Get HTML Report

To receive a standalone HTML report with interactive visualizations:

```bash
curl -X POST http://localhost:5000/api/tnr/interactive \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "UkIGPwAA3j8JcL+/8YCKP4bYbz/rZgw/jhv5voB+vL8fh/q/CGenvtLejT+ePQJASTBBP9I0ET9M0FC/IAruvjQeCcDyN4e/q7GVP07Sor+1ZHC//ah/v8I1tb6WnFU/MbWUP28XUL9v41e/JbURwCpOAL/gCDo/0oSJvmreCMDZsEU/fEW+PcK6Cj/aogi/a5Yiv2kc9L4vqAc/qwoIP7GQor4eEcA+JO/RvqKJqj5/whO+h3iTvw==",
    "sampleRate": 44100,
    "format": "html"
  }' \
  -o tnr_report.html
```

### Interactive Visualization with WAV File

```bash
curl -X POST http://localhost:5000/api/tnr/interactive \
  -F "audio=@your_audio_file.wav" \
  -F "format=html" \
  -o tnr_report.html
```

### Advanced Interactive Visualizations

You can customize the analysis and visualization parameters:

```bash
curl -X POST http://localhost:5000/api/tnr/interactive \
  -F "audio=@your_audio_file.wav" \
  -F "nfft=32768" \
  -F "peakProminenceDb=8.0" \
  -F "minFreq=100" \
  -F "maxFreq=8000" \
  -F "format=json"
```

### Interactive Visualization Types

The endpoint provides several interactive visualization types:

1. **waveform** - Interactive time-domain representation with zoom and pan capabilities
2. **spectrum** - Interactive frequency spectrum with peak detection and hover information
3. **spectrogram** - Interactive time-frequency analysis with color scale for amplitude
4. **spectrogram_3d** - 3D surface plot of the spectrogram for enhanced visualization
5. **waterfall** - Time-frequency-amplitude 3D visualization showing evolution of spectrum over time
6. **tnr_comparison** - Interactive bar chart comparing TNR values across detected peaks

These interactive visualizations allow users to:
- Zoom in/out on specific regions
- Pan across the visualization
- Hover over elements for detailed information
- Export plots as PNG images
- Toggle visibility of different data series

When using the HTML output format, you'll receive a standalone HTML page with all interactive visualizations that can be viewed in any modern web browser without additional dependencies.
