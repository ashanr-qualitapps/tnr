# API Request Examples

## Calculate TNR from Mobile Audio Recording

Below is a curl example for the TNR calculation endpoint. This example uses a very small sample of audio data for demonstration purposes.

### Curl Request

```bash
curl -X POST http://localhost:5000/api/tnr/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "audio": "UkIGPwAA3j8JcL+/8YCKP4bYbz/rZgw/jhv5voB+vL8fh/q/CGenvtLejT+ePQJASTBBP9I0ET9M0FC/IAruvjQeCcDyN4e/q7GVP07Sor+1ZHC//ah/v8I1tb6WnFU/MbWUP28XUL9v41e/JbURwCpOAL/gCDo/0oSJvmreCMDZsEU/fEW+PcK6Cj/aogi/a5Yiv2kc9L4vqAc/qwoIP7GQor4eEcA+JO/RvqKJqj5/whO+h3iTvw==",
    "sampleRate": 44100,
    "channels": 1,
    "nfft": 4096,
    "criticalBandWidth": 100,
    "peakProminenceDb": 6.0
  }'
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
      "TNR_dB": 11.76
    },
    {
      "f0": 2500.25,
      "P_tone": 0.0032,
      "P_noise": 0.00025,
      "TNR_dB": 11.06
    }
  ],
  "metadata": {
    "audio_length": 48000,
    "duration_seconds": 1.0,
    "analyzed_samples": 4096
  }
}
```
