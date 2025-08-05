from flask import Flask, jsonify, request, render_template_string, make_response
from datetime import datetime
import numpy as np
import base64
from tnr_calculator import compute_tnr, generate_advanced_visualizations
import io
from scipy.io import wavfile  # Added for WAV file processing
# Import the new interactive visualizer module
from interactive_visualizer import create_interactive_visualizations, create_html_report
# Import the ECMA-74 TNR calculation function
from ecma_tnr import tnr_ecma_st, tnr_ecma_418_2
import os

app = Flask(__name__)

# In-memory storage for demo purposes
users = [
    {"id": 1, "name": "John Doe", "email": "john@example.com"},
    {"id": 2, "name": "Jane Smith", "email": "jane@example.com"}
]

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "TNR API"
    })


@app.route('/api/tnr/calculate_ecma', methods=['POST'])
def calculate_tnr_ecma():
    """
    Endpoint to calculate Tonal Noise Ratio (TNR) using the ECMA-74 standard.
    Accepts audio data as WAV file or base64-encoded JSON.
    """
    # Check if request has the file part
    if 'audio' in request.files:
        audio_file = request.files['audio']
        
        try:
            # Read WAV file directly from the file stream
            file_stream = io.BytesIO(audio_file.read())
            fs, audio_data = wavfile.read(file_stream)
            
            # Convert to float32 if needed
            if audio_data.dtype == np.int16:
                audio = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio = (audio_data.astype(np.float32) - 128) / 128.0
            else:  # Already float
                audio = audio_data
            
            # Handle stereo audio - use first channel only
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = audio[:, 0]
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return jsonify({
                "error": f"Error processing WAV file: {str(e)}",
                "details": traceback_str
            }), 500
    
    # Handle JSON request (existing code path)
    elif request.is_json:
        data = request.get_json()
        
        if not data or 'audio' not in data or 'sampleRate' not in data:
            return jsonify({"error": "Audio data and sample rate are required"}), 400
        
        try:
            # Decode the base64 audio data
            audio_bytes = base64.b64decode(data['audio'])
            
            # Try to interpret as float32 first (default expected format)
            try:
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
            except:
                # Fallback: try to interpret as int16 (common mobile format)
                try:
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    # Normalize int16 audio to float range [-1.0, 1.0]
                    audio = audio / 32768.0
                except:
                    return jsonify({"error": "Unsupported audio format. Please provide float32 or int16 audio data."}), 400
            
            # Handle stereo audio (mobile phones often record stereo)
            if 'channels' in data and data['channels'] == 2:
                # Convert from interleaved to separate channels
                # Use first channel only for analysis
                audio = audio[::2]
            
            # Ensure audio isn't empty
            if len(audio) == 0:
                return jsonify({"error": "Empty audio data received"}), 400
                
            # Get sample rate
            fs = float(data['sampleRate'])
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return jsonify({
                "error": str(e),
                "details": traceback_str
            }), 500
    
    else:
        return jsonify({"error": "Please provide audio as WAV file in form-data or as base64 in JSON"}), 400
    
    # Ensure audio isn't empty
    if len(audio) == 0:
        return jsonify({"error": "Empty audio data received"}), 400
    
    # Calculate TNR using ECMA-74 standard
    try:
        # Call the ECMA-74 TNR calculation function
        t_tnr, tnr, prom, tones_freqs = tnr_ecma_st(audio, fs)
        
        # Check if any tonal noise was detected
        has_tonal_noise = not (isinstance(t_tnr, (int, float)) and t_tnr == 0)
        
        # Calculate mean TNR if it's a list/array, otherwise use the value directly
        # Also clip TNR values to a maximum of 10.0
        if has_tonal_noise:
            if isinstance(t_tnr, (list, np.ndarray)):
                # Clip each value to max 10.0 before calculating the mean
                clipped_tnr = [min(float(val), 10.0) for val in t_tnr]
                mean_tnr = float(np.mean(clipped_tnr))
            else:
                # Clip single value to max 10.0
                mean_tnr = min(float(t_tnr), 10.0)
        else:
            mean_tnr = 0.0
            
        # Ensure tonal frequencies are JSON serializable
        tonal_freqs = []
        if has_tonal_noise and isinstance(tones_freqs, (list, np.ndarray)) and len(tones_freqs) > 0:
            tonal_freqs = [round(float(freq), 1) for freq in tones_freqs]
            
        # Prepare the TNR results
        tnr_results = {
            'tnr_value': round(mean_tnr, 1),
            'has_tonal_noise': has_tonal_noise,
            'tonal_frequencies': tonal_freqs
        }
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return jsonify({
            "error": f"Error calculating ECMA-74 TNR: {str(e)}",
            "details": traceback_str
        }), 500
    
    # Prepare response data
    response_data = {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        "results": tnr_results,
        "metadata": {
            "audio_length": len(audio),
            "duration_seconds": len(audio) / fs,
            "sample_rate": fs
        }
    }
    
    return jsonify(response_data)

@app.route('/api/tnr/calculate_ecma_418_2', methods=['POST'])
def calculate_tnr_ecma_418_2():
    """
    Endpoint to calculate Tonal Noise Ratio (TNR) using the ECMA 418-2 standard.
    Accepts audio data as WAV file or base64-encoded JSON.
    """
    # Check if request has the file part
    if 'audio' in request.files:
        audio_file = request.files['audio']
        
        try:
            # Read WAV file directly from the file stream
            file_stream = io.BytesIO(audio_file.read())
            fs, audio_data = wavfile.read(file_stream)
            
            # Convert to float32 if needed
            if audio_data.dtype == np.int16:
                audio = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio = (audio_data.astype(np.float32) - 128) / 128.0
            else:  # Already float
                audio = audio_data
            
            # Handle stereo audio - use first channel only
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = audio[:, 0]
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return jsonify({
                "error": f"Error processing WAV file: {str(e)}",
                "details": traceback_str
            }), 500
    
    # Handle JSON request (existing code path)
    elif request.is_json:
        data = request.get_json()
        
        if not data or 'audio' not in data or 'sampleRate' not in data:
            return jsonify({"error": "Audio data and sample rate are required"}), 400
        
        try:
            # Decode the base64 audio data
            audio_bytes = base64.b64decode(data['audio'])
            
            # Try to interpret as float32 first (default expected format)
            try:
                audio = np.frombuffer(audio_bytes, dtype=np.float32)
            except:
                # Fallback: try to interpret as int16 (common mobile format)
                try:
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    # Normalize int16 audio to float range [-1.0, 1.0]
                    audio = audio / 32768.0
                except:
                    return jsonify({"error": "Unsupported audio format. Please provide float32 or int16 audio data."}), 400
            
            # Handle stereo audio (mobile phones often record stereo)
            if 'channels' in data and data['channels'] == 2:
                # Convert from interleaved to separate channels
                # Use first channel only for analysis
                audio = audio[::2]
            
            # Ensure audio isn't empty
            if len(audio) == 0:
                return jsonify({"error": "Empty audio data received"}), 400
                
            # Get sample rate
            fs = float(data['sampleRate'])
            
        except Exception as e:
            import traceback
            traceback_str = traceback.format_exc()
            return jsonify({
                "error": str(e),
                "details": traceback_str
            }), 500
    
    else:
        return jsonify({"error": "Please provide audio as WAV file in form-data or as base64 in JSON"}), 400
    
    # Ensure audio isn't empty
    if len(audio) == 0:
        return jsonify({"error": "Empty audio data received"}), 400
    
    # Calculate TNR using ECMA 418-2 standard
    try:
        # Call the ECMA 418-2 TNR calculation function
        tnr_values, prominences, tonal_freqs, cb_ratios = tnr_ecma_418_2(audio, fs)
        
        # Check if any tonal noise was detected
        has_tonal_noise = not (isinstance(tnr_values, (int, float)) and tnr_values == 0)
        
        # Calculate metrics based on the results
        if has_tonal_noise:
            if isinstance(tnr_values, (list, np.ndarray)) and len(tnr_values) > 0:
                # Get the maximum TNR value according to ECMA 418-2
                max_tnr = float(np.max(tnr_values))
                mean_tnr = float(np.mean(tnr_values))
                # ECMA 418-2 weighted TNR (gives more weight to higher TNRs)
                weighted_tnr = float(np.sum(tnr_values**2) / np.sum(tnr_values)) if np.sum(tnr_values) > 0 else 0.0
                
                # Convert frequencies and TNR values to serializable format
                serializable_tonal_freqs = [round(float(freq), 1) for freq in tonal_freqs]
                serializable_tnr_values = [round(float(val), 1) for val in tnr_values]
                
                # Create pairs of frequency and TNR for clearer results
                freq_tnr_pairs = [{"frequency": freq, "tnr": tnr} 
                                  for freq, tnr in zip(serializable_tonal_freqs, serializable_tnr_values)]
                
                # Sort by TNR value descending
                freq_tnr_pairs.sort(key=lambda x: x["tnr"], reverse=True)
            else:
                max_tnr = float(tnr_values)
                mean_tnr = float(tnr_values)
                weighted_tnr = float(tnr_values)
                freq_tnr_pairs = []
        else:
            max_tnr = 0.0
            mean_tnr = 0.0
            weighted_tnr = 0.0
            freq_tnr_pairs = []
            
        # Prepare the TNR results according to ECMA 418-2 format
        tnr_results = {
            'has_tonal_noise': has_tonal_noise,
            'max_tnr': round(max_tnr, 1),
            'mean_tnr': round(mean_tnr, 1),
            'weighted_tnr': round(weighted_tnr, 1),
            'tonal_components': freq_tnr_pairs
        }
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return jsonify({
            "error": f"Error calculating ECMA 418-2 TNR: {str(e)}",
            "details": traceback_str
        }), 500
    
    # Prepare response data
    response_data = {
        "status": "success",
        "timestamp": datetime.utcnow().isoformat(),
        "standard": "ECMA 418-2",
        "results": tnr_results,
        "metadata": {
            "audio_length": len(audio),
            "duration_seconds": len(audio) / fs,
            "sample_rate": fs
        }
    }
    
    return jsonify(response_data)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
