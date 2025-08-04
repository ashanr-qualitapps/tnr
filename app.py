from flask import Flask, jsonify, request
from datetime import datetime
import numpy as np
import base64
from tnr_calculator import compute_tnr

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

@app.route('/api/users', methods=['GET'])
def get_users():
    return jsonify({"users": users})

@app.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = next((u for u in users if u["id"] == user_id), None)
    if user:
        return jsonify(user)
    return jsonify({"error": "User not found"}), 404

@app.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    if not data or 'name' not in data or 'email' not in data:
        return jsonify({"error": "Name and email are required"}), 400
    
    new_id = max([u["id"] for u in users]) + 1 if users else 1
    new_user = {
        "id": new_id,
        "name": data["name"],
        "email": data["email"]
    }
    users.append(new_user)
    return jsonify(new_user), 201

@app.route('/api/tnr/calculate', methods=['POST'])
def calculate_tnr():
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
            
        # Get sample rate and optional parameters
        fs = float(data['sampleRate'])
        nfft = data.get('nfft', 65536)
        cb_width_hz = data.get('criticalBandWidth', None)
        peak_prominence_db = data.get('peakProminenceDb', 6.0)
        
        # Calculate TNR
        results = compute_tnr(
            audio=audio,
            fs=fs,
            nfft=nfft,
            cb_width_hz=cb_width_hz,
            peak_prominence_db=peak_prominence_db
        )
        
        # Convert numpy types to Python types for JSON serialization
        for result in results:
            result['f0'] = float(result['f0'])
            result['P_tone'] = float(result['P_tone'])
            result['P_noise'] = float(result['P_noise'])
            result['TNR_dB'] = float(result['TNR_dB'])
        
        return jsonify({
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "results": results,
            "metadata": {
                "audio_length": len(audio),
                "duration_seconds": len(audio) / fs,
                "analyzed_samples": min(len(audio), nfft)
            }
        })
    
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        return jsonify({
            "error": str(e),
            "details": traceback_str
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
