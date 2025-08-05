# TNR Flask API

A simple Flask REST API with basic user management endpoints and TNR (Tone-to-Noise Ratio) calculation functionality.

## Setup

### Option 1: Local Python Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

### Option 2: Docker Setup

1. Build and run with docker-compose:
```bash
docker-compose up --build
```

2. Or build and run with Docker commands:
```bash
# Build the image
docker build -t tnr-api .

# Run the container
docker run -p 5000:5000 tnr-api
```

## API Endpoints

- `GET /api/health` - Health check

### TNR Analysis
- `POST /api/tnr/calculate_ecma_418_2 ` - Calculate TNR from audio data ECMA 418-2 compliant

## Example Usage

```bash
# Health check
curl http://localhost:5000/api/health

