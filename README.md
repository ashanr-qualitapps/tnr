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

### User Management
- `GET /api/health` - Health check
- `GET /api/users` - Get all users
- `GET /api/users/<id>` - Get user by ID
- `POST /api/users` - Create new user (requires JSON body with name and email)

### TNR Analysis
- `POST /api/tnr/calculate` - Calculate TNR from audio data
- `POST /api/tnr/visualize` - Generate advanced visualizations for TNR analysis

## Example Usage

```bash
# Health check
curl http://localhost:5000/api/health

# Get all users
curl http://localhost:5000/api/users

# Create a user
curl -X POST http://localhost:5000/api/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Bob Johnson", "email": "bob@example.com"}'

# See curl_examples.md for TNR analysis examples
```
