#!/bin/bash
# Start script for Render deployment

# Get the port from environment variable (Render provides this)
PORT=${PORT:-8000}

# Start the FastAPI application
uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
