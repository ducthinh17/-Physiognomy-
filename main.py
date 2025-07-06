#!/usr/bin/env python3
"""
Entry point for Render deployment
"""
import os
import uvicorn
from api.main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
