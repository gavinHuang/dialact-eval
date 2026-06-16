"""
Standalone IVR mock server entry point.

Usage:
    IVR_CONFIG=flows/example.yaml python -m ivr.main
    # or
    uvicorn ivr.main:app
"""

import uvicorn
from .server import app

if __name__ == "__main__":
    uvicorn.run("ivr.main:app", host="0.0.0.0", port=8001, reload=True)
