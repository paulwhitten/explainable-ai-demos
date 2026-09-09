#!/bin/bash

# Change to the application directory
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Start the uvicorn server
exec uvicorn main_api:app --host 0.0.0.0 --port 8000
