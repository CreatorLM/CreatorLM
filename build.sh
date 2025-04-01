#!/bin/bash
# Clear any cached installations
rm -rf ~/.cache/pip

# Install Python dependencies with explicit upgrades
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements.txt

# Install system dependencies
sudo apt-get update
sudo apt-get install -y ffmpeg

# Verify critical installations
python -m pip list | grep -E "uvicorn|fastapi"
which ffmpeg