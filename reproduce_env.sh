#!/bin/bash
set -e

echo "Setting up Dreamer environment..."

# 1. Download portable Python 3.8 if not present
if [ ! -d ".tools/python3.8" ]; then
    echo "Downloading portable Python 3.8..."
    mkdir -p .tools
    curl -L -o .tools/python3.8.tar.gz https://github.com/indygreg/python-build-standalone/releases/download/20240107/cpython-3.8.18+20240107-x86_64-unknown-linux-gnu-install_only.tar.gz
    
    echo "Extracting Python..."
    cd .tools
    tar -xf python3.8.tar.gz
    mv python python3.8
    rm python3.8.tar.gz
    cd ..
else
    echo "Portable Python 3.8 found."
fi

# 2. Create Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    .tools/python3.8/bin/python3 -m venv venv
else
    echo "Virtual environment found."
fi

# 3. Install Dependencies
echo "Installing dependencies..."
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo "Setup complete!"
echo "Run training with: ./venv/bin/python dreamer.py ..."
