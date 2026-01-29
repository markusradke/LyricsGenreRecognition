#!/bin/bash
cd "$(dirname "$0")/.."
python3 -m venv lyrics_env
source lyrics_env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "Environment setup complete!"

# make shure the script is exexucatble beforehand