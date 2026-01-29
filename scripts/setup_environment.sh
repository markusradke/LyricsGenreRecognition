#!/bin/bash
cd "$(dirname "$0")/.."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install --upgrade setuptools wheel
pip install -r requirements.txt
echo "Environment setup complete!"

# make shure the script is exexucatble beforehand with chmod +x scripts/setup_environment.sh