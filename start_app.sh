#!/bin/bash

# Navigate to the project directory
cd /home/lu/streamlit/DRFB

# Activate the virtual environment
source .venv/bin/activate

# Print a message
echo "Starting DRFB Streamlit App..."

# Run the application
streamlit run streamlit_app.py
