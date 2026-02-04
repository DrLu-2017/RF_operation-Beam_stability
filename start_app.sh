#!/bin/bash

# Navigate to the project directory
cd /home/lu/streamlit/albums-main

# Activate the virtual environment
source .venv/bin/activate

# Print a message
echo "Starting ALBuMS Streamlit App..."

# Run the application
streamlit run streamlit_app.py
