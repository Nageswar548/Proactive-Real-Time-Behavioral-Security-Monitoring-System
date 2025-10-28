# record_phone.py

import subprocess
import sys
import os

# --- Set the Action to 'Using_Phone' ---
os.environ['ACTION_NAME'] = 'Using_Phone'

# --- Run the main application ---
try:
    print("Starting data collection for: USING PHONE (Press 'q' to stop)...")
    # This runs the main_app.py file with the action name set above
    subprocess.run([sys.executable, 'main_app.py'])
except Exception as e:
    print(f"An error occurred: {e}")