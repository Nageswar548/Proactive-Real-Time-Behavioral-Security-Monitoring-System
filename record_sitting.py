# record_sitting.py

import subprocess
import sys
import os

# --- Set the Action to 'Sitting' ---
os.environ['ACTION_NAME'] = 'Sitting'

# --- Run the main application ---
try:
    print("Starting data collection for: SITTING (Press 'q' to stop)...")
    # This runs the main_app.py file with the action name set above
    subprocess.run([sys.executable, 'main_app.py'])
except Exception as e:
    print(f"An error occurred: {e}")