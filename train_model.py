# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# --- 1. Setup Paths and Model Saving ---
DATA_PATH = os.path.join('Action_Data', 'coords.csv')
MODEL_PATH = os.path.join('security_action_model.pkl')
ENCODER_PATH = os.path.join('action_encoder.pkl')

# --- 2. Load and Prepare Data ---
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"ERROR: Data file not found at {DATA_PATH}.")
    print("Please ensure you have run the recording scripts (record_phone.py and record_sitting.py) first.")
    exit()

print(f"Data loaded. Total rows: {len(df)}")
print(f"Actions found: {df['action'].unique()}")

# Separate features (X) from labels (y)
X = df.drop('action', axis=1) # All columns except 'action' are features
y_raw = df['action']          # The 'action' column is the label

# --- 3. Encode Labels ---
# Convert action names (text) into numbers (0, 1, 2, ...)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
print(f"Labels encoded: {label_encoder.classes_}")

# --- 4. Split Data for Training and Testing ---
# 80% for training the model, 20% for testing how well it works
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
print(f"Training data size: {len(X_train)} samples")
print(f"Testing data size: {len(X_test)} samples")

# --- 5. Train the Random Forest Model ---
print("\nStarting model training...")
# Random Forest is a robust classifier good for this type of data
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
print("Model training complete.")

# --- 6. Evaluate the Model ---
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# --- 7. Save the Trained Model and Encoder ---
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(classifier, f)
    print(f"\nModel successfully saved to {MODEL_PATH}")

with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)
    print(f"Label Encoder successfully saved to {ENCODER_PATH}")

print("\nTraining process finished. You can now proceed to the Real-Time Detection step (Step 6).")