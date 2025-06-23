# TinDog
Tinder For Dogs-

1)Responsive Design:
  -Ensure the site is fully responsive and works well on different devices (mobiles, tablets, desktops).
  -Use Bootstrap's grid system and responsive utilities.
  
2)Consistent and Modern UI:
  -Use modern and clean design principles.
  -Ensure consistency in font sizes, colors, and spacing.
  -Use a cohesive color scheme that is appealing and appropriate for the theme.

Includes features such as hero sections,attractive and responsive buttons,a pricing table and a Carousel for brandings



import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from datetime import datetime

# === Step 1: Extract Polynomial Coefficients ===
def extract_polynomial_features(csv_path, landmark='nose', degree=3):
    df = pd.read_csv(csv_path)
    t = np.arange(len(df))

    coeffs = []
    for axis in ['x', 'y', 'z']:
        col = f'{landmark}_{axis}'
        if col in df.columns:
            poly = np.polyfit(t, df[col], degree)
            coeffs.extend(poly)
        else:
            coeffs.extend([0]*(degree+1))  # padding if column missing
    return coeffs

# === Step 2: Load All Training Data ===
def load_dataset(root='sessions', landmark='nose', degree=3):
    data, labels = [], []
    for session_path in glob.glob(os.path.join(root, 'session_*')):
        csv_path = os.path.join(session_path, 'CSVs')
        if not os.path.exists(csv_path):
            continue
        for file in glob.glob(os.path.join(csv_path, '*.csv')):
            label = os.path.splitext(os.path.basename(file))[0].lower()
            features = extract_polynomial_features(file, landmark, degree)
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

# === Step 3: Train Model and Save ===
def train_and_save_model(X, y, model_path='exercise_classifier.pkl'):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("\nClassification Report:")
    print(classification_report(y_test, clf.predict(X_test), target_names=le.classes_))

    joblib.dump({'model': clf, 'labels': le}, model_path)
    print(f"\nModel saved to: {model_path}")

# === Step 4: Predict New Sample ===
def predict_exercise(csv_file, model_path='exercise_classifier.pkl', landmark='nose'):
    model_data = joblib.load(model_path)
    clf = model_data['model']
    le = model_data['labels']
    features = extract_polynomial_features(csv_file, landmark)
    pred = clf.predict([features])[0]
    print(f"\nPredicted Exercise: {le.inverse_transform([pred])[0]}")
    return le.inverse_transform([pred])[0]

# === Step 5: Accept Feedback and Learn Incrementally ===
def update_model_with_feedback(csv_file, true_label, model_path='exercise_classifier.pkl', landmark='nose'):
    model_data = joblib.load(model_path)
    clf = model_data['model']
    le = model_data['labels']

    features = extract_polynomial_features(csv_file, landmark)
    X = np.array(model_data.get('X', []))
    y = np.array(model_data.get('y', []))

    # Encode label if new
    if true_label not in le.classes_:
        le.classes_ = np.append(le.classes_, true_label)

    y_new = le.transform([true_label])[0]
    X = np.vstack([X, features]) if X.size else np.array([features])
    y = np.append(y, y_new)

    clf.fit(X, y)
    joblib.dump({'model': clf, 'labels': le, 'X': X, 'y': y}, model_path)
    print(f"Model updated with feedback and saved to: {model_path}")

# === Main Training Call ===
if __name__ == '__main__':
    print("\n[INFO] Loading and training model from all session CSVs...")
    X, y = load_dataset()
    if len(X) == 0:
        print("[ERROR] No training data found.")
    else:
        train_and_save_model(X, y)
