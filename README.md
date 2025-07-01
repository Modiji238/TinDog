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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# === Step 1: Load All CSVs and Prepare Sequence Data ===
def load_lstm_dataset(root='sessions', landmark='nose', sequence_length=50):
    X, y = [], []
    for session_path in glob.glob(os.path.join(root, 'session_*')):
        csv_path = os.path.join(session_path, 'CSVs')
        if not os.path.exists(csv_path):
            continue
        for file in glob.glob(os.path.join(csv_path, '*.csv')):
            label = os.path.splitext(os.path.basename(file))[0].lower()
            df = pd.read_csv(file)
            coords = df[[f'{landmark}_x', f'{landmark}_y', f'{landmark}_z']].values

            # Sliding window to create sequences
            for i in range(0, len(coords) - sequence_length + 1, 5):
                segment = coords[i:i + sequence_length]
                if segment.shape == (sequence_length, 3):
                    X.append(segment)
                    y.append(label)
    return np.array(X), np.array(y)

# === Step 2: Build and Train LSTM Model ===
def train_lstm_model(X, y, model_path='lstm_classifier.h5'):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42
    )

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(32, activation='relu'),
        Dense(y_cat.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=50, batch_size=16, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    model.save(model_path)
    joblib.dump(le, 'lstm_label_encoder.pkl')
    print("Model and label encoder saved.")

    preds = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(np.argmax(y_test, axis=1), np.argmax(preds, axis=1), target_names=le.classes_))

# === Step 3: Prediction Function ===
def predict_from_sequence(seq, model_path='lstm_classifier.h5', encoder_path='lstm_label_encoder.pkl'):
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    le = joblib.load(encoder_path)
    pred = model.predict(np.expand_dims(seq, axis=0))
    return le.inverse_transform([np.argmax(pred)])[0], np.max(pred)

# === Main Entry ===
if __name__ == '__main__':
    print("[INFO] Loading data and training LSTM model...")
    X, y = load_lstm_dataset()
    if len(X) == 0:
        print("[ERROR] No data found.")
    else:
        train_lstm_model(X, y)


# === Data Augmentation Ideas (as notes): ===
# 1. Time Warping: Speed up or slow down the sequence (resample with interpolation)
# 2. Jittering: Add slight Gaussian noise to coordinates
# 3. Magnitude Scaling: Multiply sequence by a small scalar (~1.1 or 0.9)
# 4. Rotation: Apply small 2D rotation on XY plane (~5 degrees)
# 5. Window Slicing: Randomly slice smaller chunks from longer sequences
# Make sure augmentations preserve anatomical correctness and temporal consistency.

# Would you like separate augmentation functions added directly into the pipeline?
