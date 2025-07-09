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


import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import os

def interpolate_csv(csv_path, target_rate=30):
    df = pd.read_csv(csv_path)
    original_len = len(df)
    original_time = np.linspace(0, 1, original_len)
    target_len = int((original_len - 1) * target_rate / 10)  # scale from ~10Hz to target
    new_time = np.linspace(0, 1, target_len)

    interpolated_data = {}
    for col in df.columns:
        interpolator = interp1d(original_time, df[col], kind='cubic')
        interpolated_data[col] = interpolator(new_time)

    new_df = pd.DataFrame(interpolated_data)
    
    # Save with suffix
    out_path = csv_path.replace('.csv', f'_interpolated_{target_rate}Hz.csv')
    new_df.to_csv(out_path, index=False)
    print(f"[✓] Interpolated CSV saved to: {out_path}")

    return new_df






import os
import pandas as pd
import numpy as np
from datetime import datetime

# === Augmentation Functions ===
def add_noise(df, noise_level=0.002):
    noise = np.random.normal(0, noise_level, df.shape)
    return df + noise

def time_warp(df, stretch_factor=1.2):
    indices = np.arange(0, len(df), stretch_factor)
    indices = indices[indices < len(df)].astype(int)
    return df.iloc[indices].reset_index(drop=True)

def reverse_sequence(df):
    return df[::-1].reset_index(drop=True)

def scale_coordinates(df, scale=1.1):
    return df * scale

def smooth_coordinates(df, window_size=5):
    return df.rolling(window=window_size, min_periods=1, center=True).mean()

# === Save Augmented CSV ===
def save_augmented_csv(df_aug, original_path, method):
    base_dir = os.path.dirname(original_path)
    filename = os.path.splitext(os.path.basename(original_path))[0]
    new_name = f"{filename}_augmented_{method}.csv"
    
    # Save in the same session/*/CSVs directory
    new_path = os.path.join(base_dir, new_name)
    df_aug.to_csv(new_path, index=False)
    print(f"Saved: {new_path}")

# === Augment Entire Landmark CSV ===
def augment_landmark_csv(csv_path):
    df = pd.read_csv(csv_path)
    landmark_cols = [col for col in df.columns if any(k in col for k in ['_x', '_y', '_z'])]
    df_landmarks = df[landmark_cols].copy()

    augmentations = {
        'noise': add_noise,
        'timewarp': time_warp,
        'reverse': reverse_sequence,
        'scale': scale_coordinates,
        'smooth': smooth_coordinates,
    }

    for method, func in augmentations.items():
        try:
            df_aug = func(df_landmarks.copy())
            save_augmented_csv(df_aug, csv_path, method)
        except Exception as e:
            print(f"Error in {method}: {e}")

# === Run Example ===
if __name__ == '__main__':
    input_csv = 'sessions/session_2025-06-06_12-00-00/CSVs/flexion.csv'  # Replace with your actual path
    augment_landmark_csv(input_csv)

    
