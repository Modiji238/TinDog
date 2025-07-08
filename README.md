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
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import random

# --- Load Data ---
def load_series(csv_path, landmark='nose'):
    df = pd.read_csv(csv_path)
    series = {}
    for axis in ['x', 'y', 'z']:
        col = f'{landmark}_{axis}'
        if col in df.columns:
            series[axis] = df[col].values
    return series

# --- 1. Jittering ---
def jitter(series, sigma=0.01):
    return series + np.random.normal(0, sigma, size=series.shape)

# --- 2. Scaling ---
def scale(series, factor=None):
    factor = factor if factor else np.random.uniform(0.9, 1.1)
    return series * factor

# --- 3. Time Warping ---
def time_warp(series, stretch_factor=None):
    from scipy.interpolate import interp1d
    stretch = stretch_factor if stretch_factor else np.random.uniform(0.8, 1.2)
    x_old = np.arange(len(series))
    x_new = np.linspace(0, len(series)-1, int(len(series)*stretch))
    f = interp1d(x_old, series, kind='linear', fill_value="extrapolate")
    return f(x_old)

# --- 4. Permutation ---
def permute(series, seg_count=4):
    segs = np.array_split(series, seg_count)
    np.random.shuffle(segs)
    return np.concatenate(segs)

# --- 5. Slicing ---
def slice_window(series, keep_ratio=0.9):
    L = len(series)
    keep = int(L * keep_ratio)
    start = np.random.randint(0, L - keep)
    return series[start:start+keep]

# --- 6. Magnitude Warping ---
def magnitude_warp(series, sigma=0.2, knot=4):
    from scipy.interpolate import CubicSpline
    xx = np.linspace(0, len(series), knot+2)
    yy = np.random.normal(1.0, sigma, knot+2)
    cs = CubicSpline(xx, yy)
    return series * cs(np.arange(len(series)))

# --- 7. Gaussian Smoothing ---
def smooth(series, sigma=1):
    return gaussian_filter1d(series, sigma)

# --- 8. Time Shift ---
def time_shift(series, shift=None):
    shift = shift if shift else np.random.randint(-10, 10)
    return np.roll(series, shift)

# --- 9. Random Drop ---
def random_drop(series, drop_prob=0.1):
    mask = np.random.rand(len(series)) > drop_prob
    return series * mask

# --- 10. Flip (Mirror) ---
def flip(series):
    return -series

# --- Plot Results ---
def plot_augmentation(original, augmented, title):
    plt.figure(figsize=(10, 4))
    plt.plot(original, label='Original', alpha=0.8)
    plt.plot(augmented, label='Augmented', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Run All Augmentations ---
def run_all_augmentations(csv_path, landmark='nose', axis='x'):
    data = load_series(csv_path, landmark)
    if axis not in data:
        print(f"{landmark}_{axis} not found in data.")
        return
    s = data[axis]

    plot_augmentation(s, jitter(s), 'Jitter')
    plot_augmentation(s, scale(s), 'Scaling')
    plot_augmentation(s, time_warp(s), 'Time Warping')
    plot_augmentation(s, permute(s), 'Permutation')
    plot_augmentation(s, slice_window(s), 'Window Slicing')
    plot_augmentation(s, magnitude_warp(s), 'Magnitude Warping')
    plot_augmentation(s, smooth(s), 'Gaussian Smoothing')
    plot_augmentation(s, time_shift(s), 'Time Shift')
    plot_augmentation(s, random_drop(s), 'Random Drop')
    plot_augmentation(s, flip(s), 'Flip')

# === EXAMPLE USAGE ===
# run_all_augmentations("sessions/session_2025-06-06_12-00-00/CSVs/flexion.csv", landmark='nose', axis='x')
