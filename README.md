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
