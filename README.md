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
from scipy.signal import stft, welch
from scipy.fftpack import fft
import librosa
import librosa.display
import pywt
from kymatio.numpy import Scattering1D
import os

# === Utility Functions ===

def load_csv_data(csv_path, landmark='nose'):
    df = pd.read_csv(csv_path)
    x = df[f'{landmark}_x'].values
    y = df[f'{landmark}_y'].values
    z = df[f'{landmark}_z'].values
    return x, y, z

# === Frequency Domain Transformation Functions ===

def plot_stft(signal, fs=30, axis='x'):
    f, t, Zxx = stft(signal, fs=fs)
    plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
    plt.title(f'STFT Magnitude - {axis}')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar()

def plot_psd(signal, fs=30, axis='x'):
    f, Pxx = welch(signal, fs=fs)
    plt.semilogy(f, Pxx)
    plt.title(f'Power Spectral Density - {axis}')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD')

def plot_mel_spectrogram(signal, sr=30, axis='x'):
    signal = librosa.util.fix_length(signal.astype(np.float32), size=512)
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=128, hop_length=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=sr, hop_length=64, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel Spectrogram - {axis}')

def plot_mfcc(signal, sr=30, axis='x'):
    signal = librosa.util.fix_length(signal.astype(np.float32), size=512)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfcc, x_axis='time')
    plt.colorbar()
    plt.title(f'MFCC - {axis}')

def plot_wavelet_packet(signal, axis='x'):
    wp = pywt.WaveletPacket(data=signal, wavelet='db1', mode='symmetric', maxlevel=3)
    levels = [node.path for node in wp.get_level(3, 'freq')]
    values = [wp[node.path].data for node in wp.get_level(3, 'freq')]
    plt.imshow(np.abs(values), aspect='auto', interpolation='nearest')
    plt.title(f'Wavelet Packet Decomposition - {axis}')
    plt.xlabel('Time')
    plt.ylabel('Sub-band')
    plt.colorbar()

def plot_wavelet_scattering(signal, axis='x'):
    T = len(signal)
    J = 5
    Q = 1
    signal = signal.astype(np.float32)
    scatter = Scattering1D(J=J, shape=T, Q=Q)
    Sx = scatter(signal)
    plt.imshow(Sx, aspect='auto', interpolation='nearest')
    plt.title(f'Wavelet Scattering Transform - {axis}')
    plt.xlabel('Scales')
    plt.ylabel('Coefficients')
    plt.colorbar()

# === General Plotting Function ===

def comparative_plot(csv_path, landmark='nose', fs=30):
    x, y, z = load_csv_data(csv_path)
    signals = {'x': x, 'y': y, 'z': z}
    funcs = [
        ('STFT', plot_stft),
        ('PSD', plot_psd),
        ('MelSpectrogram', plot_mel_spectrogram),
        ('MFCC', plot_mfcc),
        ('Wavelet Packet', plot_wavelet_packet),
        ('Wavelet Scattering', plot_wavelet_scattering)
    ]
    
    for axis, signal in signals.items():
        plt.figure(figsize=(18, 12))
        for idx, (name, func) in enumerate(funcs, 1):
            plt.subplot(3, 2, idx)
            try:
                func(signal, fs=fs, axis=axis)
            except Exception as e:
                plt.title(f"{name} - {axis} (Error: {e})")
        plt.tight_layout()
        plt.suptitle(f'Comparative Frequency Transformations - {landmark.upper()}_{axis.upper()}', fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.show()

# Example usage:
# comparative_plot('path_to_csv.csv', landmark='nose')
