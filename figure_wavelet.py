import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.io as sio
dataPath = 'E:/Learning/paper_loss/visio/sources/wavelet/'
fs = 360  # sr
t = np.linspace(0, 3, fs * 3)  # 
ecg_signal = sio.loadmat(dataPath + '208m.mat')['val']
noise_signal = sio.loadmat(dataPath + 'emm.mat')['val']
ecg_signal = ecg_signal[0, :fs*3] / 200
noise_signal = noise_signal[0, :fs*3] / 200

def generate_noisy_signal(signal, noise, snr_db):

    signal_power = np.mean(signal**2)
    # base on SNR
    noise_power = signal_power / (10**(snr_db / 10))

    adjusted_noise = noise * np.sqrt(noise_power / np.mean(noise**2))
   
    noisy_signal = signal + adjusted_noise
    return noisy_signal

snr_levels = [-6, 0, 6, 12]

wavelet = 'db1'  # Daubechies 4
level = 4  

fig, axes = plt.subplots(level + 2, len(snr_levels), figsize=(15, 10))
for i, snr_db in enumerate(snr_levels):
    noisy_signal = generate_noisy_signal(ecg_signal, noise_signal, snr_db)
 
    coeffs = pywt.wavedec(noisy_signal, wavelet, level=level)
    cA = coeffs[0]  # A4
    cD = coeffs[1:]  # D4, D3, D2, D1

    axes[0, i].plot(t, noisy_signal, label=f'Noisy ECG signal (SNR={snr_db} dB)', color='blue')
    axes[0, i].set_title(f'Noisy ECG signal (SNR={snr_db} dB)')

    # from D1 to D4 and A4
    for j in range(level):
        axes[j + 1, i].plot(cD[level - j - 1], color='red')
        axes[j + 1, i].set_title(f'Detail Coefficient (DC{j + 1})')
    # last row A4
    axes[level+1, i].plot(cA, color='green')
    axes[level+1, i].set_title(f'Approximation Coefficient (AC{level})')

plt.tight_layout()
plt.show(block=True)