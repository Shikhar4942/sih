#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Resizing
from tensorflow.keras.models import Model

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def generate_micro_doppler_signal(signal_type, fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    if signal_type == 'bird':
        body_doppler = 0.5 * np.sin(2 * np.pi * 50 * t)
        wing_flap = 0.3 * np.sin(2 * np.pi * 200 * t * np.sin(2 * np.pi * 4 * t))
        chirping = 0.2 * np.sin(2 * np.pi * 600 * t)
        return body_doppler + wing_flap + chirping
    elif signal_type == 'uav':
        body_doppler = 0.7 * np.sin(2 * np.pi * 30 * t)
        main_rotor = 0.4 * np.sin(2 * np.pi * 250 * t * np.sin(2 * np.pi * 3 * t))
        tail_rotor = 0.3 * np.sin(2 * np.pi * 500 * t * np.sin(2 * np.pi * 2 * t))
        return body_doppler + main_rotor + tail_rotor
    elif signal_type == 'ornithopter':
        body_doppler = 0.5 * np.sin(2 * np.pi * 40 * t)
        wing_flap = 0.4 * np.sin(2 * np.pi * 200 * t * np.sin(2 * np.pi * 2 * t))
        harmonic = 0.3 * np.sin(2 * np.pi * 500 * t * np.sin(2 * np.pi * 1 * t))
        return body_doppler + wing_flap + harmonic

fs = 10000
duration = 1.0
signal_types = ['bird', 'uav', 'ornithopter']
spectrograms = []
noisy_spectrograms = []

for signal_type in signal_types:
    clean_signal = generate_micro_doppler_signal(signal_type, fs, duration)
    noisy_signal = clean_signal + 0.3 * np.random.randn(len(clean_signal))
    _, _, Sxx_clean = signal.spectrogram(clean_signal, fs, nperseg=256, noverlap=128)
    _, _, Sxx_noisy = signal.spectrogram(noisy_signal, fs, nperseg=256, noverlap=128)
    spectrograms.append(10 * np.log10(Sxx_clean + 1e-8))
    noisy_spectrograms.append(10 * np.log10(Sxx_noisy + 1e-8))


spectrograms = np.array(spectrograms)
noisy_spectrograms = np.array(noisy_spectrograms)
_, freq_bins, time_bins = spectrograms.shape
x_train = noisy_spectrograms.reshape(len(noisy_spectrograms), freq_bins, time_bins, 1)
y_train = spectrograms.reshape(len(spectrograms), freq_bins, time_bins, 1)

x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())
y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min())

input_img = Input(shape=(freq_bins, time_bins, 1))

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Conv2D(256, (3, 3), activation='relu', padding='same')(x)


x = Conv2DTranspose(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(encoded)
x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoded = Resizing(freq_bins, time_bins)(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss=ssim_loss)
autoencoder.fit(x_train, y_train, epochs=50, batch_size=4, validation_split=0.2)

denoised_spectrograms = autoencoder.predict(x_train)
for i, signal_type in enumerate(signal_types):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title(f"Original ({signal_type.capitalize()})")
    plt.imshow(y_train[i, :, :, 0], aspect='auto', cmap='jet')
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.title(f"Noisy ({signal_type.capitalize()})")
    plt.imshow(x_train[i, :, :, 0], aspect='auto', cmap='jet')
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.title(f"Denoised ({signal_type.capitalize()})")
    plt.imshow(denoised_spectrograms[i, :, :, 0], aspect='auto', cmap='jet')
    plt.colorbar()
    plt.show()


# In[ ]:




