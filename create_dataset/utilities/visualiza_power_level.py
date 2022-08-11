import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from numpy.linalg import norm
import scipy.io.wavfile as wav
from datetime import datetime
# import split_sound

# get the value of SNR
def SNR(x1, x2):
    # return 10*np.log(norm(x1) / norm(x2))
    p_x1 = np.sqrt(norm(x1))
    p_x2 = np.sqrt(norm(x2))
    return 10*np.log( (p_x1/len(x1))/(p_x2)/len(x2) )

def signal_by_db(x1,x2, snr, handle_method='cut'):
    """
    Generate noise mixed speech with given signal noise rate.
    Args:
        x1: speech, ndArray
        x2: noise, ndArray
        snr:  signal noise rate, int
        handle_method: generate the wav according the shorter one or the longer one. 'cut' or 'append'

    Returns:
        mix: mixed wav ndArray with

    """
    x1 = x1.astype(np.int32)
    x2 = x2.astype(np.int32)
    l1 = x1.shape[0]
    l2 = x2.shape[0]
    if l1 != l2:
        if handle_method == 'cut':
            ll = min(l1,l2)
            x1 = x1[:ll]
            x2 = x2[:ll]
        elif handle_method == 'append':
            ll = max(l1, l2)
            if l1 < ll:
                x1 = np.append(x1, x1[:ll-l1])
            if x2 < ll:
                x2 = np.append(x2, x2[:l2-ll])
                ll2 = min(x1.shape[0], x2.shape[0])
                x1 = x1[:ll2]
                x2 = x2[:ll2]
    # x2 = x2 / norm(x2) * norm(x1) / (10.0 ** (0.05 * snr))
    if x1.shape != x2.shape:
        x1 = x1[:, None]  # expan x1 from 1 dimension to 2 dimensions
    mix = x1+x2
    return mix


if __name__ == '__main__':
    snr_level = [0, 10, 20]
    num_FFT = 512
    hop_size = 128

    # draw the power graph

    sr, relay_data = wav.read('./7_14_9_18_85_0.0.wav') #mic1
    sr, noise_data=wav.read('../make_data/1_30s_7.8.wav') #mic2
    relay_data = relay_data.astype(float)
    S = librosa.stft(relay_data, n_fft=num_FFT, hop_length=hop_size, window='hanning')
    plt.figure(figsize=(10, 10))
    S = np.log(np.abs(S) ** 2)
    plt.subplot(311)
    plt.imshow(
    librosa.power_to_db(librosa.feature.melspectrogram(y=relay_data, sr=sr, n_fft=num_FFT, hop_length=hop_size),
                            ref=np.max), cmap="hot")
    plt.title('Power(dB) of product')
    plt.colorbar(format='%+2.0f dB')

    # create relay sounds with differnet signal noise rate

    # noisy_speech = signal_by_db(relay_data, noise_data, 20, 'cut')
    # time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # soundfile.write('./'+time+'.wav', noisy_speech, sr)
    # S = librosa.stft(relay_data, n_fft=num_FFT, hop_length=hop_size, window='hanning')
    # S = np.log(np.abs(S) ** 2)
    # plt.subplot(312)
    # plt.imshow(
    # librosa.power_to_db(librosa.feature.melspectrogram(y=noisy_speech, sr=sr, n_fft=num_FFT, hop_length=hop_size),
    #                         ref=np.max), cmap="hot")
    # plt.title('Power(dB) of product with higher signal noise rate')
    # plt.colorbar(format='%+2.0f dB')

    plt.show()
