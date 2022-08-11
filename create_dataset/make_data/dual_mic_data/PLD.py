import librosa
import numpy as np
import matplotlib.mlab as mlab
import soundfile
from scipy import signal as ss

speech = "01"
noise_type = "babble"
snr_level = ["0" "5" "10" "15"]

# Simulation Parameter
frameLength = 256  # frame length
npoint = 256  # n-point of fft
sp = 0.5  # percent of overlap
Length_Frame = sp * frameLength  # overlap frame
Length_Freq_Index = sp * npoint + 1  # no overlap frame
segment_window = np.ones(frameLength)        # Rectangle Window
mic_len = 0.1  # Distance between 2 Mic
sound_speed = 340  # Speed of Sound

# Smoothing Factor
a2 = 0.9  # Forgetting Factor Phi_Noise
a3 = 0.8  # Forgetting Factor Phi_Noise
q = 0.75  # HR Coeff

# Import Clean Speech
# signal_c, Fs_c = librosa.load('./relay_dual_mic_new/pyaudio_output4_relay.wav')
signal_c, Fs_c = librosa.load('./relay_dual_mic_new/pm_pyaudio_output4_relay.wav')
#
# Create Discrete Frequency Vector
t = np.arange(0, npoint//2)
fsignal = Fs_c * t / npoint

# signal, Fs = librosa.load('./relay_dual_mic_new/pyaudio_output1.wav')
signal, Fs = librosa.load('./relay_dual_mic_new/pm_pyaudio_output1.wav')

signal1 = signal_c
signal2 = signal

# Select Gramma Value for Wiener Filter
snr_12 = 20 * np.log10(np.linalg.norm(signal1) / np.linalg.norm(signal2))
if snr_12 < 3:
    gramma = 3.5
elif snr_12 < 6:
    gramma = 3.8
elif snr_12 < 8:
    gramma = 1.5
else:
    gramma = 0.2

# Calculate Signal Parameter
nframe = np.floor(len(signal1) / (frameLength * (1 - sp))) - 1

# Create Output Data
en_signal = np.zeros((int((nframe + 1) * Length_Frame)))
output_HR = np.zeros((int((nframe + 1) * Length_Frame)))

# Create Processing Data
phi_noise = np.zeros((int(nframe), int(Length_Freq_Index)))
G = np.zeros((int(nframe), int(Length_Freq_Index)))

## Signal Processing
# Initial Frame Window
currentFrame = 1
nextFrame = currentFrame + frameLength - 1

# Initial Err Value
Log_Err_C = 0

for k in range(int(nframe)):

    # Import Current Frame Signal
    signal1_d = signal1[currentFrame:nextFrame+1]
    signal2_d = signal2[currentFrame:nextFrame+1]

    # Fourier Transform
    signal1_ft_double = np.fft.fft(signal1_d, npoint)
    signal1_ft = signal1_ft_double[1:int(npoint / 2) + 1 + 1]
    signal1_ft[2:-1] = 2 * signal1_ft[2:-1]
    signal1_ft_m = np.abs(signal1_ft)  # Magnitude Data
    signal1_ft_ph = np.angle(signal1_ft)  # Phase Data

    signal2_ft_double = np.fft.fft(signal2_d, npoint)
    signal2_ft = signal2_ft_double[1:int(npoint / 2) + 1 + 1]
    signal2_ft[2:-1] = 2 * signal2_ft[2:-1]
    signal2_ft_m = np.abs(signal2_ft)  # Magnitude Data
    signal2_ft_ph = np.angle(signal2_ft)  # Phase Data

    # PSD Calculation
    psdsignal1 = ss.periodogram(signal1_d,
                                # window=segment_window,
                                # nfft=npoint,
                                # scaling='spectrum',
                                fs=Fs)[1]
    psdsignal2 = ss.periodogram(signal2_d,
                                # window=segment_window,
                                # nfft=npoint,
                                # scaling='spectrum',
                                fs=Fs)[1]
    # if signal1_d.shape[0] != 256:
    #     print(k)
    cross_psd = np.abs(ss.csd(signal1_d, signal2_d, fs=Fs)[1])

    # PLDNE Calculation Normalize
    phi_PLDNE = np.abs((psdsignal1 - psdsignal2) / (psdsignal1 + psdsignal2 + 1e-30))

    # Delta PLD Calculation
    phi_PLD = np.maximum((psdsignal1 - psdsignal2), 0)

    if k == 1:
        phi_noise[k, :] = (1 - a2) * psdsignal1
    else:
        for l in range(4, len(fsignal) - 10):
            if phi_PLDNE[l] < 0.2:
                phi_noise[k, 1] = a2 * phi_noise[k - 1, 1] + (1 - a2) * (psdsignal1[1])
            elif phi_PLDNE[l] > 0.8:
                phi_noise[k, l] = phi_noise[k - 1, l]
            else:
                phi_noise[k, l] = a3 * phi_noise[k - 1, l] + (1 - a3) * psdsignal2[l]

            # Calculate Transfer Function
            Tn1n2 = np.sinc(2 * np.pi * fsignal[l] * mic_len / sound_speed)
            H12 = (cross_psd[l] - Tn1n2 * phi_noise[k, l]) / (psdsignal1[l] - phi_noise[k, l])
            H12 = np.minimum(H12, 1)  # Limit Value

            # Create Weiner Filter
            G_c = phi_PLD[l] / (phi_PLD[l] + gramma * (1 - np.abs(H12) ** 2) * phi_noise[k, l] + 1e-30)
            G_c = np.minimum(np.maximum(G_c, 0), 1)  # Limit Value
            G[k, l] = G_c  # Append Data
    # Specctral Subpression
    en_signal_f_m = G[k, :] * signal1_ft_m
    en_signal_f = en_signal_f_m * np.exp(complex(0, 1) * signal1_ft_ph)

    # Inverse Fourier Transform
    en_signal_d = np.fft.ifft(en_signal_f, npoint)
    en_signal_c = en_signal_d[1:frameLength]
    en_signal_c = np.real(en_signal_c)

    # Export PLD Enhance Signal
    en_signal[currentFrame: nextFrame] = en_signal_c

    # Harmonic Regeneration
    harmo = np.maximum(en_signal_c, 0)

    # Harmonic Regeneration Fourier Transform
    HR_fft = np.fft.fft(harmo, npoint)
    HR_F = HR_fft[1: int(npoint / 2) + 1 + 1]  # Single - Side Fourier
    HR_F[2: - 1] = 2 * HR_F[2: -1]
    HR_fft_abs = np.abs(HR_F)  # Magnitude
    HR_fft_ph = np.angle(HR_F)  # Phase

    # Smoothing Recursive
    HR_fft_M = q * en_signal_f_m + 2. * (1 - q) * (HR_fft_abs)
    HR_fft_M[1] = 0

    # Export Harmonic Regeneration Signal
    output_HR_fft = HR_fft_M * np.exp(complex(0, 1) * signal1_ft_ph)
    output_HR_ifft = np.fft.ifft(output_HR_fft, npoint)
    output_HR_c = np.real(output_HR_ifft[1:frameLength+1])
    output_HR[currentFrame: nextFrame] = output_HR_c

    currentFrame = int(nextFrame - sp * frameLength + 1)
    nextFrame = int(currentFrame + frameLength - 1)

# print(output_HR)
soundfile.write('./pld.wav', output_HR, Fs)


