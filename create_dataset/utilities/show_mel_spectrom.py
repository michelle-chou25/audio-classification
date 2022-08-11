import librosa
import numpy as np
import torchaudio
import librosa.display as display
import matplotlib.pyplot as plt
import matplotlib

fig = plt.figure(figsize=(256, 16))
s_show, sr_show = librosa.load("./relay_dual_mic/pyaudio_output1.wav")
s_show1, sr_show1 = librosa.load("./relay_dual_mic/pyaudio_output4.wav")
# ax = matplotlib.axes.Axes(fig, rect=[0, 0, 0, 0])
# mfcc_show = librosa.feature.mfcc(s_show, sr_show)
# display.specshow(mfcc_show, sr=sr_show, x_axis='s', y_axis='mel')

display.waveshow(s_show[:44100*2], 44100, alpha=0.8, x_axis='s', offset=0.5)
# plt.colorbar()
# plt.savefig("./mel_test_e.png")
plt.savefig("./s_show.png")
plt.show()

