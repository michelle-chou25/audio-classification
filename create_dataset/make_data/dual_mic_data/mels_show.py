import librosa
import numpy as np
import librosa.display as display
import matplotlib.pyplot as plt
import soundfile

fig = plt.figure(num=1, figsize=(128, 32))
# s_show, sr_show = librosa.load("./env_dual_mic/pyaudio_output1.wav")
# s_show1, sr_show1 = librosa.load("./env_dual_mic/pyaudio_output4.wav")

s_show, sr_show = librosa.load("./relay_dual_mic_new/pm_pyaudio_output1.wav")
s_show1, sr_show1 = librosa.load("./relay_dual_mic_new/pm_pyaudio_output4_relay.wav")
s_show2, sr_show2 = librosa.load("./pld.wav")



sr = 22050
fig.add_subplot(3, 1, 1)
display.waveshow(s_show[-sr:-10000], sr, alpha=0.8, x_axis='s', offset=0.5)

fig.add_subplot(3, 1, 2)
display.waveshow(s_show1[-sr:-10000], sr, alpha=0.8, x_axis='s', offset=0.5)

fig.add_subplot(3, 1, 3)
display.waveshow(s_show2[-sr:-10000], sr, alpha=0.8, x_axis='s', offset=0.5)

plt.savefig("./mfccs_show.png")
plt.show()