import librosa
import numpy as np
import librosa.display as display
import matplotlib.pyplot as plt
import soundfile

fig = plt.figure(num=1, figsize=(128, 32))
# s_show, sr_show = librosa.load("./env_dual_mic/pyaudio_output1.wav")
# s_show1, sr_show1 = librosa.load("./env_dual_mic/pyaudio_output4.wav")

s_show, sr_show = librosa.load("./pyaudio_output1.wav")
s_show1, sr_show1 = librosa.load("./pyaudio_output2.wav")
print(sr_show, sr_show1)
sr = 22050
cal_len_ = int(0.3 * sr)
start_ = int(0.5 * sr)
stride_ = int(0.001 * sr)
mfcc_fixed = librosa.feature.mfcc(y=s_show[start_:start_ + cal_len_], sr=sr)
print('start:', start_)
fig.add_subplot(3, 1, 1)
display.waveshow(s_show[0:sr//3], sr, alpha=0.8, x_axis='s', offset=0.5)


best_cor = 0
best_offset = 0

# 从start_往后偏移
for offset in range(0, start_, stride_):
    mfcc_move = librosa.feature.mfcc(y=s_show1[start_ + offset:start_ + cal_len_ + offset], sr=sr)
    cor = np.corrcoef(mfcc_fixed.reshape(-1), mfcc_move.reshape(-1))[0, 1]
    if cor > best_cor:
        best_cor = cor
        best_offset = start_ + offset
    # print(cor)
print('################################################################')
# 从start_往前偏移
for offset in range(0, start_, stride_):
    mfcc_move = librosa.feature.mfcc(y=s_show1[start_ - offset:start_ + cal_len_ - offset], sr=sr)
    cor = np.corrcoef(mfcc_fixed.reshape(-1), mfcc_move.reshape(-1))[0, 1]
    if cor > best_cor:
        best_cor = cor
        best_offset = start_ - offset
    # print(cor)

print("best_cor:", best_cor)
print("best_offset:", best_offset)

env = s_show[start_:start_+int(13*sr)]
# soundfile.write('./relay_dual_mic_new/pm_pyaudio_output1.wav', env, sr)
# relay = s_show1[best_offset:best_offset+int(13*sr)]
# soundfile.write('./relay_dual_mic_new/pm_pyaudio_output4_relay.wav', relay, sr)
# ret = relay - env
# soundfile.write('./ret.wav', ret, sr)

fig.add_subplot(3, 1, 2)
display.waveshow(s_show1[:sr//3], sr, alpha=0.8, x_axis='s', offset=0.5)

fig.add_subplot(3, 1, 3)
offset = start_-best_offset
show = np.concatenate([np.zeros(offset), s_show1[:sr-offset]])
display.waveshow(show[:sr//3], sr, alpha=0.8, x_axis='s', offset=0.5)

plt.savefig("./mfcc_show.png")
plt.show()