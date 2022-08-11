import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

fig = plt.figure(figsize=(256, 16))
file_name0 = '1_ground_audio.wav'
s_show0, sr_show0 = librosa.load(rf"./zcr_data/{file_name0}")

file_name1 = '8_12min_env.wav'
s_show1, sr_show1 = librosa.load(rf"./zcr_data/{file_name1}")

file_name2 = 'qiang.WAV'
s_show2, sr_show2 = librosa.load(rf"./zcr_data/{file_name2}")

file_name3 = 'zhong.WAV'
s_show3, sr_show3 = librosa.load(rf"./zcr_data/{file_name3}")

file_name4 = 'ruo2.WAV'
s_show4, sr_show4 = librosa.load(rf"./zcr_data/{file_name4}")
# zcr = librosa.feature.zero_crossing_rate(s_show, frame_length = 2048, hop_length = 512, center = True)
sr_show = np.concatenate([s_show0[:10240], np.zeros(512),
                          s_show1[:10240], np.zeros(256),
                          s_show2[:10240], np.zeros(256),
                          # s_show3[:20480], np.zeros(512),
                          # s_show2[:20480], np.zeros(512),
                          ]
                         )

zcrs_init = librosa.feature.zero_crossing_rate(sr_show, frame_length=2048,  hop_length=512)  # len(zcrs_init) = len(s_show)/512

# 画出音频波形和每一帧下的过零率
# plt.figure(figsize=(14, 5))
zcrs_times = librosa.frames_to_time(np.arange(len(zcrs_init[0])), sr=sr_show4, hop_length=512)
librosa.display.waveplot(sr_show, sr=sr_show4, alpha=0.7)
plt.plot(zcrs_times, zcrs_init[0], label='ZCR', lw=3, color='green')
plt.legend()
plt.savefig(f"./zcr_data/{'qiang1'}.png")
plt.show()
