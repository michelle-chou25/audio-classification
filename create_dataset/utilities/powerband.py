import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display as display

file_name=r'7_14_9_18_85_0.0.wav'
data, sr = librosa.load(file_name, mono=False)
print(data.shape)
fig = plt.figure(num=1, figsize=(128, 32))
fig.add_subplot(2, 2, 1)
s = np.abs(librosa.stft(data))
print(librosa.power_to_db(s ** 2).shape)
plt.colorbar(format='%+2.0f dB')
plt.axis('tight')
plt.show()

