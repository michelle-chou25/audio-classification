import os

import librosa
import numpy as np
import soundfile

files_path = rf'E:\workspace\audio_relay\dataset\one_mic\train\OK\OK_org'
new_path = rf'E:\workspace\audio_relay\dataset'

files = os.listdir(files_path)

for one in files:
    data, sr = librosa.load(os.path.join(files_path, one))
    f = np.random.uniform(0.94, 1)

    if np.random.uniform() < 0.5:
        ret = np.stack([data * f, data], -1)
    else:
        ret = np.stack([data, data * f], -1)
    soundfile.write(os.path.join(new_path, one), ret, sr)