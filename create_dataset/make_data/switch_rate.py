import os
import librosa
import numpy as np
import torchaudio
s_show, sr_show = librosa.load(rf'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\TestData\12min_env.wav',)


def start_convert(files_path, new_path, save_rate=22050):
    files = os.listdir(files_path)
    for one in files:
        one_path = os.path.join(files_path, one)
        ori_data, ori_sr = torchaudio.load(one_path)
        ori_data = torchaudio.transforms.Resample(ori_sr, save_rate)(ori_data)
        save_name = os.path.join(new_path, one)
        librosa.output.write_wav(save_name, ori_data[0].numpy().astype(np.float32), save_rate)

start_convert(rf"\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\all\OK",
              rf"\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\22050\OK")
