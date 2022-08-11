import os.path
import numpy as np
import librosa
import soundfile as sf


def split_sound(input_file_path, output_path, volume=1., rate=None, interval_second=1.5, stride=0.5, file_name=''):
    rd = np.random.randint(0, high=999999)
    ori_data, ori_sr = librosa.load(input_file_path, sr=None)
    print('ori rate:', ori_sr, 'Hz')
    if rate is not None:
        ori_data = librosa.resample(ori_data, orig_sr=ori_sr, target_sr=rate)
        ori_sr = rate
    ori_data *= volume
    all_second = len(ori_data) / ori_sr
    interval_rate = int(ori_sr * interval_second)
    print('all second:', all_second, "s")
    print('all frame:', len(ori_data))
    print('new rate:', ori_sr, 'Hz')
    for idx, i in enumerate(range(0, len(ori_data), int(stride * ori_sr))):
        if i + interval_rate < len(ori_data):
            out_data = ori_data[i:i + interval_rate]
            sf.write(os.path.join(output_path, f"{file_name}_v{volume}_{idx * 0.5}_{idx * 0.5 + 0.5}_.wav"),
                     out_data, ori_sr, subtype='PCM_24')


files_path = rf'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\train\OK\NoiseData'
output_path = './NoiseData'

if not os.path.exists(output_path):
    os.mkdir(output_path)

if os.path.isfile(files_path):
    split_sound(files_path, output_path, volume=1.0, interval_second=1.5, stride=0.5, rate=22050, file_name='12min')
else:
    files = os.listdir(files_path)
    for one in files:
        split_sound(os.path.join(files_path, one), output_path, volume=1.0, interval_second=1.5,
                    stride=0.4, rate=22050, file_name=one.replace('.WAV', ''))