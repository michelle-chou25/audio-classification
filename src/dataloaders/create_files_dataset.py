# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import json
import os

import librosa
import torchaudio
import numpy as np
import torch
import torch.nn.functional
import yaml
from torch.utils.data import Dataset

np.random.seed(996)

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup


def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list


def preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


class LoaderOneAudio:
    def __init__(self, ng_file_path, ok_file_path, mixup_factor=[(0.7, 1.0), (0.8, 1.0)], sound_rate=22050,
                 interval_second=1.5, shift_second=0.1, dataset_factor=0.5, time_mask_factor=0.35):
        ng_files = os.listdir(ng_file_path)
        ok_files = os.listdir(ok_file_path)
        self.sound_rate = sound_rate
        self.dataset_factor = dataset_factor
        self.mixup_factor = mixup_factor
        self.interval_second = interval_second
        self.shift_second = shift_second
        self.time_mask_factor = time_mask_factor
        self.wav_ng = []
        self.wav_ok = []
        for one in ng_files:
            data, sr = torchaudio.load(os.path.join(ng_file_path, one))
            if data.shape[0] != 1 or sr != sound_rate:
                data = torchaudio.transforms.Resample(sr, sound_rate)(data[0, :])
            else:
                data = data[0]
            self.wav_ng.append(data)
            self.wav_ng.append(torch.zeros([200]))
            print(f"NG loaded:shape:{data.shape[0]}, sr:{sr}", os.path.join(ng_file_path, one))
        self.wav_ng = torch.cat(self.wav_ng)
        for one in ok_files:
            data, sr = torchaudio.load(os.path.join(ok_file_path, one))
            if data.shape[0] != 1 or sr != sound_rate:
                data = torchaudio.transforms.Resample(sr, sound_rate)(data[0, :])
            else:
                data = data[0]
            self.wav_ok.append(data)
            self.wav_ok.append(torch.zeros([200]))
            print(f"OK loaded:shape:{data.shape[0]}, sr:{sr}", os.path.join(ok_file_path, one))
        self.wav_ok = torch.cat(self.wav_ok)
        self.one_audio_data_len = int(self.interval_second * self.sound_rate)
        print("NG audio total:", int(len(self.wav_ng) // (self.shift_second * self.sound_rate)))

    def __len__(self):
        return int((len(self.wav_ng) - self.one_audio_data_len) // (self.shift_second * self.sound_rate))

    def __getitem__(self, item):
        datum = {}
        ng_start = int(item * (self.shift_second * self.sound_rate))
        ng = self.wav_ng[ng_start: ng_start + self.one_audio_data_len]
        if np.random.rand() < self.time_mask_factor:
            rnd_start = np.random.randint(self.one_audio_data_len - self.one_audio_data_len // 3 - 1)
            rnd_length = np.random.randint(self.one_audio_data_len // 3)
            ng[rnd_start:rnd_start + rnd_length] = 0
            datum['mask'] = (rnd_start, rnd_length)
        else:
            datum['mask'] = (0, 0)

        ok_start = np.random.randint(0, len(self.wav_ok) - self.one_audio_data_len)
        ok = self.wav_ok[ok_start:ok_start + self.one_audio_data_len]
        # if np.random.rand() < self.time_mask_factor:
        #     rnd_start = np.random.randint(self.one_audio_data_len - self.one_audio_data_len // 3 - 1)
        #     rnd_length = np.random.randint(self.one_audio_data_len // 3)
        #     ok[rnd_start:rnd_start + rnd_length] = 0
        #     datum['mask'] = (rnd_start, rnd_length)
        # else:
        #     datum['mask'] = (0, 0)

        if np.random.rand() < self.dataset_factor:
            mix_ng_factor = np.random.uniform(self.mixup_factor[0][0], self.mixup_factor[0][1])
            mix_ok_factor = np.random.uniform(self.mixup_factor[1][0], self.mixup_factor[1][1])
            data = ng * mix_ng_factor + ok * mix_ok_factor
            # data = data - data.mean()
            datum['wav'] = data[None, :].clone()
            datum['labels'] = '/m/ng'
            datum['f'] = (mix_ng_factor, mix_ok_factor)
            return datum
        else:
            if np.random.rand() < 0.8:
                # data = ok - ok.mean()
                datum['wav'] = ok[None, :].clone()
                datum['labels'] = '/m/ok'
                datum['f'] = (-1, -1)
                datum['mask'] = (ok_start, 0)
            else:
                # data = ng - ng.mean()
                datum['wav'] = ng[None, :].clone()
                datum['labels'] = '/m/ng'
                datum['f'] = (-1, -1)
            return datum
list_ = []
pool = 3
num = 0
for j in range(pool):
    loa = LoaderOneAudio(
        "/home/zhouhe/datasets/wz_relay_file/NG",
        "/home/zhouhe/datasets/wz_relay_file/OK",
    )
    ng_save_path = '/home/share_data/ext/PVDefectData/test2021/zh/dt/wz/files_mode_wz_relay/NG'
    ok_save_path = '/home/share_data/ext/PVDefectData/test2021/zh/dt/wz/files_mode_wz_relay/OK'
    for i in range(len(loa)):
        ret = loa[i]
        if ret['labels'] == '/m/ng':
            save_path = f"{ng_save_path}/{num}_({ret['mask'][0]},{ret['mask'][1]})#({str(round(ret['f'][0], 2))},{str(round(ret['f'][1], 2))}).wav"
        else:
            save_path = f"{ok_save_path}/{num}_({ret['mask'][0]},{ret['mask'][1]})#({str(round(ret['f'][0], 2))},{str(round(ret['f'][1], 2))}).wav"
            list_.append(ret['mask'][0])
        librosa.output.write_wav(save_path, ret['wav'][0].numpy(), 22050)
        print(num, '/', len(loa)*pool)
        num += 1
print('end!')
