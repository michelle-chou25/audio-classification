# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch
import csv
import os

import torchaudio
import numpy as np
import torch
import torch.nn.functional
import yaml
from torch.utils.data import Dataset
import soundfile as sf

# np.random.seed(996)
# torch.manual_seed(996)
# torch.cuda.manual_seed_all(996)

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

class LoaderOneAudio:
    def __init__(self, ng_file_path, ok_file_path, mixup_factor=[(0.8, 1.0), (0.8, 1.0)], sound_rate=22050,
                 interval_second=1.5, shift_second=0.3, dataset_factor=0.5, time_mask_factor=0.35, training=False):
        ng_files = os.listdir(ng_file_path)
        ok_files = os.listdir(ok_file_path)
        self.sound_rate = sound_rate
        self.dataset_factor = dataset_factor
        self.mixup_factor = mixup_factor
        self.interval_second = interval_second
        self.shift_second = shift_second
        self.time_mask_factor = time_mask_factor
        self.training = training
        self.wav_ng = []
        self.wav_ok = []
        for one in ng_files:
            data, sr = torchaudio.load(os.path.join(ng_file_path, one))
            if one == '0_15s_ruo_long.wav':
                data *= 1.2
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
            if one == '1_ground_audio.wav':
                data *= 0.4
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
        print("OK audio total:", int(len(self.wav_ok) // (self.shift_second * self.sound_rate)))

    def __len__(self):
        if self.training:
            return int((len(self.wav_ng) - self.one_audio_data_len) // (self.shift_second * self.sound_rate))
        else:
            ng_num = int((len(self.wav_ng) - self.one_audio_data_len) // (self.shift_second * self.sound_rate))
            ok_num = int((len(self.wav_ok) - self.one_audio_data_len) // (self.shift_second * self.sound_rate))
            return ng_num + ok_num

    def __getitem__(self, item):
        datum = {}
        if self.training:
            ng_start = int(item * (self.shift_second * self.sound_rate))
            ng = self.wav_ng[ng_start: ng_start + self.one_audio_data_len]

            if np.random.rand() < self.time_mask_factor:
                rnd_start = np.random.randint(self.one_audio_data_len - self.one_audio_data_len // 3 - 1)
                rnd_length = np.random.randint(self.one_audio_data_len // 3)
                ng[rnd_start:rnd_start + rnd_length] = 0.

            ok_start = np.random.randint(0, len(self.wav_ok) - self.one_audio_data_len)
            ok = self.wav_ok[ok_start:ok_start + self.one_audio_data_len]
            # if np.random.rand() < 0.2:
            #     rnd_start = np.random.randint(self.one_audio_data_len - self.one_audio_data_len // 3 - 1)
            #     rnd_length = np.random.randint(self.one_audio_data_len // 3)
            #     ok[rnd_start:rnd_start + rnd_length] *= 0.5
            if np.random.rand() < self.dataset_factor:
                mix_ng_factor = np.random.uniform(self.mixup_factor[0][0], self.mixup_factor[0][1])
                mix_ok_factor = np.random.uniform(self.mixup_factor[1][0], self.mixup_factor[1][1])
                data = ng * mix_ng_factor + ok * mix_ok_factor

                # _factor = np.random.uniform(0.7, 1.3)
                # data *= _factor

                data = data #- data.mean()
                datum['wav'] = data[None, :].clone()
                datum['labels'] = '/m/ng'
                datum['f'] = (mix_ng_factor, mix_ok_factor)
                return datum
            else:
                if np.random.rand() < 0.8:
                    data = ok #- ok.mean()
                    datum['wav'] = data[None, :].clone()
                    datum['labels'] = '/m/ok'
                    datum['f'] = (-1, -1)
                else:
                    data = ng #- ng.mean()
                    datum['wav'] = data[None, :].clone()
                    datum['labels'] = '/m/ng'
                    datum['f'] = (-1, -1)
                return datum
        else:
            ng_all_num = int((len(self.wav_ng) - self.one_audio_data_len) // (self.shift_second * self.sound_rate))
            if item < ng_all_num:
                ng_start = int(item * (self.shift_second * self.sound_rate))
                ng = self.wav_ng[ng_start: ng_start + self.one_audio_data_len]
                data = ng #- ng.mean()
                datum['wav'] = data[None, :].clone()
                datum['labels'] = '/m/ng'
            else:
                ok_start = int((item - ng_all_num) * (self.shift_second * self.sound_rate))
                ok = self.wav_ok[ok_start: ok_start + self.one_audio_data_len]
                data = ok #- ok.mean()
                datum['wav'] = data[None, :].clone()
                datum['labels'] = '/m/ok'
            return datum
# #train
save_path = '../../dataset/train/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
if not os.path.exists(os.path.join(save_path, 'NG')):
    os.mkdir(os.path.join(save_path, 'NG'))
if not os.path.exists(os.path.join(save_path, 'OK')):
    os.mkdir(os.path.join(save_path, 'OK'))
for j in range(3):
    loa = LoaderOneAudio(
        rf"C:\workspace\zhouhe\datasets\relay\wz_relay_file\NG",
        rf"C:\workspace\zhouhe\datasets\relay\wz_relay_file\OK",
        training=True,
        mixup_factor=[(0.8, 1.0), (0.8, 1.0)],
    )
    for i in range(len(loa)):
        ret = loa[i]
        class_name = 'NG' if ret['labels'] == '/m/ng' else 'OK'
        save_name = f"{save_path}{class_name}/{j}_{ret['labels'][-2:]}({str(round(ret['f'][0], 2))}, {str(round(ret['f'][1], 2))})_{i}.wav"
        sf.write(save_name, ret['wav'][0].numpy().astype(np.float32), 22050, subtype='PCM_24')
exit(-1)


# #val
# for j in range(1):
#     loa = LoaderOneAudio(
#         rf"C:\workspace\zhouhe\datasets\relay\wz_relay_file_val\NG",
#         rf"C:\workspace\zhouhe\datasets\relay\wz_relay_file_val\OK",
#         training=False,
#         shift_second=1.0,
#         dataset_factor=0.5
#     )
#     for i in range(len(loa)):
#         ret = loa[i]
#         class_name = 'NG' if ret['labels'] == '/m/ng' else 'OK'
#         save_name = f"../../dataset/val/{class_name}/{j}_{ret['labels'][-2:]}_{i}.wav"
#         sf.write(save_name, ret['wav'][0].numpy().astype(np.float32), 22050, subtype='PCM_24')
# exit(-1)


class AudiosetDataset(Dataset):
    t_m = []
    t_std = []

    def __init__(self, config_path, audio_conf, training=False, label_csv=None, mixup_factor=[0.3, 0.7]):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        cfgs = yaml.load(open(config_path, 'r', encoding='utf-8').read(), Loader=yaml.FullLoader)
        train_cfgs = cfgs['TRAINING']
        # self.mixup_factor = mixup_factor
        self.datapath = train_cfgs['dataset_json_file']
        self.training = training
        # with open(dataset_json_file, 'r') as fp:
        #     data_json = json.load(fp)

        # self.data = data_json['data']

        self.data = LoaderOneAudio(
            os.path.join(train_cfgs['dataset_json_file'], 'NG'),
            os.path.join(train_cfgs['dataset_json_file'], 'OK'),
            mixup_factor=train_cfgs['mixup_factor'],  # NG和OK混合比例
            dataset_factor=train_cfgs['dataset_factor'],  # NG占比
            interval_second=train_cfgs['interval_second'],  # 每一帧时间长度1.5s
            shift_second=train_cfgs['shift_second'],  # 每一个 batch size 音频间隔时间0.3s
            sound_rate=train_cfgs['sound_rate'],
            time_mask_factor=train_cfgs['time_mask_factor'],
            training=self.training
        )
        self.audio_conf = audio_conf
        print('---------------the {:s} dataloader---------------'.format(self.audio_conf.get('mode')))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'),
                                                                      self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print(
                'use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

    def _wav2fbank(self, data):
        waveform, sr = data, self.data.sound_rate
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                  # frame_length=25,
                                                  # frame_shift=10
                                                  )

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def __getitem__(self, index):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        datum = self.data[index]
        label_indices = np.zeros(self.label_num)
        fbank = self._wav2fbank(datum['wav'])
        for label_str in datum['labels'].split(','):
            label_indices[int(self.index_dict[label_str])] = 1.0

        label_indices = torch.FloatTensor(label_indices)

        # # freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        # timem = torchaudio.transforms.TimeMasking(self.timem)
        # fbank = torch.transpose(fbank, 0, 1)
        # fbank = fbank.unsqueeze(0)
        # # if self.freqm != 0:
        # #     fbank = freqm(fbank)
        # if self.timem != 0:
        #     fbank = timem(fbank)
        # fbank = fbank.unsqueeze(0)
        # fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank, label_indices

    def __len__(self):
        return len(self.data)
