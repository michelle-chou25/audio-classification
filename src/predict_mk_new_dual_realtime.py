import csv
import os
import time
import cv2
import librosa
import soundfile
import torchaudio
import torch
import ast
import torch
import argparse
import wave
import pyaudio
import numpy as np

CHUNK = 1024  # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 采样位数
# FORMAT = pyaudio.paFloat32  # 采样位数
CHANNELS = 1  # 单声道
RATE = 22050  # 采样频率

pppp = pyaudio.PyAudio()
# p.open()
info = pppp.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    if (pppp.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", pppp.get_device_info_by_host_api_device_index(0, i).get('name'))


def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


class Relay:
    def __init__(self, audio_conf, label_csv):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file
        """
        # self.datapath = dataset_json_file
        # with open(dataset_json_file, 'r') as fp:
        #     data_json = json.load(fp)

        # self.data = data_json['data']
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

    def _wav2fbank(self, waveform, sr):
        # mixup
        # waveform, sr = torchaudio.load(filename)
        waveform = waveform - waveform.mean()

        fbank0 = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                   window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                   channel=0,  ########0
                                                   frame_shift=10)
        fbank1 = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                   window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                   channel=1,  #########1
                                                   frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank0.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank0 = m(fbank0)
            fbank1 = m(fbank1)
        elif p < 0:
            fbank0 = fbank0[0:target_length, :]
            fbank1 = fbank1[0:target_length, :]
        fbank = torch.stack([fbank0, fbank1])

        return fbank, 0

    def preprocess(self, data, sr):
        """
        returns: image, audio, nframes
        where image is a FloatTensor of size (3, H, W)
        audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
        nframes is an integer
        """
        # do mix-up for this sample (controlled by the given mixup rate)
        # label_indices = np.zeros(self.label_num)
        fbank, mix_lambda = self._wav2fbank(data, sr)
        # for label_str in datum['labels'].split(','):
        #     label_indices[int(self.index_dict[label_str])] = 1.0

        # label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        # freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        # timem = torchaudio.transforms.TimeMasking(self.timem)
        # fbank = torch.transpose(fbank, 0, 1)
        # if self.freqm != 0:
        #     fbank = freqm(fbank)
        # if self.timem != 0:
        #     fbank = timem(fbank)
        # fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        fbank = fbank[None, :]
        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank  # , label_indices


_eff_b = 0
_target_length = 172

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--weight_path", type=str,
                    default="../exp/audio_model.11.pth",
                    # default="../exp/audio_model_wa.pth",
                    help="model file path")
parser.add_argument("--model", type=str, default="efficientnet", help="audio model architecture",
                    choices=["efficientnet", "resnet", "mbnet"])
parser.add_argument("--eff_b", type=int, default=_eff_b,
                    help="which efficientnet to use, the larger number, the more complex")
parser.add_argument("--n_class", type=int, default=2, help="number of classes")
parser.add_argument('--impretrain', help='if use imagenet pretrained CNNs', type=ast.literal_eval, default='False')
parser.add_argument("--att_head", type=int, default=4, help="number of attention heads")
parser.add_argument("--target_length", type=int, default=_target_length, help="the input length in frames")
parser.add_argument("--dataset_mean", type=float, default=-4.6476,
                    help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, default=4.5699, help="the dataset std, used for input normalization")
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used",
                    choices=["audioset", "esc50", "speechcommands"])
parser.add_argument("--data-val", type=str, default='../wz_relay/wz_relay.json', help="validation data json")
parser.add_argument("--label-csv", type=str, default='../wz_relay/wz_relay.csv', help="csv with class labels")

args = parser.parse_args()

# audio_model = torch.jit.load('./7.19(add)_mbv3_cpu.pth')
# audio_model = torch.jit.load('./7.19_add_0~220_mbv3_cpu.pth')###########ok
# audio_model = torch.jit.load('./7.20_chao_ruo_error_env_mbv3_cpu.pth')
# audio_model = torch.jit.load('./7.19_add_0~220_ruo_mbv3_cpu.pth')
audio_model = torch.jit.load('./7.20_chao_ruo_error_env_mbv3_cpu.pth')
val_audio_conf = {'num_mel_bins': 384, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean,
                  'std': args.dataset_std, 'noise': False}
print(audio_model)
audio_model.float().eval()  # .cuda()

data = np.zeros([1, 172, 384], dtype=np.float32)
data = torch.from_numpy(data)  # .cuda()
r = Relay(val_audio_conf, args.label_csv)
# ori_data, ori_sr = librosa.load('/home/zhouhe/datasets/wz_relay/NG/902091_7.0_7.5.wav', sr=None)
# ori_data, ori_sr = torchaudio.load('./0_30s_test_ruo.wav')
interval = 1.5
p1 = pyaudio.PyAudio()  # 实例化对象
stream1 = p1.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=int(RATE * interval),
                  input_device_index=1,
                  )  # 打开流，传入响应参数
p2 = pyaudio.PyAudio()  # 实例化对象
stream2 = p2.open(format=FORMAT,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  frames_per_buffer=int(RATE * interval),
                  input_device_index=4,
                  )  # 打开流，传入响应参数
# CV show 框
no = np.full([300, 300, 3], [0, 255, 0], dtype=np.uint8)
error = np.full([300, 300, 3], [0, 0, 255], dtype=np.uint8)
cv2.imshow('result:', no)
interval_num = 3
time_interval = interval/interval_num  # 0.15
idx = 0
realtime_data1 = stream1.read(int(RATE * interval))
realtime_data2 = stream2.read(int(RATE * interval))
realtime_data1 = np.frombuffer(realtime_data1, dtype=np.int16)[None, :]
realtime_data2 = np.frombuffer(realtime_data2, dtype=np.int16)[None, :]
while True:
    data_mk1 = stream1.read(int(RATE * time_interval))
    data_mk2 = stream2.read(int(RATE * time_interval))

    start_ = time.time()
    out_data1 = np.frombuffer(data_mk1, dtype=np.int16)[None, :]
    out_data2 = np.frombuffer(data_mk2, dtype=np.int16)[None, :]
    out_data1 = realtime_data1 = np.concatenate([realtime_data1[:, int(RATE * time_interval):], out_data1], axis=-1)
    out_data2 = realtime_data2 = np.concatenate([realtime_data2[:, int(RATE * time_interval):], out_data2], axis=-1)
    out_data1 = torch.from_numpy(out_data1 / 32768)
    out_data2 = torch.from_numpy(out_data2 / 32768)
    data = r.preprocess(torch.concat([out_data1, out_data2]), RATE)
    ret = audio_model(data.float())
    print('predict:', ret)
    if ret.argmax() == 0:
        cv2.imshow('result:', error)
    else:
        cv2.imshow('result:', no)
    cv2.waitKey(1)
    idx += 1
    save_name = os.path.join("../test_ret/7.21",
                             f"{time.localtime().tm_mon}_{time.localtime().tm_mday}_{time.localtime().tm_hour}_{time.localtime().tm_min}_{idx}_{str(ret[0][0].detach().numpy().round(2))}.wav")
    # if ret[0][1] < 0.5:
    #     soundfile.write(save_name, torch.stack([out_data1[0], out_data2[0]], dim=-1).numpy().astype(np.float32), RATE)
    # soundfile.write(save_name, torch.stack([out_data1[0], out_data2[0]], dim=-1).numpy().astype(np.float32), RATE)
    print(time.time() - start_)
print(ret)
stream1.stop_stream()  # 关闭流
stream1.close()
p1.terminate()

stream2.stop_stream()  # 关闭流
stream2.close()
p2.terminate()
