import csv
import os
import time
# import cv2
import librosa
import torchaudio
import torch
import ast
from models.Models_dual import MBNet, EffNetAttention, ResNetAttention
import argparse
from collections import OrderedDict
import numpy as np
import soundfile as sf
# CHUNK = 1024  # 每个缓冲区的帧数
# CHANNELS = 1  # 单声道
# RATE = 44100  # 采样频率
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
                                                   channel=0,########0
                                                   frame_shift=10)
        fbank1 = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                   window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                   channel=1,#########1
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

# ori_data, ori_sr = torchaudio.load(r'./q+r_1.WAV')
# ori_data, ori_sr = torchaudio.load(r'./q_1.WAV')
# ori_data, ori_sr = torchaudio.load(r'./环境声音_1.WAV')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--weight_path", type=str,
                    # default="../exp/3.14_file/models/audio_model_wa.pth",
                    # default="../exp/3.14_file/models/best_audio_model.pth",
                    default="../exp/3.14_file/models/audio_model.35.pth",
                    help="model file path")
parser.add_argument("--model", type=str, default="mbnet", help="audio model architecture",
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

if args.model == 'efficientnet':
    audio_model = EffNetAttention(label_dim=args.n_class, b=args.eff_b, pretrain=args.impretrain,
                                         head_num=args.att_head)
elif args.model == 'resnet':
    audio_model = ResNetAttention(label_dim=args.n_class, pretrain=args.impretrain)
elif args.model == 'mbnet':
    audio_model = MBNet(label_dim=args.n_class, pretrain=False)
audio_model.cuda()
val_audio_conf = {'num_mel_bins': 384, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean,
                  'std': args.dataset_std, 'noise': False}

# state_dictBA = torch.load("../exp/3.14/models/audio_model.26.pth", map_location='cpu')
# state_dictBA = torch.load("../exp/3.14_no_adddataset/models/audio_model.26.pth", map_location='cpu')
# state_dictBA = torch.load("../exp/7.19/models/best_audio_model.pth", map_location='cpu')
# state_dictBA = torch.load("../exp/7.19_add_0~220/models/audio_model.40.pth", map_location='cpu')
state_dictBA = torch.load("../exp/7.20_chao_ruo_error_env/models/audio_model.38.pth", map_location='cpu')
new_state_dictBA = OrderedDict()
for k, v in state_dictBA.items():
    if k[:7] == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    new_state_dictBA[name] = v
audio_model.load_state_dict(new_state_dictBA)

#torch.jit.load()

save_cpu_model = torch.jit.script(audio_model.cpu())
save_cpu_model.save('7.20_chao_ruo_error_env_mbv3_cpu.pth')

save_gpu_model = torch.jit.script(audio_model.cpu())
save_gpu_model.save('7.20_chao_ruo_error_env_mbv3_gpu.pth')
exit(0)

# print(audio_model)
audio_model.float().eval()  # .cuda()
# audio_model.half()

r = Relay(val_audio_conf, args.label_csv)

# ori_data, ori_sr = torchaudio.load(rf'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\val\NG\1\q+r_1.WAV')
# ori_data, ori_sr = torchaudio.load(r'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\val\NG\1\q_1.WAV')
# ori_data, ori_sr = torchaudio.load(r'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\val\NG\2\5min.wav')
# ori_data, ori_sr = torchaudio.load(r'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\val\OK\环境0\env0.WAV')
ori_data, ori_sr = torchaudio.load(rf'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\TestData\环境声音_1.WAV')
# ori_data, ori_sr = torchaudio.load(rf'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\TestData\12min_env.wav')
# ori_data, ori_sr = torchaudio.load(r'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\val\OK\12分钟\22_12min_env.wav.wav')


interval = 1.5
# if ori_data.shape[0] != 1 or ori_sr != 22050:
ori_data = torchaudio.transforms.Resample(ori_sr, 22050)(ori_data)
# ori_data = torchaudio.transforms.Resample(ori_sr, 22050)(ori_data)
ori_sr = 22050

all_second = len(ori_data) / ori_sr
interval_rate = int(ori_sr * 1.5)
if not os.path.exists(rf"../audio_need_test/audio_test"):
    os.mkdir(rf"../audio_need_test/audio_test")

#slice audio to clips, every clip lasts 1.5s
for idx, i in enumerate(range(0, len(ori_data[0]), int(1.5 * ori_sr))):
# for idx, i in enumerate(range(0, len(ori_data[0]), int(0.25 * ori_sr))):
    offset = int(0.0 * ori_sr)
    if i + interval_rate < len(ori_data[0]):
        start_ = time.time()
        out_data = ori_data[:, i + offset:i + offset + interval_rate]
        data = r.preprocess(out_data, ori_sr)
        #ret[0][0] is every clip's probability of "NG", ret[0][1] is the probability of "OK"
        ret = audio_model(data.cuda())
        # print('predict:', ret.argmax(), label.argmax())
        print('result:', ret.argmax(), ret[0])
        save_name = os.path.join(rf"../audio_need_test/audio_test", f"{idx}#_{str(ret[0][0].cpu().detach().numpy().round(2))}_#({idx * interval}~{idx * interval + interval}).wav")
        # librosa.output.write_wav(save_name, out_data.numpy().astype(np.float32), ori_sr)
        save_audio = out_data[0]
        sf.write(save_name, torch.stack([out_data[0], out_data[1]], dim=-1).numpy().astype(np.float32), ori_sr, subtype='PCM_24')
        # print(time.time() - start_)
# print(ret)
