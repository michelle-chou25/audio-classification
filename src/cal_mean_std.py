import argparse
import json
import os
import shutil
import torch
import torchaudio
import soundfile
# from ..dataloaders.audioset_dataset_dual_mic import AudiosetDataset
# from ..utilities import *
# from ..models.Models_dual import MBNet, EffNetAttention, ResNetAttention
# from ..traintest import train, validate
# import ast
# from torch.utils.data import WeightedRandomSampler
import numpy as np
# from generate_log import logger


# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# seed = 996
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

# file = r'..\dataset\dual_mic\'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def wav2fbank(filename):
    # filename = r'D:\workspace\audio_relay/dataset/dual_mic/train/NG/7.13/7_13_17_37_0_0.0.wav'
    waveform, sr = torchaudio.load(filename)
    fbank0 = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                              window_type='hanning', num_mel_bins=384, dither=0.0,
                                              channel=0,
                                              frame_shift=10)
    fbank1 = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                               window_type='hanning', num_mel_bins=384, dither=0.0,
                                               channel=1,
                                               frame_shift=10)
    fbank = torch.stack([fbank1, fbank0])
    return fbank


mean = 0.0
std = 0.0
#  Get mean
dataset_json_file=r'D:\workspace\audio_relay\wz_relay_dual\wz_relay_val.json'

with open(dataset_json_file, 'r') as fp:
    data_json = json.load(fp)
data = data_json['data']
all_data = []
for item in data:
    fbank=wav2fbank(item['wav'])
    all_data.append(fbank)
sum=np.stack(all_data)
print(sum.mean(), sum.std())
print(all_data.__len__())


