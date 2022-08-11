import csv
import os
import time
from torch.utils.mobile_optimizer import optimize_for_mobile
import librosa
import torchaudio

import dataloaders
import torch
import ast
import models
import argparse
from collections import OrderedDict
import numpy as np
from torch import Tensor

# from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
_eff_b = 2
_target_length = 300
def my_rfft():
    # array([3.445+0.j   , 0.02 -0.135j, 0.115+0.j   ])
    np.fft.rfft([0.9, 0.9, 0.88, 0.765])
my_rfft()
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--weight_path", type=str,
                    default="/home/zhouhe/workspace/acv/acv/audio/psla-main/exp/2/models/best_audio_model.pth",
                    help="model file path")
parser.add_argument("--model", type=str, default="efficientnet", help="audio model architecture",
                    choices=["efficientnet", "resnet", "mbnet"])
parser.add_argument("--eff_b", type=int, default=_eff_b,
                    help="which efficientnet to use, the larger number, the more complex")
parser.add_argument("--n_class", type=int, default=2, help="number of classes")
parser.add_argument('--impretrain', help='if use imagenet pretrained CNNs', type=ast.literal_eval, default='True')
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
    audio_model = models.EffNetAttention(label_dim=args.n_class, b=args.eff_b, pretrain=args.impretrain,
                                         head_num=args.att_head, export=True)
elif args.model == 'resnet':
    audio_model = models.ResNetAttention(label_dim=args.n_class, pretrain=args.impretrain)
# elif args.model == 'mbnet':
#     audio_model = models.MBNet(label_dim=args.n_class, pretrain=args.effpretrain)
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean,
                  'std': args.dataset_std, 'noise': False}


class SpeechRecognizer(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.labels = ["N", "O"]

    def forward(self, waveforms: Tensor):
        """Given a single channel speech data, return transcription.

        Args:
            waveforms (Tensor): Speech tensor. Shape `[1, num_frames]`.

        Returns:
            str: The resulting transcript
        """
        logits = self.model(waveforms)  # [batch, num_seq, num_label]
        best_path = torch.argmax(logits[0])  # [num_seq,]
        # char = self.labels[best_path]
        return best_path



audio_model.eval()
audio_model = SpeechRecognizer(audio_model)
example_inputs = torch.rand(1, 66150, dtype=torch.float32)
audio_traced = torch.jit.trace(audio_model, example_inputs=example_inputs)
# audio_traced.save("./jit_save/mmmy.pt")
traced_script_module_optimized = optimize_for_mobile(audio_traced)
traced_script_module_optimized._save_for_lite_interpreter("./jit_save/mmmy.ptl")
