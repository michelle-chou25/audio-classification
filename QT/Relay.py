import torch
import torchaudio

from const import AUDIO_RATE, AUDIO_INTERVAL,AUDIO_FRAMES_PER_BUFFER,\
    AI_INFERENCE_INTERVAL_DEFAULT,AI_SAVE_AUDIO_DEFAULT,AI_MELBINS,AI_NORM_MEAN,AI_NORM_STD,\
    AI_TARGET_LENGTH, AI_UINT16_MAX, UI_LOG_SHOW_MAX_LINE_NUM


class Relay:
    """
    data preprocessing
    """
    def __init__(self):
        self.melbins = AI_MELBINS
        self.norm_mean = AI_NORM_MEAN
        self.norm_std = AI_NORM_STD
        self.target_length = AI_TARGET_LENGTH

    def _wav2fbank(self, waveform, sr):
        waveform = waveform - waveform.mean()
        fbank0 = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                   window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                   channel=0,  ########0
                                                   frame_shift=10)
        fbank1 = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                   window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                   channel=1,  #########1
                                                   frame_shift=10)
        n_frames = fbank0.shape[0]
        p = self.target_length - n_frames #把fbank[0]的长度保持在172
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank0 = m(fbank0)
            fbank1 = m(fbank1)
        elif p < 0:
            fbank0 = fbank0[0:self.target_length, :]
            fbank1 = fbank1[0:self.target_length, :]
        fbank = torch.stack([fbank0, fbank1])
        return fbank, 0

    def preprocess(self, data, sr):
        fbank, mix_lambda = self._wav2fbank(data, sr)
        fbank = (fbank - self.norm_mean) / self.norm_std
        fbank = fbank[None, :] # 从[2, 172,384] 变为[1,2,172,384]
        return fbank