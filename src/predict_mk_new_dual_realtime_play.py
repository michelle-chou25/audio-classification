import csv
import multiprocessing
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
from utilities.record_and_play import Player, Recorder

if "__main__" == __name__:
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
        def __init__(self):
            self.melbins = 384
            self.norm_mean = -4.6476
            self.norm_std = 4.5699
            self.target_length = 172

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
            p = self.target_length - n_frames

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
            """
            returns: image, audio, nframes
            where image is a FloatTensor of size (3, H, W)
            audio is a FloatTensor of size (N_freq, N_frames) for spectrogram, or (N_frames) for waveform
            nframes is an integer
            """
            fbank, mix_lambda = self._wav2fbank(data, sr)
            fbank = (fbank - self.norm_mean) / (self.norm_std)
            fbank = fbank[None, :]
            return fbank


    # audio_model = torch.jit.load('./7.19(add)_mbv3_cpu.pth')
    # audio_model = torch.jit.load('./7.19_add_0~220_mbv3_cpu.pth')###########ok
    # audio_model = torch.jit.load('./7.20_chao_ruo_error_env_mbv3_cpu.pth')
    # audio_model = torch.jit.load('./7.19_add_0~220_ruo_mbv3_cpu.pth')
    audio_model = torch.jit.load('./model_files/7.20_chao_ruo_error_env_mbv3_cpu.pth')
    audio_model.float().eval()  # .cuda()

    data = np.zeros([1, 172, 384], dtype=np.float32)
    data = torch.from_numpy(data)  # .cuda()
    r = Relay()

    # CV show 框
    no = np.full([300, 300, 3], [0, 255, 0], dtype=np.uint8)
    error = np.full([300, 300, 3], [0, 0, 255], dtype=np.uint8)
    cv2.imshow('result:', no)

    RATE = 22050
    interval = 1.5
    num_frames = 1323  # 22050*1.5/25

    record_audio_q = multiprocessing.Queue(maxsize=1000)
    play_audio_q = multiprocessing.Queue(maxsize=1000)
    record = Recorder([1, 3], record_audio_q, num_frames=num_frames)
    play = Player(play_audio_q, num_frames=num_frames)

    record1 = multiprocessing.Process(target=record.run)
    record1.daemon = True
    record1.start()

    play1 = multiprocessing.Process(target=play.run)
    play1.daemon = True
    play1.start()

    data_frame_num = int(RATE * interval)
    inference_interval_frame = int(300 / (1000*interval/(RATE * interval / num_frames)) * num_frames)  # 300ms(必须是60的倍数) 推理一次
    current_frame_num = 0
    num_id = 0
    one_input = np.zeros([data_frame_num, 2])
    while True:
        audio_one = record_audio_q.get()
        current_frame_num += num_frames
        one_input = np.concatenate([one_input[num_frames:, :], audio_one])
        num_id += 1
        play_audio_q.put(audio_one, block=False)
        if current_frame_num == inference_interval_frame:  # 如果满inference_interval_frame进行一次推理
            current_frame_num = 0
            data_org = one_input / 32768
            data = torch.from_numpy(data_org)
            data = torch.transpose(data, 1, 0)
            data = r.preprocess(data, RATE)
            ret = audio_model(data.float())
            if ret.argmax() == 0:
                cv2.imshow('result:', error)
            else:
                cv2.imshow('result:', no)
            cv2.waitKey(1)
            # save_name = os.path.join("../test_ret/8.10",
            #                          f"{time.localtime().tm_mon}_{time.localtime().tm_mday}_{time.localtime().tm_hour}_{time.localtime().tm_min}_{num_id}_{str(ret[0][0].detach().numpy().round(2))}.wav")
            # if ret[0][1] < 0.5:
            #     soundfile.write(save_name, data_org.astype(np.float32), RATE)
            # soundfile.write(save_name, data_org.astype(np.float32), RATE)

    # stream1.stop_stream()  # 关闭流
    # stream1.close()
    # p1.terminate()
    #
    # stream2.stop_stream()  # 关闭流
    # stream2.close()
    # p2.terminate()
