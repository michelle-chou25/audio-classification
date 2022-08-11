import os
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from numpy.linalg import norm
import scipy.io.wavfile as wav
from datetime import datetime
import split_sound
from shutil import copyfile

# get the value of SNR
def SNR(x1, x2):
    # return 10*np.log(norm(x1) / norm(x2))
    p_x1 = np.sqrt(norm(x1))
    p_x2 = np.sqrt(norm(x2))
    return 10*np.log( (p_x1/len(x1))/(p_x2)/len(x2) )


def signal_by_db(x1,x2, snr, handle_method='cut'):
    """
    Generate noise mixed speech with given signal noise rate.
    Args:
        x1: speech, ndArray
        x2: noise, ndArray
        snr:  signal noise rate, int
        handle_method: generate the wav according the shorter one or the longer one. 'cut' or 'append'

    Returns:
        mix: mixed wav ndArray with

    """
    x1 = x1.astype(np.int32)
    x2 = x2.astype(np.int32)
    l1 = x1.shape[0]
    l2 = x2.shape[0]
    if l1 != l2:
        if handle_method == 'cut':
            ll = min(l1,l2)
            x1 = x1[:ll]
            x2 = x2[:ll]
        elif handle_method == 'append':
            ll = max(l1, l2)
            if l1 < ll:
                x1 = np.append(x1, x1[:ll-l1])
            if x2 < ll:
                x2 = np.append(x2, x2[:l2-ll])
                ll2 = min(x1.shape[0], x2.shape[0])
                x1 = x1[:ll2]
                x2 = x2[:ll2]
    # x2 = x2 / norm(x2) * norm(x1) / (10.0 ** (0.05 * snr))
    if x1.shape != x2.shape:
        x1 = x1[:, None]  # expan x1 from 1 dimension to 2 dimensions
    mix = x1+x2
    return mix


def get_wav_list(file_dir, wav_list):
    """
    Get all .wav files in a nested directory
    """
    temp_list = os.listdir(file_dir)
    for temp_list_each in temp_list:
        if os.path.isfile(file_dir+'/'+temp_list_each):
            temp_path = file_dir+'/'+temp_list_each
            if os.path.splitext(temp_path)[1] == '.wav' or os.path.splitext(temp_path)[1] == '.WAV':
                wav_list.append(temp_path)
            else:
                continue
        else:
            get_wav_list(file_dir+'/'+temp_list_each, wav_list)  # loop recursively
    return wav_list


    # Generate NG relay file with noise and different snr
def generate_mixed_audio(target_path, audios1, audios2, tag=""):

    for relay_file in random.sample(audios1,40):
        relay_data, sr = librosa.load(relay_file)
        for noise_file in random.sample(audios2, 50):
            noise_data, sr = librosa.load(noise_file)
            if np.ndim(relay_data) == np.ndim(noise_data):
                # mixed_data=relay_data*random.uniform(0.1,1) + noise_data*random.uniform(0.1,1)
                soundfile.write(os.path.join(target_path, datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + tag + '.wav'),
                                relay_data*random.uniform(0.1,1) + noise_data*random.uniform(0.01,0.1), sr)
                # for snr in snr_level:
                #     noisy_speech = signal_by_db(relay_data, noise_data, snr, 'cut')
                #     soundfile.write(os.path.join(target_path, datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + tag+'.wav'),
                #                  noisy_speech, sr)
            else:
                continue
    return  get_wav_list(target_path,wav_list=[])



if __name__ == '__main__':
    snr_level = [0, 10, 20]
    num_FFT = 512
    hop_size = 128
    relay_scale_bottom=1e-11
    relay_scale_top = 1e-9
    factory_scale_bottom=1e-11
    factory_scale_top = 1e-9
    speech_scale_bottom=1e-12
    speech_scale_top=1e-10

    # Split audio data to multiple 1.5s ones
    split_sound()
    input_file = r'../../audio_data/12min_env.wav'
    output_path = r'../splited_factory'
    split_sound.split_sound(input_file_path=input_file, output_path=output_path,
                            volume=1.0, interval_second=1.5, stride=0.7,rate=22050,
                            file_name=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    splitted_factory = get_wav_list(file_dir=output_path, wav_list=[])

    input_file = r'../../audio_data/speech_noise.wav'
    output_path = r'../splited_speech'
    split_sound.split_sound(input_file_path=input_file, output_path=output_path,
                            volume=1.0, interval_second=1.5, stride=0.7,rate=22050,
                            file_name=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    splitted_speech = get_wav_list(file_dir=output_path, wav_list=[])


    relay_li=get_wav_list(file_dir=r'../../audio_data/NG——pure Relay', wav_list=[])
    weak_relay, strong_relay=[],[]
    output_weak, output_strong=r'../splited_weak_relay', r'../splited_strong_relay'
    for file in relay_li:
        if ('ruo' or 'r') in os.path.basename(file):
            weak_relay.append(file)
        else:
            strong_relay.append(file)
    for file in weak_relay:
        split_sound.split_sound(input_file_path=file, output_path=r'../splited_weak_relay',
                                volume=1.0, interval_second=1.5, stride=0.7,rate=22050,
                                file_name=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    splitted_weak = get_wav_list(file_dir=output_weak, wav_list=[])
    for file in strong_relay:
        split_sound.split_sound(input_file_path=file, output_path=r'../splited_strong_relay',
                                volume=1.0, interval_second=1.5, stride=0.7,rate=22050,
                                file_name=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
    splitted_strong = get_wav_list(file_dir=output_strong, wav_list=[])

    splitted_strong=get_wav_list(file_dir=r'../splited_strong_relay', wav_list=[])
    r''
    splitted_weak=get_wav_list(file_dir=r'../splited_weak_relay',wav_list=[])
    splitted_factory=get_wav_list(file_dir= r'../splited_factory',wav_list=[])
    splitted_speech=get_wav_list(file_dir=r'../splited_speech',wav_list=[])


    factory_speech=generate_mixed_audio(r'../OK_SNR', splitted_factory, splitted_speech, 'factory_speech')
    strong_factoy=generate_mixed_audio(r'../NG_SNR/stron_factory',  splitted_strong, splitted_factory, 'strong_factoy')
    strong_speech=generate_mixed_audio(r'../NG_SNR/strong_speech', splitted_strong,splitted_speech, 'strong_speech')
    weak_factory=generate_mixed_audio(r'../NG_SNR/weak_factory',splitted_weak, splitted_factory, 'weak_factory')
    weak_speech=generate_mixed_audio(r'../NG_SNR/weak_speech', splitted_weak,splitted_speech, 'weak_speech')
    strong_factory_speech=strong_factory_speech=generate_mixed_audio(
        r'../NG_SNR/strong_factory_speech', strong_factoy,splitted_speech, 'strong_factory_speech')
    weak_factory_speech=generate_mixed_audio(
        r'../NG_SNR/weak_factory_speech',  weak_factory,splitted_speech,'weak_factory_speech')


    # Generate OK labels data
    # target_path = r'..\OK_snr_noise_audio'
    # speech_noise_list = get_wav_list(file_dir=r'..\splited_speech_noise', wav_list=[])
    # factory_noise_list = get_wav_list(file_dir=r'..\splited_factory_background_noise', wav_list=[])
    # for speech in speech_noise_list:
    #     sr, speech_data = wav.read(speech)
    #     for factory in factory_noise_list:
    #         sr, factory_data = wav.read(factory)
    #         mixed_noise = speech_data+factory_data
    #         soundfile.write(
    #             os.path.join(target_path,datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '.wav'),
    #             mixed_noise, sr
    #         )
