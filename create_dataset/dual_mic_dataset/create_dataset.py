import os
import librosa
import numpy as np
import soundfile


def offset(data, offset_factor=(5, 200)):
    r = np.random.randint(offset_factor[0], offset_factor[1])

    if np.random.uniform() < 0.5:
        ret = np.concatenate([data[r:], data[:r]])
    else:
        ret = np.concatenate([data[-r:], data[:-r]])
    return ret


def ng():
    env_path = './org/env'
    env_folders = os.listdir(env_path)
    env_files = []
    for one_folder in env_folders:
        one_folder_files = os.listdir(os.path.join(env_path, one_folder))
        env_files.extend(map(lambda x: os.path.join(env_path, one_folder, x), one_folder_files))
    print(len(env_files))
    relay_path = './org/relay'
    relay_folders = os.listdir(relay_path)
    relay_files = []
    for one_folder in relay_folders:
        one_folder_files = os.listdir(os.path.join(relay_path, one_folder))
        relay_files.extend(map(lambda x: os.path.join(relay_path, one_folder, x), one_folder_files))
    print(len(relay_files))

    all_env = len(env_files)
    for i in range(all_env):
        audio_relay_path = relay_files[i % len(relay_files)]
        audio_env_path = env_files[i]
        relay_data, sr_r = librosa.load(audio_relay_path)
        env_data, sr_e = librosa.load(audio_env_path)

        relay_data_f = np.random.uniform(0.7, 1.1)
        env_data_f = np.random.uniform(0.5, 1.0)

        channel1 = relay_data * relay_data_f + env_data * env_data_f

        env_data_offset = offset(env_data)
        env_data_offset_f = np.random.uniform(0.8, 1.0)
        channel2 = env_data_offset * env_data_f * env_data_offset_f

        file_name = os.path.split(audio_relay_path)[-1]
        soundfile.write(
            rf"./result/NG/{np.round(env_data_offset_f, 2)}_{np.round(relay_data_f, 2)}_{np.round(env_data_f, 2)}_{file_name}",
            np.stack([channel1, channel2], axis=-1), sr_r, 'PCM_24')
        print(i, '/', len(env_files))


def ok():
    env_path = './org/env'
    env_folders = os.listdir(env_path)
    env_files = []
    for one_folder in env_folders:
        one_folder_files = os.listdir(os.path.join(env_path, one_folder))
        env_files.extend(map(lambda x: os.path.join(env_path, one_folder, x), one_folder_files))
    print(len(env_files))
    relay_path = './org/relay'
    relay_folders = os.listdir(relay_path)
    relay_files = []
    for one_folder in relay_folders:
        one_folder_files = os.listdir(os.path.join(relay_path, one_folder))
        relay_files.extend(map(lambda x: os.path.join(relay_path, one_folder, x), one_folder_files))
    print(len(relay_files))

    all_env = len(env_files)
    for i in range(all_env):
        audio_relay_path = relay_files[i % len(relay_files)]
        audio_env_path = env_files[i]
        relay_data, sr_r = librosa.load(audio_relay_path)
        env_data, sr_e = librosa.load(audio_env_path)

        # relay_data_f = np.random.uniform(0.7, 1.1)
        # env_data_f = np.random.uniform(0.5, 1.0)
        #
        # channel1 = relay_data * relay_data_f + env_data * env_data_f

        env_data_offset = offset(env_data)
        env_data_offset_f = np.random.uniform(0.8, 1.2)
        channel2 = env_data_offset * env_data_offset_f

        file_name = os.path.split(audio_relay_path)[-1]
        soundfile.write(
            rf"./result/OK/{np.round(env_data_offset_f, 2)}_{file_name}",
            np.stack([env_data, channel2], axis=-1), sr_r, 'PCM_24')
        print(i, '/', len(env_files))

ok()