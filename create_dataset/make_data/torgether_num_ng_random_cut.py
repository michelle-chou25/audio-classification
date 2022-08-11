import os

import librosa
import numpy as np
import soundfile as sf

def run(ng_audioes_path, ground_audio, results_path, cut_factor=(0.25, 0.75), mix_factor=(1., 1.), rate=22050):
    one_audio_data_len = int(rate*1.5)
    ground_target, ground_tsr = librosa.load(ground_audio)
    if rate is not None:
        ground_target = librosa.resample(ground_target, orig_sr=ground_tsr, target_sr=rate)
        ground_tsr = rate

    files = os.listdir(ng_audioes_path)
    for idx, one in enumerate(files):
        ng_target, ng_tsr = librosa.load(os.path.join(ng_audioes_path, one))

        # 随机裁剪数据
        reserve = np.random.uniform(cut_factor[0], cut_factor[1])
        rnd_length = int(reserve*rate) - 1
        skip = np.random.uniform(0, 1 - reserve)
        rnd_start = int(skip*rate)
        ng_target[rnd_start:rnd_start + rnd_length] = 0.

        if rate is not None:
            ng_target = librosa.resample(ng_target, orig_sr=ng_tsr, target_sr=rate)
            ng_tsr = rate

        if len(ground_target) >= len(ng_target):
                rd = np.random.randint(0, len(ground_target)-len(ng_target))
                print("skip second:", rd/22050)
                ret_audio = ng_target*mix_factor[0] + ground_target[rd:rd+len(ng_target)]*mix_factor[1]
        else:
                ret_audio = ng_target[:len(ground_target)] * mix_factor[0] + ground_target * mix_factor[1]
        sf.write(os.path.join(results_path, f"{idx}_s{str(round(skip, 2))}_r{str(round(reserve, 2))}__"+one), ret_audio, ng_tsr, subtype='PCM_24')


run(rf'./ng_split_data/qiang',
    rf'E:\workspace\audio_relay\create_dataset\make_data\torgether_save\12env_speech0.2.wav',
    rf'./torgether_num_ng_random_cut',
    mix_factor=[1.1, 0.70]
    )
