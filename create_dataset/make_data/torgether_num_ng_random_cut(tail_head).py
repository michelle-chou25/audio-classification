import os

import librosa
import numpy as np
import soundfile as sf

def run(ng_audioes_path, ground_audio, results_path, cut_factor=(0.20, 0.30), mix_factor=(1., 1.), rate=22050):
    """
    Args:
        ng_audioes_path:
        ground_audio:
        results_path:
        cut_factor: NG保留比例
        mix_factor:
        rate:

    Returns:

    """
    one_audio_data_len = int(rate*1.5)
    ground_target, ground_tsr = librosa.load(ground_audio)
    if rate is not None:
        ground_target = librosa.resample(ground_target, orig_sr=ground_tsr, target_sr=rate)
        ground_tsr = rate

    files = os.listdir(ng_audioes_path)
    for idx, one in enumerate(files):
        ng_target, ng_tsr = librosa.load(os.path.join(ng_audioes_path, one))

        # 随机裁剪数据的头尾部分
        reserve = np.random.uniform(cut_factor[0], cut_factor[1])
        rnd_length = int(reserve * rate) - 1
        if np.random.uniform() < 1.0: # 50%的概率保留头部
            flag = 'head'
            ng_target[rnd_length:] = 0.
        else:#保留尾部
            flag = 'tail'
            ng_target[:one_audio_data_len - rnd_length] = 0

        if rate is not None:
            ng_target = librosa.resample(ng_target, orig_sr=ng_tsr, target_sr=rate)
            ng_tsr = rate

        if len(ground_target) >= len(ng_target):
                rd = np.random.randint(0, len(ground_target)-len(ng_target))
                print("skip second:", rd/22050)
                ret_audio = ng_target*mix_factor[0] + ground_target[rd:rd+len(ng_target)]*mix_factor[1]
        else:
                ret_audio = ng_target[:len(ground_target)] * mix_factor[0] + ground_target * mix_factor[1]
        sf.write(os.path.join(results_path, f"{idx}_{flag}_r{str(round(reserve, 2))}__"+one), ret_audio, ng_tsr, subtype='PCM_24')


run(rf'./ng_split_data/qiang',
    rf'C:\workspace\zhouhe\audio_relay\audio_data\环境声音_1.WAV',
    rf'./torgether_num_ng_random_cut(tail_head)',
    mix_factor=[1.0, 0.70],
    cut_factor=(0.45, 0.6)
    )
