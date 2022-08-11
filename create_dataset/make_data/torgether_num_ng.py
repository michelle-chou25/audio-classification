import os

import librosa
import numpy as np
import soundfile as sf

def run(ng_audioes_path, ground_audio, results_path, mix_factor=(1., 1.), rate=22050):
    ground_target, ground_tsr = librosa.load(ground_audio)
    if rate is not None:
        ground_target = librosa.resample(ground_target, orig_sr=ground_tsr, target_sr=rate)
        ground_tsr = rate

    files = os.listdir(ng_audioes_path)
    for idx, one in enumerate(files):
        ng_target, ng_tsr = librosa.load(os.path.join(ng_audioes_path, one))
        if rate is not None:
            ng_target = librosa.resample(ng_target, orig_sr=ng_tsr, target_sr=rate)
            ng_tsr = rate

        if len(ground_target) >= len(ng_target):
                rd = np.random.randint(0, len(ground_target)-len(ng_target))
                print("skip second:", rd/22050)
                ret_audio = ng_target*mix_factor[0] + ground_target[rd:rd+len(ng_target)]*mix_factor[1]
        else:
                ret_audio = ng_target[:len(ground_target)] * mix_factor[0] + ground_target * mix_factor[1]
        sf.write(os.path.join(results_path, str(idx)+'.wav'), ret_audio, ng_tsr, subtype='PCM_24')


# run('../../aduio_data', './环境声音_1.WAV', './results/2.23')
run(rf'C:\workspace\zhouhe\audio_relay\dataset\train\OK\special_env',
    rf'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\train\NG\2.23\ruo2.WAV',
    rf'./torgether_save',
    mix_factor=[1.3, 2]
    )
