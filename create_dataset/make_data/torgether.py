import librosa
import numpy as np
import soundfile as sf

def run(ng_audio, ground_audio, results_path, mix_factor=(1., 1.), rate=22050):
    ground_target, ground_tsr = librosa.load(ground_audio)
    if rate is not None:
        ground_target = librosa.resample(ground_target, orig_sr=ground_tsr, target_sr=rate)
        ground_tsr = rate

    ng_target, ng_tsr = librosa.load(ng_audio)
    ng_target = ng_target[ng_tsr*180:]
    if rate is not None:
        ng_target = librosa.resample(ng_target, orig_sr=ng_tsr, target_sr=rate)
        ng_tsr = rate

    if len(ground_target) >= len(ng_target):
            rd = np.random.randint(0, len(ground_target)-len(ng_target))
            print("skip second:", rd/22050)
            ret_audio = ng_target*mix_factor[0] + ground_target[rd:rd+len(ng_target)]*mix_factor[1]
    else:
            ret_audio = ng_target[:len(ground_target)] * mix_factor[0] + ground_target * mix_factor[1]
    sf.write(results_path, ret_audio, ng_tsr, subtype='PCM_24')


# run('../../aduio_data', './环境声音_1.WAV', './results/2.23')
run(rf'../../audio_data/speech_noise.wav',
    rf'../../audio_data/12min_env.wav',
    rf'./torgether_save/12env_speech.wav',
    mix_factor=[0.2, 0.99]
    )
