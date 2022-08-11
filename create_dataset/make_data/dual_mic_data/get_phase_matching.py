import librosa
import numpy as np


def get_offset(audio1, audio2, rate):
    sr = rate
    cal_len_ = int(0.3 * sr)
    start_ = int(0.8 * sr)
    stride_ = int(0.001 * sr)
    mfcc_fixed = librosa.feature.mfcc(y=audio1[start_:start_ + cal_len_], sr=sr)
    best_cor = 0
    best_offset = 0
    # 从start_往后偏移
    for offset in range(0, start_, stride_):
        mfcc_move = librosa.feature.mfcc(y=audio2[start_ + offset:start_ + cal_len_ + offset], sr=sr)
        cor = np.corrcoef(mfcc_fixed.reshape(-1), mfcc_move.reshape(-1))[0, 1]
        if cor > best_cor:
            best_cor = cor
            best_offset = start_ + offset
    # 从start_往前偏移
    for offset in range(0, start_, stride_):
        mfcc_move = librosa.feature.mfcc(y=audio2[start_ - offset:start_ + cal_len_ - offset], sr=sr)
        cor = np.corrcoef(mfcc_fixed.reshape(-1), mfcc_move.reshape(-1))[0, 1]
        if cor > best_cor:
            best_cor = cor
            best_offset = start_ - offset
    ret = best_offset - start_
    print("音频2索引移动：", ret)
    return ret


if "__main__" == __name__:
    s_show, sr_show = librosa.load("./pyaudio_output1.wav")
    s_show1, sr_show1 = librosa.load("./pyaudio_output2.wav")
    ret = get_offset(s_show, s_show1, sr_show)
    print(ret)
