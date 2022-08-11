import multiprocessing
import queue
import time
import wave
import librosa
import numpy as np
import pyaudio
import soundfile
from datetime import datetime
from tqdm import tqdm


CHUNK = int(22050*1.5)  # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 采样位数
CHANNELS = 1  # 单声道
RATE = 22050  # 采样频率


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


def record_audio(queue, device=0):
    """ 录音功能 """
    p = pyaudio.PyAudio()  # 实例化对象
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                    input_device_index=device,
                    )  # 打开流，传入响应参数
    print("start...", device)
    while True:
        data = stream.read(CHUNK)
        queue[0].put(data, block=False)
        if not queue[1].empty():
            print(device, 'out')
            stream.close()
            break


def play_audio(wave_path, chunk=1024):
    """
    This function is to play
    """
    print("Playing: ", wave_path)
    wf = wave.open(wave_path, 'rb')
    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()
    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # read data
    data = wf.readframes(chunk)
    # play stream (3)
    frames = []
    while len(data) > 0:
        data = wf.readframes(chunk)
        frames.append(data)
    for d in tqdm(frames):
        stream.write(d)
    # stop stream (4)
    stream.stop_stream()
    stream.close()
    # close PyAudio (5)
    p.terminate()


if __name__ == "__main__":
    p1 = pyaudio.PyAudio()  # 实例化对象
    stream1 = p1.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK,
                      input_device_index=1,######################################
                      )  # 打开流，传入响应参数
    print("start...", 1)

    p2 = pyaudio.PyAudio()  # 实例化对象
    stream2 = p2.open(format=FORMAT,
                      channels=CHANNELS,
                      rate=RATE,
                      input=True,
                      frames_per_buffer=CHUNK,
                      input_device_index=2,######################################
                      )  # 打开流，传入响应参数
    print("start...", 2)

    i = 0
    mic_data = []
    while True:
            # start_ = time.time()
            # stream1.start_stream()
            # stream2.start_stream()
            # print("s:", time.time() - start_)
            audio1 = stream1.read(CHUNK)
            audio2 = stream2.read(CHUNK)
            # start_ = time.time()
            # stream1.stop_stream()
            # stream2.stop_stream()
            # print("e:", time.time() - start_)
            out_data1 = np.frombuffer(audio1, dtype=np.int16)[None, :]
            out_data2 = np.frombuffer(audio2, dtype=np.int16)[None, :]
            out_data = np.stack([out_data2[0], out_data1[0]], axis=-1)
            # out_data = np.stack([np.ones(1024), np.zeros(1024)], axis=-1)
            file_name = rf"".join("./+2_channel_"+str(i)+datetime.now().strftime("%Y_%m_%d_%H_%M_%S")+".wav")
            soundfile.write(file_name, out_data, RATE, 'PCM_24')
            play_audio(wave_path=file_name)
            # ret = np.conc)atenate(mic_data) / 32768
            # ret = get_offset(ret[:, 0], ret[:, 1], RATE)
            i += 1
    p1.close(stream1)
    p2.close(stream2)

