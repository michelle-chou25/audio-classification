import multiprocessing
import numpy as np
import pyaudio
import soundfile

CHUNK = int(1024)  # 每个缓冲区的帧数
FORMAT = pyaudio.paInt16  # 采样位数
CHANNELS = 1  # 单声道
RATE = 22050  # 采样频率

# p = pyaudio.PyAudio()
# for i in range(p.get_device_count()):
#     print(p.get_device_info_by_index(i))
# exit()

class Recorder:
    def __init__(self, mic_ids, queue):
        self.mic_ids = mic_ids
        self.queue = queue

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        p1 = pyaudio.PyAudio()  # 实例化对象
        stream1 = p1.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK,
                          input_device_index=self.mic_ids[0],
                          )  # 打开流，传入响应参数
        print("start...", 1)
        p2 = pyaudio.PyAudio()  # 实例化对象
        stream2 = p2.open(format=FORMAT,
                          channels=CHANNELS,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK,
                          input_device_index=self.mic_ids[1],
                          )  # 打开流，传入响应参数
        print("start...", 2)

        # i = 0
        # mic_data = []
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
            self.queue.put(out_data)

            # out_data = np.stack([np.ones(1024), np.zeros(1024)], axis=-1)
            # file_name = rf"".join("./+2_channel_" + str(i) + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".wav")
            # soundfile.write(file_name, out_data, RATE, 'PCM_24')
            # self.result.append(file_name)
            # i += 1
        p1.close(stream1)
        p2.close(stream2)


class Player:
    def __init__(self, queue):
        self.queue = queue
        super(Player, self).__init__()

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(channels=2,
                        rate=22050,
                        output=True,
                        format=FORMAT,
                        )
        while True:
            ret = self.queue.get()
            stream.write(ret, num_frames=CHUNK)
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    q = multiprocessing.Queue(maxsize=1000)
    record = Recorder([1, 3], q)
    play = Player(q)

    p1 = multiprocessing.Process(target=play.run)
    p1.daemon = True
    p1.start()

    p2 = multiprocessing.Process(target=record.run)
    p2.daemon = True
    p2.start()

    p1.join()
    p2.join()

