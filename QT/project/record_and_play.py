import numpy as np
import pyaudio
from relay_predict_app import LOG_RED


def get_usb_mic(logger):
    p = pyaudio.PyAudio()
    mic = []
    info = p.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        device = p.get_device_info_by_host_api_device_index(0, i)
        if device.get('maxInputChannels') > 0 and device.get('name').__contains__('USB'):
            logger.put(f"USB mic id {i}  -  {p.get_device_info_by_host_api_device_index(0, i).get('name')}")
            mic.append(i)
    return mic


class Recorder:
    def __init__(self, mic_num, queue, keepalive_queue, rate=22050, channel=1, format=pyaudio.paInt16,
                 num_frames=1024, log_queue=None):
        self.mic_num = mic_num
        self.queue = queue
        self.num_frames = num_frames
        self.rate = rate
        self.channel = channel
        self.format = format
        self.log_queue = log_queue
        self.keepalive_queue = keepalive_queue

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        try:
            mic_ids = get_usb_mic(self.log_queue)
            if len(mic_ids) < self.mic_num:
                self.log_queue.put(LOG_RED(f'USB mic 数量（不小于2个）错误,检测到{len(mic_ids)}个。请插入USB mic后再按开始按钮。'))
                raise ValueError('ERROR:USB mic error!')

            p1 = pyaudio.PyAudio()  # 实例化对象
            stream1 = p1.open(format=self.format,
                              channels=self.channel,
                              rate=self.rate,
                              input=True,
                              frames_per_buffer=self.num_frames,
                              input_device_index=mic_ids[0],
                              )  # 打开流，传入响应参数
            p2 = pyaudio.PyAudio()  # 实例化对象
            stream2 = p2.open(format=self.format,
                              channels=self.channel,
                              rate=self.rate,
                              input=True,
                              frames_per_buffer=self.num_frames,
                              input_device_index=mic_ids[1],
                              )  # 打开流，传入响应参数

            # i = 0
            # mic_data = []
            while True:
                # start_ = time.time()
                # stream1.start_stream()
                # stream2.start_stream()
                # print("s:", time.time() - start_)
                audio1 = stream1.read(self.num_frames)
                audio2 = stream2.read(self.num_frames)
                # start_ = time.time()
                # stream1.stop_stream()
                # stream2.stop_stream()
                # print("e:", time.time() - start_)
                out_data1 = np.frombuffer(audio1, dtype=np.int16)[None, :]
                out_data2 = np.frombuffer(audio2, dtype=np.int16)[None, :]

                out_data = np.stack([out_data2[0], out_data1[0]], axis=-1)
                if self.queue.empty():
                    self.queue.put(out_data)
                else:
                    self.log_queue.put('Recorder queue full!!!!!!!!')

            p1.close(stream1)
            p2.close(stream2)
        except Exception as e:
            self.keepalive_queue.put('ERROR#Recorder:'+str(e))


class Player:
    def __init__(self, queue, rate=22050, channel=2, format=pyaudio.paInt16, num_frames=1024):
        self.queue = queue
        self.num_frames = num_frames
        self.rate = rate
        self.channel = channel
        self.format = format
        super(Player, self).__init__()

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(channels=self.channel,
                        rate=self.rate,
                        output=True,
                        format=self.format,
                        )
        while True:
            ret = self.queue.get()
            stream.write(ret, num_frames=self.num_frames)
        stream.stop_stream()
        stream.close()
        p.terminate()
