import multiprocessing
import Inference
from project.record_and_play import Recorder, Player
from const import AUDIO_RATE, AUDIO_INTERVAL,AUDIO_FRAMES_PER_BUFFER,\
    AI_INFERENCE_INTERVAL_DEFAULT,AI_SAVE_AUDIO_DEFAULT,AI_MELBINS,AI_NORM_MEAN,AI_NORM_STD,\
    AI_TARGET_LENGTH,AI_UINT16_MAX, UI_LOG_SHOW_MAX_LINE_NUM


class Processes:
    def __init__(self, model_path, log_queue, keepalive_queue, play_flag=False, logger_ng_flag_set=False,
                 logger_ok_flag_set=False,
                 inference_interval_set=AI_INFERENCE_INTERVAL_DEFAULT, save_audio_set=AI_SAVE_AUDIO_DEFAULT):
        """

        :param model_path: 模型路径
        :param log_queue: log队列
        :param keepalive_queue: recorder player keep alive队列
        :param play_flag: 播音 True/False
        :param logger_ng_flag_set: 是否显示NG
        :param logger_ok_flag_set: 是否显示OK
        :param inference_interval_set: 检测间隔使劲按
        :param save_audio_set:
        """
        self.model_path = model_path
        self.log_queue = log_queue
        self.rate = AUDIO_RATE
        self.interval = AUDIO_INTERVAL
        # 22050*1.5/25, 1323指的是1500ms分25份 每份60ms，60ms中占了1323个振幅信息
        # 音频采集frames_per_buffer设置为1323， 也就是1323个采样点
        self.num_frames = AUDIO_FRAMES_PER_BUFFER
        self.inference_interval = inference_interval_set  # 300ms(必须是60（其实是60ms）的倍数) 推理一次
        self.play_flag = play_flag
        self.logger_ng_flag_set = logger_ng_flag_set
        self.logger_ok_flag_set = logger_ok_flag_set
        self.save_audio_set = save_audio_set  # 预测结果>阈值,则保存音频文件
        self.keepalive_queue = keepalive_queue
        self.record_audio_q = None
        self.play_audio_q = None

    def run(self):
        self.record_audio_q = multiprocessing.Queue(maxsize=10)
        self.play_audio_q = multiprocessing.Queue(maxsize=10)
        play = Player(self.play_audio_q, rate=self.rate, channel=2, num_frames=self.num_frames)
        inference = Inference(self.model_path, self.record_audio_q, self.play_audio_q, self.rate, self.interval,
                              self.inference_interval, self.num_frames, self.log_queue, self.play_flag,
                              self.logger_ng_flag_set, self.logger_ok_flag_set, self.save_audio_set)
        try:
            record = Recorder(2, self.record_audio_q, rate=self.rate, channel=1, num_frames=self.num_frames,
                              log_queue=self.log_queue, keepalive_queue=self.keepalive_queue)
        except:
            return False, (None, None, None)

        record1 = multiprocessing.Process(target=record.run)  # target run方法调用的可调用对象是record.run()
        record1.name = '录音进程'
        # 为True，record1为后台运行的守护进程，当record1的父进程终止时，record1也随之终止，并且设定为True后，
        # record1不能创建自己的新进程，必须在record1.start()之前设置
        record1.daemon = True
        record1.start()  # 生成进程

        infer1 = multiprocessing.Process(target=inference.run)
        infer1.name = '检测进程'
        infer1.daemon = True
        infer1.start()

        if self.play_flag:
            play1 = multiprocessing.Process(target=play.run)
            play1.name = '播音进程'
            play1.daemon = True
            play1.start()
            return True, (record1, play1, infer1)
        else:
            return True, (record1, None, infer1)