import multiprocessing
import os
import sys
import time
import pyaudio
import soundfile
import torchaudio
import torch
import numpy as np
from PySide6.QtGui import QIcon
from record_and_play import Player, Recorder
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import QFileDialog, QRadioButton, QWidget, QComboBox

AUDIO_RATE = 22050
AUDIO_INTERVAL = 1.5  # 采样时间1.5秒
# 22050*（1500ms/25）, 1323指的是1500ms分25份 每份60ms，按照22050的采样频率（也就是每秒采样22050个振幅）
# 60ms中占了1323个振幅信息（采样点）
AUDIO_FRAMES_PER_BUFFER = 1323

AI_INFERENCE_INTERVAL_DEFAULT = 300  # 采样间隔
AI_SAVE_AUDIO_DEFAULT = 0.0  # 保存阈值
AI_MELBINS = 384
AI_NORM_MEAN = -4.6476
AI_NORM_STD = 4.5699
AI_TARGET_LENGTH = 172
AI_UINT16_MAX = 32768

UI_LOG_SHOW_MAX_LINE_NUM = 1000


def LOG_RED(string):
    return "<font color=red>" + string + "</font>" "<font color=black> </font>"


def LOG_GREEN(string):
    return "<font color=green>" + string + "</font>" "<font color=black> </font>"


class Relay:
    def __init__(self):
        self.melbins = AI_MELBINS
        self.norm_mean = AI_NORM_MEAN
        self.norm_std = AI_NORM_STD
        self.target_length = AI_TARGET_LENGTH

    def _wav2fbank(self, waveform, sr):
        waveform = waveform - waveform.mean()
        fbank0 = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                   window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                   channel=0,  ########0
                                                   frame_shift=10)
        fbank1 = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                   window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                   channel=1,  #########1
                                                   frame_shift=10)
        n_frames = fbank0.shape[0]
        p = self.target_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank0 = m(fbank0)
            fbank1 = m(fbank1)
        elif p < 0:
            fbank0 = fbank0[0:self.target_length, :]
            fbank1 = fbank1[0:self.target_length, :]
        fbank = torch.stack([fbank0, fbank1])
        return fbank, 0

    def preprocess(self, data, sr):
        fbank, mix_lambda = self._wav2fbank(data, sr)
        fbank = (fbank - self.norm_mean) / self.norm_std
        fbank = fbank[None, :]  # 从[2, 172,384] 变为[1,2,172,384]
        return fbank


class Inference:
    """

    :param model_path: 模型路径
    :param log_queue: log队列
    :param keepalive_queue: recorder player keep alive队列
    :param play_flag: 播音 True/False
    :param logger_ng_flag_set: 是否显示NG
    :param logger_ok_flag_set: 是否显示OK
    :param inference_interval_set: 检测间隔时间
    :param save_audio_set:
    """
    def __init__(self, model_path, record_audio_q, play_audio_q, rate, interval, inference_interval, num_frames,
                 log_queue, play_flag=False, logger_ng_flag_set=False, logger_ok_flag_set=False,
                 save_audio_set=AI_SAVE_AUDIO_DEFAULT):
        self.model_path = model_path
        self.rate = rate
        self.interval = interval
        self.inference_interval = inference_interval
        self.num_frames = num_frames
        self.record_audio_q = record_audio_q
        self.play_audio_q = play_audio_q
        self.play_flag = play_flag
        self.logger_ng_flag_set = logger_ng_flag_set
        self.logger_ok_flag_set = logger_ok_flag_set
        self.save_audio_set = save_audio_set
        self.log_queue = log_queue

    def run(self):
        r = Relay()
        # Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
        audio_model = torch.jit.load(self.model_path)
        audio_model.float().eval()

        data_frame_num = int(self.rate * self.interval)  # 22050的采样频率，采样1.5秒
        # 1323* 300/(1500ms/(33075/1323))=3315,  33075/1323=60ms
        inference_interval_frame = int(
            self.inference_interval / (1000 * self.interval / (
                    self.rate * self.interval / self.num_frames)) * self.num_frames)  # 300ms(必须是60的倍数) 推理一次
        current_frame_num = 0
        num_id = 0
        save_id = 0
        # [33075,2]
        one_input = np.zeros([data_frame_num, 2])
        self.log_queue.put(f"检测程序开始运行...")
        while True:
            # 获取multiprocessing.Queue()第一个数据
            audio_one = self.record_audio_q.get()  # [1323,2] 1323个采样点，2个通道
            current_frame_num += self.num_frames
            # one_input[self.num_frames:, :].shape=[31275,2], audio_one.shape=[1323,2], concatenate的结果是
            # 把one_input的前[1323,2]个元素替换为audio_one的内容
            one_input = np.concatenate([one_input[self.num_frames:, :], audio_one])
            num_id += 1
            if self.play_flag:
                self.play_audio_q.put(audio_one, block=False)  # 在播放queue里加入 audio_one这么多 buffer待播放
            if current_frame_num == inference_interval_frame:  # 如果满inference_interval_frame（3315， 也就是5次self.num_frames）进行一次推理
                current_frame_num = 0
                data_org = one_input / AI_UINT16_MAX  # 把data_org 变成[-1,1]的浮点数
                data = torch.from_numpy(data_org)
                data = torch.transpose(data, 1, 0)
                data = r.preprocess(data, self.rate)
                ret = audio_model(data.float())
                ok_p = ret[0][1].detach().numpy().round(2)   # tensor.detach()Returns a new Tensor, detached from the current graph. # The result will never require gradient, 转为numpy， 保留两位小数
                if ret.argmax() == 0: #最大值的索引为0
                    if self.logger_ng_flag_set:
                        self.log_queue.put(LOG_RED(f"NG:{str(ret[0][0].detach().numpy().round(2))}"))
                else:
                    if self.logger_ok_flag_set:
                        self.log_queue.put(f"OK:{str(ok_p)}")
                if ok_p <= self.save_audio_set:  # 12ms #ok的概率小于给定的阈值，就保存音频文件（可认为低于这个阈值的ok是需要检查的）
                    save_id += 1
                    save_name = os.path.join("./save_audio",
                                             f"{time.localtime().tm_mon}_{time.localtime().tm_mday}"
                                             f"_{time.localtime().tm_hour}_{time.localtime().tm_min}"
                                             f"_{save_id}_{str(ok_p)}.wav")
                    soundfile.write(save_name, data_org, AUDIO_RATE)


class RelayProject:
    def __init__(self, model_path, log_queue, keepalive_queue, play_flag=False, logger_ng_flag_set=False,
                 logger_ok_flag_set=False,
                 inference_interval_set=AI_INFERENCE_INTERVAL_DEFAULT, save_audio_set=AI_SAVE_AUDIO_DEFAULT):
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


class MyWidget(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.joins = []
        self.model_path_set = None
        self.player_flag_set = False
        self.logger_ng_flag_set = False
        self.logger_ok_flag_set = False
        self.inference_interval_set = AI_INFERENCE_INTERVAL_DEFAULT
        self.save_audio_set = AI_SAVE_AUDIO_DEFAULT
        self.log_queue = multiprocessing.Queue(maxsize=10)
        self.keepalive_queue = multiprocessing.Queue(maxsize=10)
        self.setWindowTitle("继电器尾音检测V1.0")
        self.play_button = QRadioButton('同步播放开关')
        self.play_button.setAutoExclusive(False)
        self.logger_ng_button = QRadioButton('显示NG')
        self.logger_ng_button.setAutoExclusive(False)
        self.logger_ok_button = QRadioButton('显示OK')
        self.logger_ok_button.setAutoExclusive(False)
        self.play_button.setChecked(self.player_flag_set)
        self.logger_ng_button.setChecked(self.logger_ng_flag_set)
        self.logger_ok_button.setChecked(self.logger_ok_flag_set)

        self.inference_interval_cb = QComboBox(self)
        self.inference_interval_cb_label = QtWidgets.QLabel('检测间隔(ms):', alignment=QtCore.Qt.AlignLeft)
        self.inference_interval_cb.addItems(['300', '420', '540', '660', '780', '900', '1140', '1500'])

        self.save_audio_cb = QComboBox(self)
        self.save_audio_cb_label = QtWidgets.QLabel('音频保存:', alignment=QtCore.Qt.AlignLeft)
        self.save_audio_cb.addItems(['不保存(0.0)', 'NG(0.3)', 'NG(0.4)', 'NG(0.5)', 'NG&OK(0.6)',
                                     'NG&OK(0.7)', 'NG&OK(0.8)', 'NG&OK(0.9)', '全保存(1.0)'])

        self.button_pth = QtWidgets.QPushButton("更改模型文件（.pth）")
        self.button_run = QtWidgets.QPushButton("开始")
        self.log = QtWidgets.QTextBrowser(self)
        self.log.document().setMaximumBlockCount(UI_LOG_SHOW_MAX_LINE_NUM)

        self.label = QtWidgets.QLabel('设置:', alignment=QtCore.Qt.AlignLeft)
        self.log.setText(LOG_GREEN("注：模型文件如果不选择将采用默认模型文件\n"))

        self.vboxlayout_global = QtWidgets.QVBoxLayout(self)

        self.gridlayout = QtWidgets.QGridLayout(self)
        self.gridlayout.addWidget(self.label, 1, 1)
        self.gridlayout.addWidget(self.play_button, 2, 1)
        self.gridlayout.addWidget(self.logger_ng_button, 2, 2)
        self.gridlayout.addWidget(self.logger_ok_button, 2, 3)
        self.gridlayout.addWidget(self.inference_interval_cb_label, 3, 1)
        self.gridlayout.addWidget(self.inference_interval_cb, 4, 1)
        self.gridlayout.addWidget(self.save_audio_cb_label, 3, 2)
        self.gridlayout.addWidget(self.save_audio_cb, 4, 2)
        self.vboxlayout = QtWidgets.QVBoxLayout(self)
        self.vboxlayout.addWidget(self.log)
        self.vboxlayout.addWidget(self.button_pth)
        self.vboxlayout.addWidget(self.button_run)

        self.q1 = QWidget()
        self.q2 = QWidget()
        self.q1.setLayout(self.gridlayout)
        self.q2.setLayout(self.vboxlayout)
        self.vboxlayout_global.addWidget(self.q1)
        self.vboxlayout_global.addWidget(self.q2)
        self.setLayout(self.vboxlayout_global)

        self.button_pth.clicked.connect(self.model_file_path)
        self.button_run.clicked.connect(self.run)
        self.play_button.toggled.connect(self.player_flag)
        self.logger_ng_button.toggled.connect(self.logger_ng_flag)
        self.logger_ok_button.toggled.connect(self.logger_ok_flag)
        self.inference_interval_cb.currentTextChanged.connect(self.inference_interval_cb_chang_value)
        self.save_audio_cb.currentTextChanged.connect(self.save_audio_cb_chang_value)

    def save_audio_cb_chang_value(self, v):
        self.save_audio_set = float(v.split('(')[-1][:-1])   # e.g. v=NG&OK(0.7)->0.7, 预测结果大于0.7的ok的尾音才被保存
        self.log_queue.put(f'{self.save_audio_cb_label.text()}{v}, 文件保存在工程目录下的save_audio文件夹里。')

    def inference_interval_cb_chang_value(self, v):
        self.inference_interval_set = int(v)
        self.log_queue.put(f'{self.inference_interval_cb_label.text()}{v}')

    def logger_ok_flag(self, checked):
        self.logger_ok_flag_set = checked
        if checked:
            self.log_queue.put(f'{self.logger_ok_button.text()}：已开启')
        else:
            self.log_queue.put(f'{self.logger_ok_button.text()}：已关闭')

    def logger_ng_flag(self, checked):
        self.logger_ng_flag_set = checked
        if checked:
            self.log_queue.put(f'{self.logger_ng_button.text()}：已开启')
        else:
            self.log_queue.put(f'{self.logger_ng_button.text()}：已关闭')

    def player_flag(self, checked):
        self.player_flag_set = checked
        if checked:
            self.log_queue.put(f'{self.play_button.text()}：已开启')
        else:
            self.log_queue.put(f'{self.play_button.text()}：已关闭')

    def model_file_path(self):
        self.model_path_set, _ = QFileDialog.getOpenFileName(self, "choose file", r"./")
        self.log_queue.put(f'新模型文件夹路径:{self.model_path_set}')

    def run(self):
        if self.button_run.text() == '开始':
            self.button_pth.hide()
            self.q1.hide()
            self.button_run.setText('结束')
            if not self.model_path_set:
                if not os.path.exists('./model.pth'):
                    self.log_queue.put('程序运行错误：本地模型文件不存在.')
                else:
                    self.model_path_set = './model.pth'
            self.log_queue.put('检测程序启动中...')
            # try:
            ret, self.joins = RelayProject(self.model_path_set, self.log_queue, play_flag=self.player_flag_set,
                                           logger_ng_flag_set=self.logger_ng_flag_set,
                                           logger_ok_flag_set=self.logger_ok_flag_set,
                                           inference_interval_set=self.inference_interval_set,
                                           save_audio_set=self.save_audio_set,
                                           keepalive_queue=self.keepalive_queue,
                                           ).run()
            if not ret:
                self.button_pth.show()
                self.q1.show()
                self.button_run.setText('开始')
            else:
                self.log_queue.put('检测程序启动完成!')
            # except Exception as e:
            #     self.printf(f"程序运行错误：{str(e)}")
            #     return

        else:
            self.button_pth.show()
            self.q1.show()
            self.button_run.setText('开始')
            for one in self.joins:
                if one:
                    self.log_queue.put(f"{one.name}(PID:{one.pid},PPID:{os.getpid()}, PPPID:{os.getppid()})已退出")
                    one.kill()


class LoggerThread(QtCore.QThread):
    def __init__(self, logger, log_queue):
        super(LoggerThread, self).__init__()
        self.logger = logger
        self.log_queue = log_queue

    def __del__(self):
        self.wait()

    def run(self):
        while True:
            if self.log_queue:
                ret = self.log_queue.get()
                self.logger.append(ret)
                self.logger.moveCursor(QtGui.QTextCursor.End)
                QtWidgets.QApplication.processEvents()

            else:
                time.sleep(1)


class KeepaliveThread(QtCore.QThread):
    def __init__(self, widget):
        super(KeepaliveThread, self).__init__()
        self.widget = widget
        self.log_queue = widget.log_queue
        self.keepalive_queue = widget.keepalive_queue

    def __del__(self):
        self.wait()

    def run(self):
        while True:
            if self.keepalive_queue:
                ret = self.keepalive_queue.get()
                type_flag, info = ret.split('#')
                if type_flag == 'ERROR':
                    self.log_queue.put(f"{info}")
                    for one in self.widget.joins:
                        self.widget.button_pth.show()
                        self.widget.q1.show()
                        self.widget.button_run.setText('开始')
                        if one:
                            self.log_queue.put(f"{one.name}(PID:{one.pid},PPID:{os.getpid()}, PPPID:{os.getppid()})已退出")
                            one.kill()
            else:
                time.sleep(1)


if "__main__" == __name__:
    if not os.path.exists("./save_audio"):
        os.mkdir("./save_audio")
    app = QtWidgets.QApplication([])
    app.setWindowIcon(QIcon('./icon.ico'))
    widget = MyWidget()
    widget.resize(640, 480)
    widget.show()
    kt = KeepaliveThread(widget)
    kt.start()
    lt = LoggerThread(widget.log, widget.log_queue)
    lt.start()
    sys.exit(app.exec())
