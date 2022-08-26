import time
import os
import torch
import torchaudio
import numpy as np
import soundfile
import Relay
from relay_predict_app import LOG_RED, LOG_GREEN
from const import AUDIO_RATE, AUDIO_INTERVAL,AUDIO_FRAMES_PER_BUFFER,\
    AI_INFERENCE_INTERVAL_DEFAULT,AI_SAVE_AUDIO_DEFAULT,AI_MELBINS,AI_NORM_MEAN,AI_NORM_STD,\
    AI_TARGET_LENGTH,AI_UINT16_MAX, UI_LOG_SHOW_MAX_LINE_NUM


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
                 save_audio_set= AI_SAVE_AUDIO_DEFAULT):
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
        """
        预测分类结果
        :return:
        """
        r = Relay()
        # Load a ScriptModule or ScriptFunction previously saved with torch.jit.save
        audio_model = torch.git.load(self.model_path)
        audio_model.float().eval()  # 设置为eval模式
        data_frame_num = int(self.rate * self.interval)  # 22050的采样频率，采样1.5秒
        # 1323* 300/(1500ms/(33075/1323))=3315,  33075/1323=60ms
        # inference_interval_frame=6615
        inference_interval_frame = int(
            self.inference_interval / (1000 * self.interval / (
                    self.rate * self.interval / self.num_frames)) * self.num_frames)  # 300ms(必须是60的倍数) 推理一次
        current_frame_num = 0
        num_id = 0
        save_id = 0
        one_input = np.zeros([data_frame_num, 2])  # [33075,2]
        self.log_queue.put(f"检测程序开始运行...")
        while True:
            audio_one = self.record_audio_q.get()  # 获取录音Queue的第一段buffer， [1323,2] 1323个采样点，2个通道
            current_frame_num += self.num_frames
            # one_input[self.num_frames:, :].shape=[31275,2], audio_one.shape=[1323,2], concatenate的结果是
            # 把one_input的第num_id份[1323,2]个元素替换为audio_one的内容, 后面的while以此类推
            one_input = np.concatenate(one_input[self.num_frames:, :], audio_one)
            num_id += 1
            if self.play_flag:
                self.play_audio_q.put(audio_one, block=False)  # 非阻塞式在play queue里加入audio_one这么多buffer播放
            if current_frame_num == inference_interval_frame: #预测
                current_frame_num = 0  # reset current_frame_num给下一次预测用
                data_org = one_input / AI_UINT16_MAX  # 把data_org 缩小到[-1,1]的浮点数
                data = torch.from_numpy(data_org)  # np转tensor
                data = torch.transpose(data, 1, 0)  # 1轴和0轴转置， 从[33075, 2] 转为 [2, 33075]， 因为fbank的通道数是第一个维度
                data = r.preprocess(data, self.rate)
                ret = audio_model(data.float())
                # tensor.detach()Returns a new Tensor, detached from the current graph.
                # The result will never require gradient, 转为numpy， 保留两位小数
                ok_p = ret[0][1].detach().numpy().round(2)
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

