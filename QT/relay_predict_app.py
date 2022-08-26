import sys
import os
import multiprocessing
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QFileDialog, QLabel, QRadioButton, \
    QVBoxLayout, QComboBox, QCheckBox, QWidget
from PySide6 import QtWidgets, QtCore, QtGui
from const import AUDIO_RATE, AUDIO_INTERVAL,AUDIO_FRAMES_PER_BUFFER,\
    AI_INFERENCE_INTERVAL_DEFAULT,AI_SAVE_AUDIO_DEFAULT,AI_MELBINS,AI_NORM_MEAN,AI_NORM_STD,\
    AI_TARGET_LENGTH, AI_UINT16_MAX, UI_LOG_SHOW_MAX_LINE_NUM
import processes


def LOG_RED(string):
    return "<font color=red>" + string + "</font>" "<font color=black> </font>"


def LOG_GREEN(string):
    return "<font color=green>" + string + "</font>" "<font color=black> </font>"


class Form(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.joins=[]
        self.model = None
        self.player_flag_set = False
        self.logger_ng_flag_set = False
        self.logger_ok_flag_set = False
        self.inference_interval_set = AI_INFERENCE_INTERVAL_DEFAULT
        self.save_audio_set = AI_SAVE_AUDIO_DEFAULT
        self.log_queue = multiprocessing.Queue(maxsize=10)
        self.keepalive_queue = multiprocessing.Queue(maxsize=10)

        self.setWindowTitle("继电器尾音检测 V1.0")
        self.setting_label = QLabel("设置:", alignment=QtCore.Qt.AlignLeft)

        self.sync_btn = QCheckBox("同步播放开关")
        self.ng_btn = QCheckBox("显示NG")
        self.ok_btn = QCheckBox("显示OK")

        self.inter_label = QLabel("检测间隔(ms):", alignment=QtCore.Qt.AlignLeft)
        self.inter_cob = QComboBox(self)
        self.inter_cob.addItems(["1500", "2000", "2500"])

        self.save_label = QLabel("音频保存(ms):", alignment=QtCore.Qt.AlignLeft)
        self.save_cob = QComboBox(self)
        self.save_cob.addItems(["保存", "不保存"])

        self.btn_change_model = QPushButton("更改模型文件(.pth)")
        self.btn_run = QPushButton("开始")
        self.log = QtWidgets.QTextBrowser(self)
        self.log.moveCursor(QtGui.QTextCursor.End)

        #主layout
        self.global_layout = QVBoxLayout(self)

        # 网格layout， 控件分行列排列
        self.gridlayout = QtWidgets.QGridLayout(self)
        self.gridlayout.addWidget(self.setting_label, 1, 1)
        self.gridlayout.addWidget(self.sync_btn, 2, 1)
        self.gridlayout.addWidget(self.ng_btn, 2, 2)
        self.gridlayout.addWidget(self.ok_btn, 2, 3)
        self.gridlayout.addWidget(self.inter_label, 3, 1)
        self.gridlayout.addWidget(self.save_label, 3, 2)
        self.gridlayout.addWidget(self.inter_cob, 4, 1)
        self.gridlayout.addWidget(self.save_cob, 4, 2)

        # log， pretrain，run layout
        self.sub_layout = QVBoxLayout(self)
        self.sub_layout.addWidget(self.log)
        self.sub_layout.addWidget(self.btn_change_model)
        self.sub_layout.addWidget(self.btn_run)

        #  网格layout和sublayout还要被添加到一个QWidget里才能生效
        # 应用layout
        self.q1 = QWidget()
        self.q2 = QWidget()
        self.q1.setLayout(self.gridlayout)
        self.q2.setLayout(self.sub_layout)
        self.global_layout.addWidget(self.q1)
        self.global_layout.addWidget(self.q2)
        self.setLayout(self.global_layout)

        # 连接函数和button
        self.btn_change_model.clicked.connect(self.change_model)
        self.btn_run.clicked.connect(self.run)

    def change_model(self):
        """
        选择文件
        :return: None
        """
        self.model, file_type = QFileDialog.getOpenFileName(self, caption="choose file", dir=r'./')
        self.log_queue.put(f"模型路径： {self.model}")
        if self.model == '':
            print("Cancel choosing")

    def save_audio_cb_chang_value(self, v):
        self.save_audio_set = float(v.split('(')[-1][:-1])   # e.g. v=NG&OK(0.7)->0.7, 预测结果大于0.7的ok的尾音才被保存
        self.log_queue.put(f'{self.save_audio_cb_label.text()}{v}, 文件保存在工程目录下的save_audio文件夹里。')

    def inference_interval_cb_chang_value(self, v):
        self.inference_interval_set = int(v)
        self.log_queue.put(f'{self.inference_interval_cb_label.text()}{v}')

    def logger_ok_flag(self, checked):
        """
        显示OK
        :param checked:
        :return:
        """
        self.logger_ok_flag_set = checked
        if checked:
            self.log_queue.put(f'{self.logger_ok_button.text()}：已开启')
        else:
            self.log_queue.put(f'{self.logger_ok_button.text()}：已关闭')

    def logger_ng_flag(self, checked):
        """
        显示NG
        :param checked:
        :return:
        """
        self.logger_ng_flag_set = checked
        if checked:
            self.log_queue.put(f'{self.logger_ng_button.text()}：已开启')
        else:
            self.log_queue.put(f'{self.logger_ng_button.text()}：已关闭')

    def player_flag(self, checked):
        """
        播放录音
        :param checked:
        :return:
        """
        self.player_flag_set = checked
        if checked:
            self.log_queue.put(f'{self.play_button.text()}：已开启')
        else:
            self.log_queue.put(f'{self.play_button.text()}：已关闭')

    def run(self):
        """
        点击”开始“
        :return:
        """
        try:
            if self.btn_run.text() == '开始':
                self.q1.hide()  #隐藏网格widget
                self.btn_run.setText("结束")  # 等待用户点击“结束”
                if not self.model:
                    if not os.path.exists(r'.\model.pth'):
                        print("模型不存在")
                    else:
                        self.model = r'.\model.pth'
                self.log_queue.put("模型启动中...")
                #  预测尾音
                ret, self.joins = processes(self.model, self.log_queue, play_flag=self.player_flag_set,
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
            else:
                self.btn_change_model.show()
                self.q1.show()
                self.btn_run.setText("开始")
                for one in self.joins:
                    if one:
                        self.log_queue.put(f"{one.name}(PID:{one.pid},PPID:{os.getpid()}, PPPID:{os.getppid()})已退出")
                        one.kill()
        except Exception as e:
            print("Error occurs: ", str(e))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = Form()
    form.resize(480,320)
    form.show()
    sys.exit(app.exec())

