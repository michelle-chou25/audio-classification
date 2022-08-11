import librosa
import numpy as np
import soundfile
import matplotlib.pyplot as plt
import librosa.display as display


def reduce_dimen(file_name):
    data, sr = librosa.load(file_name, mono=False)
    print(data.shape)
    soundfile.write('channel1.wav', data[0])
    soundfile.write('channel2.wav',data[1])
    fig = plt.figure(num=1, figsize=(128, 32))
    soundfile.write("channel1.wav", data[0], sr)
    soundfile.write("channel2.wav",data[1], sr)
    data1, sr = librosa.load("channel1.wav")
    data2, sr = librosa.load("channel2.wav")

    fig.add_subplot(2, 2, 1)
    display.waveshow(data1[-sr:-10000], sr, alpha=0.8, x_axis='s', offset=0.5)
    plt.title("Time domain figure of mic1", fontproperties='SimHei')
    plt.axis('tight')
    fig.add_subplot(2, 2, 2)
    mel = librosa.feature.melspectrogram(y=data1, sr=sr)
    mel = librosa.power_to_db(mel, ref=np.max)
    display.TimeFormatter(lag=True)
    librosa.display.specshow(mel)
    plt.title("Mel spectrogram of mic1", fontproperties='SimHei')

    fig.add_subplot(2, 2, 3)
    display.waveshow(data2[-sr:-10000], sr, alpha=0.8, x_axis='s', offset=0.5)
    plt.title("Time domain figure of mic2", fontproperties='SimHei')
    plt.axis('tight')
    fig.add_subplot(2, 2, 4)
    mel = librosa.feature.melspectrogram(y=data2, sr=sr)
    mel = librosa.power_to_db(mel, ref=np.max)
    display.TimeFormatter(lag=True)
    librosa.display.specshow(mel)
    plt.title("Mel spectrogram of mic2", fontproperties='SimHei')

    plt.savefig("2channel_to_1channel.png")
    plt.show()


if __name__ == '__main__':
    file_name = r'7_14_9_18_85_0.0.wav'
    reduce_dimen(file_name)



