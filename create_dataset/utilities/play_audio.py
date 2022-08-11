import pyaudio
import wave
from tqdm import tqdm


def play_audio(wave_path, chunk=1024):
    """
    This function is to play audio file
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