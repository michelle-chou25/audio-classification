from pydub import AudioSegment


sound = AudioSegment.from_file("7_20_15_23_83_0.99_very_weak.wav", "wav")
db = sound.dBFS  # get the db value of the wav file
gain=+10
normalized_sound = sound.apply_gain(gain)
normalized_sound.export("out.wav", format="wav")
