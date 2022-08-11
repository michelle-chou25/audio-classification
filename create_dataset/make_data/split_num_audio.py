import os.path
import librosa


def split_sound(input_file_path, output_path, block_num=2, file_name=''):
    ori_data, ori_sr = librosa.load(input_file_path, sr=None)
    all_second = len(ori_data) / ori_sr
    interval_rate = int(len(ori_data) // block_num)
    print('all second:', all_second, "s")
    print('all frame:', len(ori_data))
    print('all rate:', ori_sr, 'Hz')
    for idx, i in enumerate(range(0, block_num)):
        out_data = ori_data[i * interval_rate: i * interval_rate + interval_rate]
        librosa.output.write_wav(os.path.join(output_path, f"{idx}_{file_name}.wav"), out_data, ori_sr)

files_path = rf'./test_data'
new_path = rf'./test_data_split'
files = os.listdir(files_path)
for one in files:
    split_sound(os.path.join(files_path, one), new_path, 50, file_name=one.replace('.WAV', ''))
