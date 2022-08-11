import os
import shutil

import numpy as np
from tqdm import tqdm

def mov_datasets_from_nfs(ori_data_path, new_data_path):
    if not os.path.exists(os.path.join(new_data_path)):
        os.mkdir(os.path.join(new_data_path))

    try:
        shutil.rmtree(os.path.join(new_data_path, 'NG'))
    except Exception as e:
        print(f"rmtree {os.path.join(new_data_path, 'NG')} ERROR:", e)
    try:
        shutil.rmtree(os.path.join(new_data_path, 'OK'))
    except Exception as e:
        print(f"rmtree {os.path.join(new_data_path, 'OK')} ERROR:", e)


    ng_folders = os.listdir(os.path.join(ori_data_path, 'NG'))
    ok_folders = os.listdir(os.path.join(ori_data_path, 'OK'))

    ng_num = 0
    ok_num = 0

    ng_folders_list = []
    for one_folder in ng_folders:
        folder_path = os.path.join(ori_data_path, 'NG', one_folder)
        lens = len(os.listdir(folder_path))
        print(f'NG({one_folder}):', lens)
        ng_folders_list.append(folder_path)
        ng_num += lens
    ok_folders_list = []
    for one_folder in ok_folders:
        folder_path = os.path.join(ori_data_path, 'OK', one_folder)
        lens = len(os.listdir(folder_path))
        print(f'OK({one_folder}):', lens)
        ok_folders_list.append(folder_path)
        ok_num += lens
    print('ALL NUM:', ng_num + ok_num)
    print("start copy dataset from NFS...")

    for one_folder in ng_folders_list:
        shutil.copytree(one_folder, os.path.join(new_data_path, 'NG'), dirs_exist_ok=True)
    for one_folder in ok_folders_list:
        shutil.copytree(one_folder, os.path.join(new_data_path, 'OK'), dirs_exist_ok=True)
    print("start copy dataset from NFS end!")

    print("all end!")


if __name__ == '__main__':
    # 如果ratio大于0， 数据切割出val
    mov_datasets_from_nfs(rf'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\train',
                          rf'C:\workspace\zhouhe\datasets\relay\wz_relay_file',
                          )

    mov_datasets_from_nfs(rf'\\10.20.200.170\data\ext\PVDefectData\test2021\zh\dt\wz\__org__\val',
                          rf'C:\workspace\zhouhe\datasets\relay\wz_relay_file_val',
                          )