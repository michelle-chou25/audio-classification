import os
import shutil


def mov_datasets_from_nfs(ori_data_path, new_data_path):
    try:
        shutil.rmtree(os.path.join(new_data_path,'NG'))
    except Exception as e:
        print(f"rmtree {os.path.join(new_data_path,'NG')} ERROR:", e)
    try:
        shutil.rmtree(os.path.join(new_data_path,'OK'))
    except Exception as e:
        print(f"rmtree {os.path.join(new_data_path,'OK')} ERROR:", e)


    print("start copy dataset from NFS...")
    dataset_num_ng = len(os.listdir(os.path.join(ori_data_path, 'NG')))
    print("NG data:", dataset_num_ng)
    dataset_num_ok = len(os.listdir(os.path.join(ori_data_path, 'OK')))
    print("OK data:", dataset_num_ok)
    print("all data:", dataset_num_ok + dataset_num_ng
          )
    shutil.copytree(os.path.join(ori_data_path, 'NG'), os.path.join(new_data_path,'NG'))
    shutil.copytree(os.path.join(ori_data_path, 'OK'), os.path.join(new_data_path,'OK'))
    print("start copy dataset from NFS end!")



if __name__ == '__main__':
    mov_datasets_from_nfs(
                          # '/home/share_data/ext/PVDefectData/test2021/zh/dt/wz/22050',
                          '/home/share_data/ext/PVDefectData/test2021/zh/dt/wz/files_mode_wz_relay',
                          '/home/zhouhe/datasets/wz_relay')