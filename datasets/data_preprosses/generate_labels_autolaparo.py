import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm

def main():
    ROOT_DIR = "/Users/yangshu/Downloads/AutoLaparo/"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'frames'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x])

    TRAIN_NUMBERS = np.arange(1,11).tolist()
    VAL_NUMBERS = np.arange(11,15).tolist()
    TEST_NUMBERS = np.arange(15,22).tolist()

    TRAIN_FRAME_NUMBERS = 0
    VAL_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0

    train_pkl = dict()
    val_pkl = dict()
    test_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_val = 0
    unique_id_test = 0

    id2phase = {1: "Preparation", 2: "Dividing Ligament and Peritoneum", 3: "Dividing Uterine Vessels and Ligament", 
                4: "Transecting the Vagina", 5: "Specimen Removal", 6: "Suturing", 7: "Washing"}

    for video_id in VIDEO_NAMES:
        vid_id = int(video_id)
        if vid_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
        elif vid_id in VAL_NUMBERS:
            unique_id = unique_id_val
        elif vid_id in TEST_NUMBERS:
            unique_id = unique_id_test

        # 总帧数(frames)
        video_path = os.path.join(ROOT_DIR, "frames", video_id)
        frames_list = os.listdir(video_path)

        # 打开Label文件
        phase_path = os.path.join(ROOT_DIR, 'labels', "label_" + video_id + '.txt')
        phase_file = open(phase_path, 'r')
        phase_results = phase_file.readlines()[1:]

        frame_infos = list()
        frame_id_ = 0
        for frame_id in tqdm(range(0, len(frames_list))):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id_
            info['original_frame_id'] = frame_id
            info['video_id'] = video_id
            info['tool_gt'] = None
            info['frames'] = len(frames_list)
            phase = phase_results[frame_id].strip().split()
            assert int(phase[0]) == frame_id + 1
            phase_id = int(phase[1])
            info['phase_gt'] = phase_id - 1
            info['phase_name'] = id2phase[int(phase[1])]
            info['fps'] = 1
            frame_infos.append(info)
            unique_id += 1
            frame_id_ += 1
        
        if vid_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += len(frames_list)
            unique_id_train = unique_id
        elif vid_id in VAL_NUMBERS:
            val_pkl[video_id] = frame_infos
            VAL_FRAME_NUMBERS += len(frames_list)
            unique_id_val = unique_id
        elif vid_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += len(frames_list)
            unique_id_test = unique_id
    
    train_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    val_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'val')
    os.makedirs(val_save_dir, exist_ok=True)
    with open(os.path.join(val_save_dir, '1fpsval.pickle'), 'wb') as file:
        pickle.dump(val_pkl, file)

    test_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)


    print('TRAIN Frams', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('VAL Frams', VAL_FRAME_NUMBERS, unique_id_val)
    print('TEST Frams', TEST_FRAME_NUMBERS, unique_id_test) 

if __name__ == '__main__':
    # main()

    # 读取pkl文件,rb是读取二进制文件，而r是读取文本文件

    file = open('/Users/yangshu/Downloads/AutoLaparo/labels_pkl/train/1fpstrain.pickle', 'rb')
    info = pickle.load(file)
    total_num = 0
    for index in info.keys():
        num = len(info[index])
        info_final = info[index][-1]
        if info_final['frame_id'] != info_final['frames']:
            print(info_final)
            print('!!!!!!!!!!!!!!!!!')
        total_num += num
    print(total_num)
    print(len(info['10']))
    print(info['10'][0])
    print(info['10'][-2])
    print(info['10'][-1])