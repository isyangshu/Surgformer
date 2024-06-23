import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm
import math

def main():
    ROOT_DIR = "/Users/yangshu/Downloads/Cataract101/"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'videos'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x])

    VIDEO_NUMS = sorted([int(i[:-4]) for i in VIDEO_NAMES])
    TRAIN_NUMBERS = VIDEO_NUMS[:51]
    VAL_NUMBERS = VIDEO_NUMS[51:61]
    TEST_NUMBERS = VIDEO_NUMS[61:101]

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

    id2phase = {1: "Incision", 2: "Viscous agent injection", 3: "Rhexis", 
                4: "Hydrodissection", 5: "Phacoemulsificiation", 6: "Irrigation and aspiration", 7: "Capsule polishing",
                8: "Lens implant setting-up", 9: "Viscous agent removal", 10: "Tonifying and antibiotics"}

    for video_name in VIDEO_NAMES:
        video_id = video_name[:-4]
        vid_id = int(video_id)

        if vid_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
        elif vid_id in VAL_NUMBERS:
            unique_id = unique_id_val
        elif vid_id in TEST_NUMBERS:
            unique_id = unique_id_test

        # 打开视频文件
        vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, './videos/' + video_name))
        # 帧率(frames per second)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        if fps != 25:
            print(video_name, 'not at 25fps', fps)
        # 总帧数(frames)
        frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 打开Label文件
        phase_path = os.path.join(ROOT_DIR, 'phase_annotations', video_id + '.txt')
        phase_file = open(phase_path, 'r')
        phase_results = phase_file.readlines()[1:]
        phase_dict = dict()
        for phase_result in phase_results:
            a, b = phase_result.split()
            phase_dict[a] = b
        start_id = int(phase_results[0].split()[0])
        end_id = int(phase_results[-1].split()[0])
        start_id_ = int(math.ceil(start_id / 25.0) * 25)
        end_id_ = int(math.floor(end_id / 25.0) * 25)

        frame_infos = list()
        frame_id_ = 0
        for frame_id in tqdm(range(start_id_, end_id_+1, 25)):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id_
            info['original_frame_id'] = frame_id // 25
            info['video_id'] = video_id
            info['tool_gt'] = None
            info['frames'] = end_id_ //25 - start_id_//25 + 1
            phase_id = int(phase_dict[str(frame_id)])
            info['phase_gt'] = phase_id - 1
            info['phase_name'] = id2phase[int(phase_id)]
            info['fps'] = 1
            frame_infos.append(info)
            unique_id += 1
            frame_id_ += 1

        if vid_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += len(frame_infos)
            unique_id_train = unique_id
        elif vid_id in VAL_NUMBERS:
            val_pkl[video_id] = frame_infos
            VAL_FRAME_NUMBERS += len(frame_infos)
            unique_id_val = unique_id
        elif vid_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += len(frame_infos)
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
    main()

    # 读取pkl文件,rb是读取二进制文件，而r是读取文本文件

    # file = open('/Users/yangshu/Downloads/cataract-101/labels_pkl/train/1fpstrain.pickle', 'rb')
    # info = pickle.load(file)
    # total_num = 0
    # for index in info.keys():
    #     num = len(info[index])
    #     info_final = info[index][-1]
    #     if info_final['frame_id'] != info_final['frames']:
    #         print(info_final)
    #         print('!!!!!!!!!!!!!!!!!')
    #     total_num += num
    # print(total_num)
    # print(len(info['269']))
    # print(info["269"])
    # print(info['270'])
    # print(info['10'][0])
    # print(info['10'][-2])
    # print(info['10'][-1])