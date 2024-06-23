import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm

def main():
    ROOT_DIR = "/jhcnas1/yangshu/data/cholec80"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'videos'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])
    TRAIN_NUMBERS = np.arange(1,41).tolist()
    VAL_NUMBERS = np.arange(41,49).tolist()
    TEST_NUMBERS = np.arange(49,81).tolist()

    TRAIN_FRAME_NUMBERS = 0
    VAL_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0

    train_pkl = dict()
    val_pkl = dict()
    test_pkl = dict()
    val_test_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_val = 0
    unique_id_test = 0

    phase2id = {'Preparation': 0, 'CalotTriangleDissection': 1, 'ClippingCutting': 2, 'GallbladderDissection': 3, 
                'GallbladderPackaging': 4, 'CleaningCoagulation': 5, 'GallbladderRetraction': 6}

    for video_name in VIDEO_NAMES:
        video_id = video_name.replace('.mp4', '')
        vid_id = int(video_name.replace('.mp4', '').replace("video", ""))
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
        tool_path = os.path.join(ROOT_DIR, 'tool_annotations', video_name.replace('.mp4', '-tool.txt'))
        tool_file = open(tool_path, 'r')
        tool = tool_file.readline().strip().split()
        tool_name = tool[1:]
        tool_dict = dict()
        while tool:
            tool = tool_file.readline().strip().split()
            if len(tool) > 0:
                tool = list(map(int, tool))
                tool_dict[str(tool[0])] = tool[1:]

        phase_path = os.path.join(ROOT_DIR, 'phase_annotations', video_name.replace('.mp4', '-phase.txt'))
        phase_file = open(phase_path, 'r')
        phase_results = phase_file.readlines()[1:]

        frame_infos = list()
        frame_id_ = 0
        for frame_id in tqdm(range(0, int(frames), 25)):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id_
            info['original_frame_id'] = frame_id
            info['video_id'] = video_id

            if str(frame_id) in tool_dict:
                info['tool_gt'] = tool_dict[str(frame_id)]
            else:
                info['tool_gt'] = None

            phase = phase_results[frame_id].strip().split()
            assert int(phase[0]) == frame_id
            phase_id = phase2id[phase[1]]
            info['phase_gt'] = phase_id
            info['phase_name'] = phase[1]
            info['fps'] = 1
            info['original_frames'] = int(frames)
            info['frames'] = int(frames) // 25
            # info['tool_names'] = tool_name
            info['phase_name'] = phase[1]
            frame_infos.append(info)
            unique_id += 1
            frame_id_ += 1
        
        vid_id = int(video_name.replace('.mp4', '').replace("video", ""))
        if vid_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += frames
            unique_id_train = unique_id
        elif vid_id in VAL_NUMBERS:
            val_pkl[video_id] = frame_infos
            VAL_FRAME_NUMBERS += frames
            unique_id_val = unique_id
        elif vid_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += frames
            unique_id_test = unique_id

    val_test_pkl = {**val_pkl, **test_pkl}
    
    train_save_dir = os.path.join(ROOT_DIR, 'labels', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    val_save_dir = os.path.join(ROOT_DIR, 'labels', 'val')
    os.makedirs(val_save_dir, exist_ok=True)
    with open(os.path.join(val_save_dir, '1fpsval.pickle'), 'wb') as file:
        pickle.dump(val_pkl, file)

    test_save_dir = os.path.join(ROOT_DIR, 'labels', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)
    with open(os.path.join(test_save_dir, '1fpsval_test.pickle'), 'wb') as file:
        pickle.dump(val_test_pkl, file)


    print('TRAIN Frams', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('VAL Frams', VAL_FRAME_NUMBERS, unique_id_val)
    print('TEST Frams', TEST_FRAME_NUMBERS, unique_id_test) 
    print('VAL TEST Frames', len(val_test_pkl))

if __name__ == '__main__':
    # main()

    # 读取pkl文件,rb是读取二进制文件，而r是读取文本文件

    file = open('/Users/yangshu/Documents/SurgVideoMAE/data/cholec80/labels/val/1fpsval.pickle', 'rb')
    info = pickle.load(file)
    total_num = 0
    for index in info.keys():
        num = len(info[index])
        total_num += num
    
    # file = open('/Users/yangshu/Documents/SurgVideoMAE/data/cholec80/labels/train/1fpstrain.pickle', 'rb')
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
    # print(len(info['video32']))
    # print(info['video32'][0])
    # print(info['video32'][-2])
    # print(info['video32'][-1])

    # file_a = open('/Users/yangshu/Documents/SurgVideoMAE/data/cholec80/labels_cross/val/5fps.pickle', 'rb')
    # info_a = pickle.load(file_a)
    # file_b = open('/Users/yangshu/Documents/SurgVideoMAE/data/cholec80/labels/val/5fpsval.pickle', 'rb')
    # info_b = pickle.load(file_b)

    # for index in tqdm(info_a.keys()):
    #     for a_ in info_a[index]:
    #         for b_ in info_b[index]:
    #             if a_['Frame_id'] == b_['original_frame_id']:
    #                 if a_['Phase_gt'] != b_['phase_gt'] and a_['Tool_gt'] != b_['tool_gt']:
    #                     print('======')
