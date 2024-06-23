# phase: {'0': 'preparation', '1': 'carlot-triangle-dissection', '2': 'clipping-and-cutting', '3': 'gallbladder-dissection', '4': 'gallbladder-packaging', '5': 'cleaning-and-coagulation', '6': 'gallbladder-extraction'}
# instrument: {'0': 'grasper', '1': 'bipolar', '2': 'hook', '3': 'scissors', '4': 'clipper', '5': 'irrigator'}
# verb: {'0': 'grasp', '1': 'retract', '2': 'dissect', '3': 'coagulate', '4': 'clip', '5': 'cut', '6': 'aspirate', '7': 'irrigate', '8': 'pack', '9': 'null_verb'}
# target: {'0': 'gallbladder', '1': 'cystic_plate', '2': 'cystic_duct', '3': 'cystic_artery', '4': 'cystic_pedicle', '5': 'blood_vessel', '6': 'fluid', '7': 'abdominal_wall_cavity', '8': 'liver', '9': 'adhesion', '10': 'omentum', '11': 'peritoneum', '12': 'gut', '13': 'specimen_bag', '14': 'null_target'}
# triplet: {'0': 'grasper,dissect,cystic_plate', '1': 'grasper,dissect,gallbladder', '2': 'grasper,dissect,omentum', '3': 'grasper,grasp,cystic_artery', '4': 'grasper,grasp,cystic_duct', '5': 'grasper,grasp,cystic_pedicle', '6': 'grasper,grasp,cystic_plate', '7': 'grasper,grasp,gallbladder', '8': 'grasper,grasp,gut', '9': 'grasper,grasp,liver', '10': 'grasper,grasp,omentum', '11': 'grasper,grasp,peritoneum', '12': 'grasper,grasp,specimen_bag', '13': 'grasper,pack,gallbladder', '14': 'grasper,retract,cystic_duct', '15': 'grasper,retract,cystic_pedicle', '16': 'grasper,retract,cystic_plate', '17': 'grasper,retract,gallbladder', '18': 'grasper,retract,gut', '19': 'grasper,retract,liver', '20': 'grasper,retract,omentum', '21': 'grasper,retract,peritoneum', '22': 'bipolar,coagulate,abdominal_wall_cavity', '23': 'bipolar,coagulate,blood_vessel', '24': 'bipolar,coagulate,cystic_artery', '25': 'bipolar,coagulate,cystic_duct', '26': 'bipolar,coagulate,cystic_pedicle', '27': 'bipolar,coagulate,cystic_plate', '28': 'bipolar,coagulate,gallbladder', '29': 'bipolar,coagulate,liver', '30': 'bipolar,coagulate,omentum', '31': 'bipolar,coagulate,peritoneum', '32': 'bipolar,dissect,adhesion', '33': 'bipolar,dissect,cystic_artery', '34': 'bipolar,dissect,cystic_duct', '35': 'bipolar,dissect,cystic_plate', '36': 'bipolar,dissect,gallbladder', '37': 'bipolar,dissect,omentum', '38': 'bipolar,grasp,cystic_plate', '39': 'bipolar,grasp,liver', '40': 'bipolar,grasp,specimen_bag', '41': 'bipolar,retract,cystic_duct', '42': 'bipolar,retract,cystic_pedicle', '43': 'bipolar,retract,gallbladder', '44': 'bipolar,retract,liver', '45': 'bipolar,retract,omentum', '46': 'hook,coagulate,blood_vessel', '47': 'hook,coagulate,cystic_artery', '48': 'hook,coagulate,cystic_duct', '49': 'hook,coagulate,cystic_pedicle', '50': 'hook,coagulate,cystic_plate', '51': 'hook,coagulate,gallbladder', '52': 'hook,coagulate,liver', '53': 'hook,coagulate,omentum', '54': 'hook,cut,blood_vessel', '55': 'hook,cut,peritoneum', '56': 'hook,dissect,blood_vessel', '57': 'hook,dissect,cystic_artery', '58': 'hook,dissect,cystic_duct', '59': 'hook,dissect,cystic_plate', '60': 'hook,dissect,gallbladder', '61': 'hook,dissect,omentum', '62': 'hook,dissect,peritoneum', '63': 'hook,retract,gallbladder', '64': 'hook,retract,liver', '65': 'scissors,coagulate,omentum', '66': 'scissors,cut,adhesion', '67': 'scissors,cut,blood_vessel', '68': 'scissors,cut,cystic_artery', '69': 'scissors,cut,cystic_duct', '70': 'scissors,cut,cystic_plate', '71': 'scissors,cut,liver', '72': 'scissors,cut,omentum', '73': 'scissors,cut,peritoneum', '74': 'scissors,dissect,cystic_plate', '75': 'scissors,dissect,gallbladder', '76': 'scissors,dissect,omentum', '77': 'clipper,clip,blood_vessel', '78': 'clipper,clip,cystic_artery', '79': 'clipper,clip,cystic_duct', '80': 'clipper,clip,cystic_pedicle', '81': 'clipper,clip,cystic_plate', '82': 'irrigator,aspirate,fluid', '83': 'irrigator,dissect,cystic_duct', '84': 'irrigator,dissect,cystic_pedicle', '85': 'irrigator,dissect,cystic_plate', '86': 'irrigator,dissect,gallbladder', '87': 'irrigator,dissect,omentum', '88': 'irrigator,irrigate,abdominal_wall_cavity', '89': 'irrigator,irrigate,cystic_pedicle', '90': 'irrigator,irrigate,liver', '91': 'irrigator,retract,gallbladder', '92': 'irrigator,retract,liver', '93': 'irrigator,retract,omentum', '94': 'grasper,null_verb,null_target', '95': 'bipolar,null_verb,null_target', '96': 'hook,null_verb,null_target', '97': 'scissors,null_verb,null_target', '98': 'clipper,null_verb,null_target', '99': 'irrigator,null_verb,null_target'}
import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm

switcher = {
    "cholect50": {
        "train": [
            1,
            15,
            26,
            40,
            52,
            65,
            79,
            2,
            18,
            27,
            43,
            56,
            66,
            92,
            4,
            22,
            31,
            47,
            57,
            68,
            96,
            5,
            23,
            35,
            48,
            60,
            70,
            103,
            13,
            25,
            36,
            49,
            62,
            75,
            110,
        ],
        "val": [8, 12, 29, 50, 78],
        "test": [6, 51, 10, 73, 14, 74, 32, 80, 42, 111],
    },
    "cholect50-challenge": {
        "train": [
            1,
            15,
            26,
            40,
            52,
            79,
            2,
            27,
            43,
            56,
            66,
            4,
            22,
            31,
            47,
            57,
            68,
            23,
            35,
            48,
            60,
            70,
            13,
            25,
            49,
            62,
            75,
            8,
            12,
            29,
            50,
            78,
            6,
            51,
            10,
            73,
            14,
            32,
            80,
            42,
        ],
        "val": [5, 18, 36, 65, 74],
        "test": [92, 96, 103, 110, 111],
    },
}


def main():
    ROOT_DIR = "/Users/yangshu/Downloads/CholecT50"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "videos"))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x])
    dataset_split = "cholect50"
    data_split = switcher.get(dataset_split)

    TRAIN_NUMBERS = data_split["train"]
    VAL_NUMBERS = data_split["val"]
    TEST_NUMBERS = data_split["test"]

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

    id2triplet = {
        "0": "grasper,dissect,cystic_plate",
        "1": "grasper,dissect,gallbladder",
        "2": "grasper,dissect,omentum",
        "3": "grasper,grasp,cystic_artery",
        "4": "grasper,grasp,cystic_duct",
        "5": "grasper,grasp,cystic_pedicle",
        "6": "grasper,grasp,cystic_plate",
        "7": "grasper,grasp,gallbladder",
        "8": "grasper,grasp,gut",
        "9": "grasper,grasp,liver",
        "10": "grasper,grasp,omentum",
        "11": "grasper,grasp,peritoneum",
        "12": "grasper,grasp,specimen_bag",
        "13": "grasper,pack,gallbladder",
        "14": "grasper,retract,cystic_duct",
        "15": "grasper,retract,cystic_pedicle",
        "16": "grasper,retract,cystic_plate",
        "17": "grasper,retract,gallbladder",
        "18": "grasper,retract,gut",
        "19": "grasper,retract,liver",
        "20": "grasper,retract,omentum",
        "21": "grasper,retract,peritoneum",
        "22": "bipolar,coagulate,abdominal_wall_cavity",
        "23": "bipolar,coagulate,blood_vessel",
        "24": "bipolar,coagulate,cystic_artery",
        "25": "bipolar,coagulate,cystic_duct",
        "26": "bipolar,coagulate,cystic_pedicle",
        "27": "bipolar,coagulate,cystic_plate",
        "28": "bipolar,coagulate,gallbladder",
        "29": "bipolar,coagulate,liver",
        "30": "bipolar,coagulate,omentum",
        "31": "bipolar,coagulate,peritoneum",
        "32": "bipolar,dissect,adhesion",
        "33": "bipolar,dissect,cystic_artery",
        "34": "bipolar,dissect,cystic_duct",
        "35": "bipolar,dissect,cystic_plate",
        "36": "bipolar,dissect,gallbladder",
        "37": "bipolar,dissect,omentum",
        "38": "bipolar,grasp,cystic_plate",
        "39": "bipolar,grasp,liver",
        "40": "bipolar,grasp,specimen_bag",
        "41": "bipolar,retract,cystic_duct",
        "42": "bipolar,retract,cystic_pedicle",
        "43": "bipolar,retract,gallbladder",
        "44": "bipolar,retract,liver",
        "45": "bipolar,retract,omentum",
        "46": "hook,coagulate,blood_vessel",
        "47": "hook,coagulate,cystic_artery",
        "48": "hook,coagulate,cystic_duct",
        "49": "hook,coagulate,cystic_pedicle",
        "50": "hook,coagulate,cystic_plate",
        "51": "hook,coagulate,gallbladder",
        "52": "hook,coagulate,liver",
        "53": "hook,coagulate,omentum",
        "54": "hook,cut,blood_vessel",
        "55": "hook,cut,peritoneum",
        "56": "hook,dissect,blood_vessel",
        "57": "hook,dissect,cystic_artery",
        "58": "hook,dissect,cystic_duct",
        "59": "hook,dissect,cystic_plate",
        "60": "hook,dissect,gallbladder",
        "61": "hook,dissect,omentum",
        "62": "hook,dissect,peritoneum",
        "63": "hook,retract,gallbladder",
        "64": "hook,retract,liver",
        "65": "scissors,coagulate,omentum",
        "66": "scissors,cut,adhesion",
        "67": "scissors,cut,blood_vessel",
        "68": "scissors,cut,cystic_artery",
        "69": "scissors,cut,cystic_duct",
        "70": "scissors,cut,cystic_plate",
        "71": "scissors,cut,liver",
        "72": "scissors,cut,omentum",
        "73": "scissors,cut,peritoneum",
        "74": "scissors,dissect,cystic_plate",
        "75": "scissors,dissect,gallbladder",
        "76": "scissors,dissect,omentum",
        "77": "clipper,clip,blood_vessel",
        "78": "clipper,clip,cystic_artery",
        "79": "clipper,clip,cystic_duct",
        "80": "clipper,clip,cystic_pedicle",
        "81": "clipper,clip,cystic_plate",
        "82": "irrigator,aspirate,fluid",
        "83": "irrigator,dissect,cystic_duct",
        "84": "irrigator,dissect,cystic_pedicle",
        "85": "irrigator,dissect,cystic_plate",
        "86": "irrigator,dissect,gallbladder",
        "87": "irrigator,dissect,omentum",
        "88": "irrigator,irrigate,abdominal_wall_cavity",
        "89": "irrigator,irrigate,cystic_pedicle",
        "90": "irrigator,irrigate,liver",
        "91": "irrigator,retract,gallbladder",
        "92": "irrigator,retract,liver",
        "93": "irrigator,retract,omentum",
        "94": "grasper,null_verb,null_target",
        "95": "bipolar,null_verb,null_target",
        "96": "hook,null_verb,null_target",
        "97": "scissors,null_verb,null_target",
        "98": "clipper,null_verb,null_target",
        "99": "irrigator,null_verb,null_target",
    }
    for video_id in VIDEO_NAMES:
        vid_id = int(video_id[3:])

        if vid_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
        elif vid_id in VAL_NUMBERS:
            unique_id = unique_id_val
        elif vid_id in TEST_NUMBERS:
            unique_id = unique_id_test

        # 总帧数(frames)
        video_path = os.path.join(ROOT_DIR, "videos", video_id)
        frames_list = os.listdir(video_path)
        frames_list = sorted([x for x in frames_list if "png" in x])

        # 打开Label文件
        triplet_path = os.path.join(ROOT_DIR, 'triplet', video_id + '.txt')
        triplet_file = open(triplet_path, 'r')
        triplet_results = triplet_file.readlines()[1:]
        assert len(frames_list) == len(triplet_results)
        frame_infos = list()
        for frame_id in tqdm(range(0, len(frames_list))):
            info = dict()
            info['unique_id'] = unique_id
            info['frame_id'] = frame_id
            info['original_frame_id'] = frame_id
            info['video_id'] = video_id
            info['frames'] = len(frames_list)
            triplet_info = triplet_results[frame_id]
            triplet_frame = triplet_info.split()[0]
            assert int(triplet_frame) == frame_id
            triplet_id = triplet_info.split()[1]
            if triplet_id == "-1":
                continue
            triplet_id = triplet_id.split(',')
            triplet_id = sorted([int(i) for i in triplet_id])
            info['triplet_gt'] = triplet_id
            info['phase_name'] = [id2triplet[str(i)] for i in triplet_id]
            info['fps'] = 1
            frame_infos.append(info)
            unique_id += 1

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


if __name__ == "__main__":
    main()

    # 读取pkl文件,rb是读取二进制文件，而r是读取文本文件
    # file = open('/Users/yangshu/Downloads/CholecT50/labels_pkl/train/1fpstrain.pickle', 'rb')
    # info = pickle.load(file)
    # total_num = 0
    # for index in info.keys():
    #     num = len(info[index])
    #     info_final = info[index][-1]
    #     print(info_final)
    #     total_num += num
    # print(total_num)
    # print(len(info['VID01']))
    # print(info['VID01'][0])
    # print(info['VID01'][-2])
    # print(info['VID01'][-1])
