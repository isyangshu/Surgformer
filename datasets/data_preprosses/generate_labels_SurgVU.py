import numpy as np
import os
import cv2
from tqdm import tqdm
import csv
import math

# 8 tasks
ROOT_DIR = "/jhcnas4/syangcw/surgvu24/videos"
VIDEO_NAMES = os.listdir(ROOT_DIR)
VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'case' in x])
FRAME_DIR = "/jhcnas4/syangcw/surgvu24/frames"
LABEL_DIR = "/jhcnas4/syangcw/surgvu24/labels"

if not os.path.exists(FRAME_DIR):
    os.makedirs(FRAME_DIR)
if not os.path.exists(LABEL_DIR):
    os.makedirs(LABEL_DIR)

for video_name in VIDEO_NAMES[2:]:
    video_path = os.path.join(ROOT_DIR, video_name)
    task_label = os.path.join(video_path, "tasks.csv")
    video_files = list()
    for filename in os.listdir(video_path):
        if filename.endswith(".mp4"):
            # 输出文件路径
            video_files.append(os.path.join(video_path, filename))
    video_files = sorted(video_files)
    print(video_files)
    label_save_file = os.path.join(LABEL_DIR, video_name+'.txt')
    video_length = len(os.listdir(video_files[0]))
    with open(task_label, 'r') as csvfile:
        # 创建 CSV 读取器
        label_reader = list()
        reader = csv.reader(csvfile)
        next(reader)
        # 逐行读取数据
        for row in reader:
            if len(row) != 7:
                print(video_name)
            else:
                start_part, start_time, stop_part, stop_time, groundtruth_taskname = row[1], row[2], row[3], row[4], row[6]
                print(int(start_part))
                if start_part == stop_part and int(start_part) == 1:
                    label_reader.append((math.ceil(float(start_time)), math.floor(float(stop_time)), groundtruth_taskname))
                elif start_part == stop_part and int(start_part) == 2:
                    label_reader.append((video_length + math.ceil(float(start_time)), video_length + math.floor(float(stop_time)), groundtruth_taskname))
                else:
                    label_reader.append((math.ceil(float(start_time)), video_length + math.floor(float(stop_time)), groundtruth_taskname))
    print(label_reader)
    break       
    # count = 0
    # for video_file in video_files:
    #     print(video_file)
    #     vidcap = cv2.VideoCapture(video_file)
    #     fps = vidcap.get(cv2.CAP_PROP_FPS)
    #     print("fps", fps)
    #     success=True
    #     save_dir = os.path.join(FRAME_DIR, video_name)
    #     os.makedirs(save_dir, exist_ok=True)
    #     frames_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) // fps
    #     print(frames_count)
        # while success is True:
        #     success,image = vidcap.read()
        #     if success:
        #         if count % fps == 0:
        #             cv2.imwrite(save_dir + str(int(count//fps)).zfill(5) + '.png', image)
        #         count+=1
        # vidcap.release()
        # cv2.destroyAllWindows()
        # print(count)
        # FRAME_NUMBERS += count

# print('Total Frams', FRAME_NUMBERS)
