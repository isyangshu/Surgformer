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

TOTAL_FRAMES = 0

for video_name in VIDEO_NAMES:
    video_path = os.path.join(ROOT_DIR, video_name)
    video_files = list()
    for filename in os.listdir(video_path):
        if filename.endswith(".mp4"):
            # 输出文件路径
            video_files.append(os.path.join(video_path, filename))
    video_files = sorted(video_files)
    count = 0
    count_1fps = 0
    for video_file in video_files:
        print(video_file)
        vidcap = cv2.VideoCapture(video_file)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        print("fps", fps)
        success=True
        save_dir = os.path.join(FRAME_DIR, video_name)
        os.makedirs(save_dir, exist_ok=True)
        frames_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) // fps
        print(frames_count)
        while success is True:
            success,image = vidcap.read()
            if success:
                if count % fps == 0:
                    height, width, _ = image.shape
                    # 计算短边长度
                    short_edge = min(height, width)

                    # 计算缩放比例
                    scale_ratio = 360 / short_edge

                    # 根据缩放比例计算新的尺寸
                    new_height = int(height * scale_ratio)
                    new_width = int(width * scale_ratio)

                    # 使用 cv2.resize() 缩放图像
                    resized_img = cv2.resize(image, (new_width, new_height))
                    cv2.imwrite(save_dir + '/' + str(int(count//fps)).zfill(5) + '.png', resized_img)
                    count_1fps += 1
                count+=1
        vidcap.release()
        cv2.destroyAllWindows()
    print(count_1fps)
    TOTAL_FRAMES += count_1fps
print(TOTAL_FRAMES)