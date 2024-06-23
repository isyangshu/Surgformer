# -----------------------------
# Cut black margin for surgical video
# Copyright (c) CUHK 2021.
# IEEE TMI 'Temporal Relation Network for Workflow Recognition from Surgical Video'
# -----------------------------

import cv2
import os
import numpy as np
import multiprocessing
from tqdm import tqdm


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter_black(image):
    binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
    binary_image2 = cv2.medianBlur(
        binary_image2, 19
    )  # filter the noise, need to adjust the parameter based on the dataset
    x = binary_image2.shape[0]
    y = binary_image2.shape[1]

    edges_x = []
    edges_y = []
    for i in range(x):
        for j in range(10, y - 10):
            if binary_image2.item(i, j) != 0:
                edges_x.append(i)
                edges_y.append(j)

    if not edges_x:
        return image

    left = min(edges_x)  # left border
    right = max(edges_x)  # right
    width = right - left
    bottom = min(edges_y)  # bottom
    top = max(edges_y)  # top
    height = top - bottom

    pre1_picture = image[left : left + width, bottom : bottom + height]

    return pre1_picture


def process_image(image_source, image_save):
    frame = cv2.imread(image_source)
    dim = (int(frame.shape[1] / frame.shape[0] * 300), 300)
    frame = cv2.resize(frame, dim)
    frame = filter_black(frame)

    img_result = cv2.resize(frame, (250, 250))
    cv2.imwrite(image_save, img_result)


def process_video(video_id, video_source, video_save):
    create_directory_if_not_exists(video_save)

    for image_id in sorted(os.listdir(video_source)):
        if image_id == ".DS_Store":
            continue
        image_source = os.path.join(video_source, image_id)
        image_save = os.path.join(video_save, image_id)

        process_image(image_source, image_save)


if __name__ == "__main__":
    source_path = "/home/yangshu/Surgformer/data/Cholec80/frames"  # original path
    save_path = "/home/yangshu/Surgformer/data/Cholec80/frames_cutmargin"  # save path

    create_directory_if_not_exists(save_path)

    processes = []

    for video_id in tqdm(os.listdir(source_path)):
        if video_id == ".DS_Store":
            continue
        video_source = os.path.join(source_path, video_id)
        video_save = os.path.join(save_path, video_id)

        process = multiprocessing.Process(
            target=process_video, args=(video_id, video_source, video_save)
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print("Cut Done")
