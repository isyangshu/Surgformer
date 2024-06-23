import csv
import os
from collections import defaultdict
import cv2
csv_phase = "/Users/yangshu/Downloads/cataract-101/annotations.csv"
save_txt_folder = "/Users/yangshu/Downloads/cataract-101/phase_annotations"
# 读取CSV文件
videos = defaultdict(list)
with open(csv_phase, "r") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过表头行
    for row in reader:
        video_id, frame_number, phase_id = row[0].split(";")
        # 生成逐帧标注的txt文件名
        videos[video_id].append([video_id, frame_number, phase_id])

count_1 = 0
count_2 = 0
for video_id in videos.keys():
    video_detail = videos[video_id]
    
    frames_position = [int(k[1]) for k in video_detail]
    phase_id = [int(k[2]) for k in video_detail]
    if phase_id[-1] == phase_id[-2]:
        count_1 += 1
        txt_filename = os.path.join(save_txt_folder, f"{video_id}.txt")

        # 打开txt文件并写入逐帧标注
        with open(txt_filename, "a") as txtfile:
            txtfile.write("Frame\tPhase\n")  # 写入标题行
            phase = phase_id[0]
            for frame in range(frames_position[0], frames_position[-1]+1):
                if frame in frames_position:
                    position = frames_position.index(frame)
                    phase = phase_id[position]
                txtfile.write(f"{frame}\t{phase}\n")
    else:
        count_2 += 1
        txt_filename = os.path.join(save_txt_folder, f"{video_id}.txt")

        frame_folder = os.path.join("/Users/yangshu/Downloads/cataract-101/videos/", "case_" + str(video_id)+".mp4")
        vidcap = cv2.VideoCapture(frame_folder)
        length = int(vidcap.get(7))
        # 打开txt文件并写入逐帧标注
        with open(txt_filename, "a") as txtfile:
            txtfile.write("Frame\tPhase\n")  # 写入标题行
            phase = phase_id[0]
            for frame in range(frames_position[0], length):
                if frame in frames_position:
                    position = frames_position.index(frame)
                    phase = phase_id[position]
                txtfile.write(f"{frame}\t{phase}\n")
print(count_1)
print(count_2)
