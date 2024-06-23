import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import matplotlib.patches as mpatches

path = "/Users/yangshu/Documents/PETL4SurgVideo/result_save/Ours/AutoLaparo/GA/WIT400M/16-4/"
txt_name = [str(i) + ".txt" for i in range(2)]

color_dict = {
    "0": (255, 0, 0),
    "1": (152, 251, 152),
    "2": (135, 206, 250),
    "3": (255, 225, 53),
    "4": (230, 230, 250),
    "5": (237, 145, 33),
    "6": (0, 139, 139),
}
color_dict_float = {
    "0": (255/255., 0, 0),
    "1": (152/255., 251/255., 152/255.),
    "2": (135/255., 206/255., 250/255.),
    "3": (255/255., 225/255., 53/255.),
    "4": (230/255., 230/255., 250/255.),
    "5": (237/255., 145/255., 33/255.),
    "6": (0, 139/255., 139/255.),
}

# 加载视频
path_str = str(path)
preds = []
targets = []
inds = []
inds_id = []
for file_name in txt_name:
    if os.path.isfile(os.path.join(path_str, file_name)):
        lines = open(os.path.join(path_str, file_name), "r").readlines()
        for line in lines[1:]:
            line = line.strip()
            name = line.split("[")[0].strip()
            if name in inds:
                continue
            name_id = int(line.split("[")[0].strip().split()[0])

            label = int(line.split("]")[1].split(" ")[1])
            data = np.fromstring(
                line.split("[")[1].split("]")[0], dtype=np.float32, sep=","
            )
            data = softmax(data)
            inds_id.append(name_id)
            inds.append(name)
            preds.append(data)
            targets.append(label)

idxs = np.argsort(np.array(inds_id))
assert len(inds) == len(inds_id)
inds = np.array(inds)[idxs]
preds = np.array(preds)[idxs]
targets = np.array(targets)[idxs]
predicts = np.argmax(preds, axis=1)

text_length = 1500
# 读取视频的类别序列
vid = np.array([int(ind.split(" ")[1]) for ind in inds])
for v in np.unique(vid):
    sub_inds = np.argwhere(vid == v)
    sub_labels = targets[sub_inds]
    sub_preds = predicts[sub_inds]

    sub_labels_new = [0] * 6000
    sub_preds_new = [0] * 6000
    scale_factor = len(sub_labels_new) / len(sub_labels)

    # 在目标数组中重复采样原始数据
    for i in range(6000):
        original_index = int(i // scale_factor)
        sub_labels_new[i] = sub_labels[original_index]
    for i in range(6000):
        original_index = int(i // scale_factor)
        sub_preds_new[i] = sub_preds[original_index]

    sub_preds = sub_preds_new
    sub_labels = sub_labels_new
    # 定义矩形的宽度和高度
    rectangle_height = 800
    rectangle_width = len(sub_labels) + text_length

    # 创建一个空白图像作为画布
    canvas_gt = np.ones((rectangle_height, rectangle_width, 3), dtype=np.uint8) * 255
    canvas_pred = np.ones((rectangle_height, rectangle_width, 3), dtype=np.uint8) * 255

    # 遍历视频的每一帧
    for i in range(text_length, rectangle_width):
        # 获取当前帧对应的类别
        category_gt = sub_labels[i-text_length]
        category_pred = sub_preds[i-text_length]
        # 在画布上绘制矩形，使用不同的颜色表示不同的类别
        color_gt = color_dict[str(category_gt[0])]
        color_pred = color_dict[str(category_pred[0])]

        cv2.rectangle(
            canvas_gt, (i, 0), (i, rectangle_height - 1), color_gt, thickness=cv2.FILLED
        )
        cv2.rectangle(
            canvas_pred,
            (i, 0),
            (i, rectangle_height - 1),
            color_pred,
            thickness=cv2.FILLED,
        )

    # 绘制图像序列
    fig = plt.figure()
    gs = plt.GridSpec(2, 2, width_ratios=[1, 0.25], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[:, 1])
    plt.subplots_adjust(
        top=0.7, bottom=0.3, left=0.1, right=1.0, hspace=0, wspace=0.1
    )
    ax1.set_axis_off()
    ax2.set_axis_off()
    ax3.set_axis_off()

    # 显示矩形画布
    ax1.imshow(canvas_gt, cmap="viridis")
    ax1.text(
        -0.05,
        0.5,
        "Ground Truth",
        transform=ax1.transAxes,
        va="center",
        fontweight="bold",
    )

    ax2.imshow(canvas_pred, cmap="viridis")
    ax2.text(
        -0.05, 0.5, "Prediction", transform=ax2.transAxes, va="center", fontweight="bold"
    )

    colors1 = ["0", "1", "2", "3", "4", "5", "6"]
    text = ["Phase 1", "Phase 2", "Phase 3", "Phase 4", "Phase 5", "Phase 6", "Phase 7"]

    x = 0  # 右上角 x 坐标
    y = 0.9  # 右上角 y 坐标
    for i, color in enumerate(colors1):
        rect = mpatches.Rectangle((x, y - i * 0.15), 0.2, 0.1, facecolor=color_dict_float[color])
        ax3.add_patch(rect)
        ax3.text(
            x + 0.3,
            y - i * 0.15,
            text[i],
            transform=ax3.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
        )

    plt.suptitle(
        "Video-" + str(v),
        fontsize=10,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="0.5", alpha=0.3),
        x=0.5,
        y=0.3,
    )

    # plt.show()
    plt.savefig("/Users/yangshu/Documents/vis_show/" + str(v) + ".pdf", bbox_inches="tight")
    # break