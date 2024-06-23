"""
Project: SelfSupSurg
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""


import os
import sys
import glob
import json
import numpy as np
from pathlib import Path
from skimage import measure
from sklearn import metrics
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import average_precision_score

np.seterr(all=None, divide=None, over=None, under=None, invalid=None)

root_dir = "Cholec80/phase"
header_phase = "accuracy, precision, recall, f-score, support, experiment\n"
header_phase_class = "F1_P1, F1_P2, F1_P3, F1_P4, F1_P5, F1_P6, F1_P7, experiment\n"


def class_metrics(labels, predictions, num_cls=7):
    exp_labels = np.array(range(num_cls))
    missing = [
        idx for idx in exp_labels if idx not in labels and idx not in predictions
    ]
    class_score = score(labels, predictions)
    for miss in missing:
        class_score = [np.insert(np.float32(sc), miss, np.nan) for sc in class_score]

    return class_score


def mAP(labels, predictions, mean=True, istensor=True):
    if istensor:
        labels = labels.detach().cpu().numpy()
        predictions = predictions.detach().cpu().numpy()
    metrics = np.array(average_precision_score(labels, predictions, average=None))
    if mean:
        metrics = np.sum([x for x in metrics if x == x]) / len(metrics)
    return metrics


def read_predictions(path, txt_name):
    path_str = str(path)
    inds_id = []
    inds = []
    preds = []
    targets = []
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

    return inds, preds, targets


def normalize_predictions(predicts):
    predicts_norm = np.argmax(predicts, axis=1)
    return predicts_norm


def compute_phase_scores(inds, labels, predicts, agg, directory=""):
    if agg == "class" and len(labels) == 0:
        return [-1] * 7, []
    if len(labels) == 0:
        return [-1] * 5, []
    labels = np.array(labels).squeeze()
    preds = normalize_predictions(predicts)
    if agg == "frame":
        # 按照视频帧累加计算对应的整个数据集的: precision, recall, f_score, true_sum
        # precision_recall_fscore_support返回每个类别对应的precision, recall, f_score, true_sum, 对应[4, 7]
        scores = score(labels, preds)
        acc = np.sum(labels == preds) * 100 / len(labels)
        acc = np.around(acc, 2)
        print(len(scores), len(scores[0]))

        mean = np.mean(np.vstack(scores).T, axis=0)
        mean[:-1] *= 100
        mean = np.around(mean, 2)
        mean = [
            acc
        ] + mean.tolist()  # 输出acc, precision, recall, f_score, label_sum_in_y_true

        std = np.std(np.vstack(scores).T, axis=0)
        std[:-1] *= 100
        std = np.around(std, 2)
        std = [0.0] + std.tolist()

    elif agg == "class":
        # 按视频划分，报告每个视频的F1得分（按类别输出分数），并对每个类别的F1求平均值来计算
        vid = np.array([int(ind.split(" ")[1][5:]) for ind in inds])

        class_f1 = []
        for v in np.unique(vid):
            sub_inds = np.argwhere(vid == v)
            sub_labels = labels[sub_inds]
            sub_preds = preds[sub_inds]

            # compute F1
            # 计算单个视频内所有帧对应的precision, recall, f_score, true_sum，返回每个类别对应的指标
            vid_score = class_metrics(sub_labels, sub_preds)
            # vid_score[2]表示每个类别对应的F1分数
            class_f1.append(np.array(vid_score[2]) * 100)
            # metrics.classification_report  # 分类报告输出
        print(len(class_f1), len(class_f1[0]))
        # class_f1:[test_video_clips, 7] 存储了每个视频/每个类别对应的F1分数，平均后得到每个类别的F1分数
        mean = np.around(np.nanmean(class_f1, axis=0), 2).tolist()
        std = np.around(np.nanstd(class_f1, axis=0), 2).tolist()

    elif agg == "video":
        # 按视频划分，报告每个视频的ACC和F1得分，并对每个视频的ACC和F1求平均值来计算
        vid = np.array([int(ind.split(" ")[1][5:]) for ind in inds])
        accs = []
        scores = []
        for v in np.unique(vid):
            sub_inds = np.argwhere(vid == v)
            sub_labels = labels[sub_inds]
            sub_preds = preds[sub_inds]

            # 计算整个视频内的Acc
            vid_acc = np.sum(sub_labels == sub_preds) * 100 / len(sub_labels)
            accs.append(vid_acc)

            # 计算单个视频内所有帧对应的precision, recall, f_score, true_sum，返回每个类别对应的指标
            vid_score = score(sub_labels, sub_preds)
            # vid_score存储每个类别对应的分数，直接计算平均，得到跨类别平均值
            # 使用“macro”平均方案，计算每个类别的F1，然后做平均（各类别F1的权重相同）
            mean = np.mean(np.vstack(vid_score).T, axis=0)
            mean[:-1] *= 100
            scores.append(mean)

        # summarize
        overall_acc = np.mean(np.stack(accs))
        overall_acc = np.around(overall_acc, 2)

        overall_f1 = np.mean(np.stack(scores), axis=0)
        overall_f1 = np.around(overall_f1, 2)

        mean = [overall_acc] + overall_f1.tolist()

        std = np.std(np.stack(scores), axis=0)
        std = np.around(std, 2)
        std = [np.std(np.stack(accs))] + std.tolist()

    elif agg == "video_relaxed":
        # 按视频划分，按照宽松标准报告每个视频的ACC和F1得分，并对每个视频的ACC和F1求平均值来计算
        vid = np.array([int(ind.split(" ")[1][5:]) for ind in inds])
        accs = []
        scores = []
        for v in np.unique(vid):
            sub_inds = np.argwhere(vid == v)
            sub_labels = labels[sub_inds]
            sub_preds = preds[sub_inds]

            vid_prec, vid_rec, vid_f1, vid_jacc, vid_acc = compute_phase_relaxed_scores(
                sub_preds, sub_labels
            )
            accs.append(vid_acc)
            scores.append(
                [np.nanmean(vid_prec), np.nanmean(vid_rec), np.nanmean(vid_f1), -1]
            )
        mean = [np.mean(np.stack(accs))] + np.mean(np.stack(scores), axis=0).tolist()
        std = [np.std(np.stack(accs))] + np.std(np.stack(scores), axis=0).tolist()

    return mean, std


def compute_phase_relaxed_scores(preds, targets, boundary_size=10):
    # EVALUATE
    # A function to evaluate the performance of the phase recognition method
    # providing jaccard index, precision, and recall for each phase
    # and accuracy over the surgery. All metrics are computed in a relaxed
    # boundary mode.
    # OUTPUT:
    #    res: the jaccard index per phase (relaxed) - NaN for non existing phase in GT
    #    prec: precision per phase (relaxed)        - NaN for non existing phase in GT
    #    rec: recall per phase (relaxed)            - NaN for non existing phase in GT
    #    acc: the accuracy over the video (relaxed)
    res, prec, rec = [], [], []
    diff = preds - targets
    updatedDiff = diff.copy()

    # obtain the true positive with relaxed boundary
    for iPhase in range(7):
        labels, num = measure.label(targets == iPhase, return_num=True)

        for iConn in range(1, num + 1):
            comp = np.argwhere(labels == iConn)
            startIdx = np.min(comp)
            endIdx = np.max(comp) + 1

            curDiff = diff[startIdx:endIdx]

            # in the case where the phase is shorter than the relaxed boundary
            t = boundary_size
            if t > len(curDiff):
                t = len(curDiff)

            # relaxed boundary
            # revised for cholec80 dataset !!!!!!!!!!!
            if (
                iPhase == 3 or iPhase == 4
            ):  # Gallbladder dissection and packaging might jump between two phases
                curDiff[:t][curDiff[:t] == -1] = 0  # late transition

                # early transition, 5 can be predicted as 6/7 at the end > 5 followed by 6/7
                curDiff[-t:][curDiff[-t:] == 1] = 0
                curDiff[-t:][curDiff[-t:] == 2] = 0

            elif (
                iPhase == 5 or iPhase == 6
            ):  # Gallbladder dissection might jump between two phases
                # late transition
                curDiff[:t][curDiff[:t] == -1] = 0
                curDiff[:t][curDiff[:t] == -2] = 0

                # early transition
                curDiff[-t:][curDiff[-t:] == 1] = 0
                curDiff[-t:][curDiff[-t:] == 2] = 0

            else:
                # general situation
                curDiff[:t][curDiff[:t] == -1] = 0  # late transition
                curDiff[-t:][curDiff[-t:] == 1] = 0  # early transition

            updatedDiff[startIdx:endIdx] = curDiff

    # compute jaccard index, prec, and rec per phase
    for iPhase in range(7):
        gt_num = (targets == iPhase).sum()
        if gt_num == 0:
            # no iPhase in current ground truth, assigned NaN values
            # SHOULD be excluded in the computation of mean (use nanmean)
            res.append(np.nan)
            prec.append(np.nan)
            rec.append(np.nan)
            continue

        # get all indices where pred is iPhase
        tp_and_fp = np.argwhere(preds == iPhase).flatten()
        tp_and_fn = np.argwhere(targets == iPhase).flatten()
        union = np.union1d(tp_and_fp, tp_and_fn)

        # compute tp
        tp = np.sum(updatedDiff[tp_and_fp] == 0)

        # divide by union to get jaccard
        jaccard = tp / len(union)
        jaccard = jaccard * 100

        res.append(jaccard)

        # Compute prec and rec
        prec.append(tp * 100 / len(tp_and_fp))
        rec.append(tp * 100 / len(tp_and_fn))

    # compute accuracy
    acc = sum(updatedDiff == 0) / len(targets)
    acc = acc * 100

    # compute f1
    prec = np.array(prec)
    rec = np.array(rec)
    f1 = 2 * prec * rec / (prec + rec)
    res = np.array(res)

    return prec, rec, f1, res, acc


def collect_metrics(directory, task="phase", agg="frame", txt_name="test.txt"):
    inds, preds, targets = read_predictions(directory, txt_name)
    score_fn = compute_phase_scores
    metrics, stds = score_fn(inds, targets, preds, agg, directory)
    return metrics, stds


def metrics_collator(directory, agg, task="phase", name="sr", txt_name="test.txt"):
    results_str = header_phase
    results_str = (
        header_phase_class if agg == "class" and task == "phase" else results_str
    )
    results_file = os.path.join(
        directory, "metrics_{:s}.csv".format("_".join([task, name, agg]))
    )
    metric_and_std = collect_metrics(directory, task, agg, txt_name)
    results = ",".join(
        map(lambda x, y: "{:.2f} +- {:.2f}".format(x, y), *metric_and_std)
    )
    results_str += ",".join([results, directory]) + "\n"

    if results_str == header_phase or results_str == header_phase_class:
        return

    try:
        with open(results_file, "w") as fp:
            fp.write(results_str)
        print("collating dir: done!!!")
    except:
        print("warning: could not write to folder:", directory)
    return


if __name__ == "__main__":
    # agg_mode = "frame"
    # agg_mode = "class"
    # agg_mode = 'video'
    agg_mode = "video_relaxed"

    src_dir = "/home/yangshu/SurgVideoMAE/Cholec80/ImageNet/phase_video_timesformer/timesformer_online_key_frame_frame8_Fixed_Stride_4_5"
    task = "phase"
    name = "ImageNet-21k"
    txt_name = [str(i) + ".txt" for i in range(2)]
    metrics_collator(src_dir, agg_mode, task, name, txt_name)

    # # 可视化混淆矩阵
    # import seaborn as sns
    # import pandas as pd
    # import matplotlib.pyplot as plt

    # y_test    = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]
    # y_predict = [1, 1, 1, 3, 3, 2, 2, 3, 3, 3, 4, 3, 4, 3]
    # cm = metrics.confusion_matrix(y_test, y_predict)
    # df = pd.DataFrame(cm)
    # ax = sns.heatmap(df,cmap="Blues",annot=True)
    # ax.set_title('confusion matrix')
    # ax.set_xlabel('predict')
    # ax.set_ylabel('true')
    # plt.show()
