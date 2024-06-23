import os
import numpy as np
import math
import sys
sys.path.append("/home/syangcw/Surgformer")
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import utils
from downstream_triplet.ivtmetrics import Recognition

def train_class_batch(model, samples, target, criterion):
    outputs = model(samples)
    loss = criterion(outputs, target.float())
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return (
        optimizer.loss_scale
        if hasattr(optimizer, "loss_scale")
        else optimizer.cur_scale
    )


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (
            lr_schedule_values is not None
            or wd_schedule_values is not None
            and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(model, samples, targets, criterion)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(model, samples, targets, criterion)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
# def validation_one_epoch(data_loader, model, device):
#     mAP = Recognition(100)
#     mAP.reset_global()
#     criterion = torch.nn.BCEWithLogitsLoss()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Val:"
#     activation  = torch.nn.Sigmoid()
#     # switch to evaluation mode
#     model.eval()

#     for batch in metric_logger.log_every(data_loader, 10, header):
#         videos = batch[0]
#         target = batch[1]

#         videos = videos.to(device, non_blocking=True)
#         target = target.to(device, non_blocking=True)

#         # compute output
#         with torch.cuda.amp.autocast():
#             output = model(videos)
#             loss = criterion(output, target.float())
        
#         metric_logger.update(loss=loss.item())
#         mAP.update(target.float().detach().cpu(), activation(output).detach().cpu()) # Log metrics 
#     mAP.video_end() 

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     mAP.synchronize_between_processes()
#     print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))
#     print("* mAP", mAP.compute_video_AP('ivt', ignore_null=True)['mAP'])

#     test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
#     test_state['mAP'] = mAP.compute_video_AP('ivt', ignore_null=True)['mAP'] * 100.0
#     return test_state


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, file, num_tasks):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Val:"

    # switch to evaluation mode
    model.eval()
    val_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target.float())
        
        for i in range(output.size(0)):
            unique_id, video_id, frame_id = ids[i].strip().split('_')
            string = "{} {} {} {} {}\n".format(
                unique_id,
                video_id,
                frame_id,
                str(output.data[i].cpu().numpy().tolist()),
                str(target[i].cpu().numpy().tolist()),
            )
            val_result.append(string)
        
        metric_logger.update(loss=loss.item())

    if not os.path.exists(file):
        # os.mknod(file)  # 用于创建一个指定文件名的文件系统节点，暂时无权限
        open(file, 'a').close()
        
    with open(file, "w") as f:
        for line in val_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))

    test_state = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    merge_file = file[:-5]
    mAP_i, mAP_v, mAP_t, mAP_iv, mAP_it, mAP_ivt = merge_val(merge_file, num_tasks)
    test_state['mAP'] = mAP_ivt
    return test_state


@torch.no_grad()
def final_triplet_test(data_loader, model, device, file):
    criterion = torch.nn.BCEWithLogitsLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()
    final_result = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]

        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)  
            loss = criterion(output, target.float())

        for i in range(output.size(0)):
            unique_id, video_id, frame_id = ids[i].strip().split('_')
            string = "{} {} {} {} {}\n".format(
                unique_id,
                video_id,
                frame_id,
                str(output.data[i].cpu().numpy().tolist()),
                str(target[i].cpu().numpy().tolist()),
            )
            final_result.append(string)

        metric_logger.update(loss=loss.item())

    if not os.path.exists(file):
        # os.mknod(file)  # 用于创建一个指定文件名的文件系统节点，暂时无权限
        open(file, 'a').close()
        
    with open(file, "w") as f:
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + ".txt")
        print("Merge File %d/%d: %s" % (x+1, num_tasks, file))
        lines = open(file, "r").readlines()
        for line in lines:
            line = line.strip()
            name = line.split("[")[0]
            label = line.split("[")[2].split("]")[0]
            data = line.split("[")[1].split("]")[0]
            label = np.fromstring(label, dtype=float, sep=',')
            data = np.fromstring(data, dtype=float, sep=",")
            if not name in dict_feats:
                dict_feats[name] = 0
                dict_label[name] = 0

            dict_feats[name] = data
            dict_label[name] = label
    
    dict_feats_sort = dict(sorted(dict_feats.items(), key=lambda item: int(item[0].split(" ")[0])))
    dict_label_sort = dict(sorted(dict_label.items(), key=lambda item: int(item[0].split(" ")[0])))
    print("Length:", len(dict_label_sort))
    print("Computing final results")
    
    mAP_i, mAP_v, mAP_t, mAP_iv, mAP_it, mAP_ivt = compute_results(dict_feats_sort, dict_label_sort, eval_path)

    return mAP_i * 100, mAP_v * 100, mAP_t * 100, mAP_iv * 100, mAP_it * 100, mAP_ivt * 100

def merge_val(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + ".txt")
        print("Merge File %d/%d: %s" % (x+1, num_tasks, file))
        lines = open(file, "r").readlines()
        for line in lines:
            line = line.strip()
            name = line.split("[")[0]
            label = line.split("[")[2].split("]")[0]
            data = line.split("[")[1].split("]")[0]
            label = np.fromstring(label, dtype=float, sep=',')
            data = np.fromstring(data, dtype=float, sep=",")
            if not name in dict_feats:
                dict_feats[name] = 0
                dict_label[name] = 0

            dict_feats[name] = data
            dict_label[name] = label
    
    dict_feats_sort = dict(sorted(dict_feats.items(), key=lambda item: int(item[0].split(" ")[0])))
    dict_label_sort = dict(sorted(dict_label.items(), key=lambda item: int(item[0].split(" ")[0])))
    print("Length:", len(dict_label_sort))
    print("Computing final results")
    
    mAP_i, mAP_v, mAP_t, mAP_iv, mAP_it, mAP_ivt = compute_results_val(dict_feats_sort, dict_label_sort, eval_path)

    return mAP_i * 100, mAP_v * 100, mAP_t * 100, mAP_iv * 100, mAP_it * 100, mAP_ivt * 100

def compute_results(dict_feat, dict_label, eval_path):
    dict_feat = divided_videos(dict_feat)
    dict_label = divided_videos(dict_label)
    activation  = torch.nn.Sigmoid()
    mAP = Recognition(100)
    mAP.reset_global()
    for video_id in dict_feat.keys():
        feats = dict_feat[video_id]
        labels = dict_label[video_id]
        mAP.reset()  
        for feat, label in zip(feats, labels):
            mAP.update(label[np.newaxis, :], activation(torch.tensor(feat)).unsqueeze(0).numpy()) # Log metrics 
        mAP.video_end() 
    print("Total Video Number: ", len(mAP.global_predictions))
    mAP_i = mAP.compute_video_AP('i', ignore_null=True)
    mAP_v = mAP.compute_video_AP('v', ignore_null=True)
    mAP_t = mAP.compute_video_AP('t', ignore_null=True)
    mAP_iv = mAP.compute_video_AP('iv', ignore_null=True)
    mAP_it = mAP.compute_video_AP('it', ignore_null=True)
    mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=True) 
    logfile = os.path.join(eval_path, "result.csv")
    print('-'*50, file=open(logfile, 'a+'))
    print('Test Results\nPer-category AP: ', file=open(logfile, 'a+'))
    print(f'I   : {mAP_i["AP"]}', file=open(logfile, 'a+'))
    print(f'V   : {mAP_v["AP"]}', file=open(logfile, 'a+'))
    print(f'T   : {mAP_t["AP"]}', file=open(logfile, 'a+'))
    print(f'IV  : {mAP_iv["AP"]}', file=open(logfile, 'a+'))
    print(f'IT  : {mAP_it["AP"]}', file=open(logfile, 'a+'))
    print(f'IVT : {mAP_ivt["AP"]}', file=open(logfile, 'a+'))
    print('-'*50, file=open(logfile, 'a+'))
    print(f'Mean AP:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
    print(f':::::: : {mAP_i["mAP"]:.4f} | {mAP_v["mAP"]:.4f} | {mAP_t["mAP"]:.4f} | {mAP_iv["mAP"]:.4f} | {mAP_it["mAP"]:.4f} | {mAP_ivt["mAP"]:.4f} ', file=open(logfile, 'a+'))
    print('='*50, file=open(logfile, 'a+'))
    print("Test results saved @ ", logfile)

    return mAP_i["mAP"], mAP_v["mAP"], mAP_t["mAP"], mAP_iv["mAP"], mAP_it["mAP"], mAP_ivt["mAP"]

def compute_results_val(dict_feat, dict_label, eval_path):
    dict_feat = divided_videos(dict_feat)
    dict_label = divided_videos(dict_label)
    activation  = torch.nn.Sigmoid()
    mAP = Recognition(100)
    mAP.reset_global()
    for video_id in dict_feat.keys():
        feats = dict_feat[video_id]
        labels = dict_label[video_id]
        mAP.reset()  
        for feat, label in zip(feats, labels):
            mAP.update(label[np.newaxis, :], activation(torch.tensor(feat)).unsqueeze(0).numpy()) # Log metrics 
        mAP.video_end() 
    print("Total Video Number: ", len(mAP.global_predictions))
    mAP_i = mAP.compute_video_AP('i', ignore_null=True)
    mAP_v = mAP.compute_video_AP('v', ignore_null=True)
    mAP_t = mAP.compute_video_AP('t', ignore_null=True)
    mAP_iv = mAP.compute_video_AP('iv', ignore_null=True)
    mAP_it = mAP.compute_video_AP('it', ignore_null=True)
    mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=True) 

    return mAP_i["mAP"], mAP_v["mAP"], mAP_t["mAP"], mAP_iv["mAP"], mAP_it["mAP"], mAP_ivt["mAP"]


def divided_videos(dict_video):
    new_dict = dict()
    for key, value in dict_video.items():
        unique_id, video_id, image_id = key.strip().split(" ")
        if video_id not in new_dict:
            new_dict[video_id] = list()
        new_dict[video_id].append(value)
    return new_dict     


# if __name__ == "__main__":
#     a = merge("/home/syangcw/Surgformer/results/surgformer_HTA_CholecT50_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4", 2)