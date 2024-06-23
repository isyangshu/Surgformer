import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import cv2
from functools import partial
from pathlib import Path
from collections import OrderedDict
import torch.nn.functional as F
import sys

sys.path.append("/home/syangcw/Surgformer")

from datasets.transforms.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from datasets.transforms.optim_factory import (
    create_optimizer,
    get_parameter_groups,
    LayerDecayValueAssigner,
)

from downstream_phase.datasets_phase import build_dataset
from downstream_phase.engine_for_videomamba import (
    train_one_epoch,
    validation_one_epoch,
    final_test,
    merge,
)
from utils_mamba import NativeScalerWithGradNormCount as NativeScaler
from utils_mamba import multiple_samples_collate
import utils_mamba
import contextlib
from model.videomamba import *


def get_args():
    parser = argparse.ArgumentParser(
        "Surgformer VideoMamba fine-tuning and evaluation script for video phase recognition",
        add_help=False,
    )
    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--update_freq", default=1, type=int)
    parser.add_argument("--save_ckpt_freq", default=10, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_base_patch16_224",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--parameters",
        default="videomamba_m16_k400_f64",
        type=str,
        metavar="MODEL Parameters",
        help="Name of parameters to train",
    )
    parser.add_argument("--tubelet_size", type=int, default=1)
    parser.add_argument("--orig_t_size", type=int, default=8)
    parser.add_argument("--input_size", default=224, type=int, help="videos input size")
    parser.add_argument("--use_learnable_pos_emb", action="store_true")
    parser.set_defaults(use_learnable_pos_emb=False)

    parser.add_argument(
        "--fc_drop_rate",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--attn_drop_rate",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Attention dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--disable_eval_during_finetuning", action="store_true", default=False
    )
    parser.add_argument("--model_ema", action="store_true", default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999, help="")
    parser.add_argument(
        "--model_ema_force_cpu", action="store_true", default=False, help=""
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=(0.9, 0.999),
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument("--layer_decay", type=float, default=0.75)

    parser.add_argument(
        "--warmup_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-6)",
    )

    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports, default 5",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m7-n4-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m7-n4-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train_interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)
    parser.add_argument("--short_side_size", type=int, default=224)

    # Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0, default 0.8.",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0, default 1.0.",
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Finetuning params
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--delete_head", action="store_true", help="whether delete head"
    )
    parser.add_argument("--model_key", default="model|module", type=str)
    parser.add_argument("--model_prefix", default="", type=str)
    parser.add_argument("--use_checkpoint", action="store_true")
    parser.set_defaults(use_checkpoint=False)
    parser.add_argument(
        "--checkpoint_num",
        default=0,
        type=int,
        help="number of layers for using checkpoint",
    )

    # Dataset parameters
    parser.add_argument("--prefix", default="", type=str, help="prefix for data")
    parser.add_argument("--split", default=" ", type=str, help="split for metadata")
    parser.add_argument(
        "--filename_tmpl", default="img_{:05}.jpg", type=str, help="file template"
    )
    parser.add_argument(
        "--data_path",
        default="you_data_path",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--eval_data_path",
        default=None,
        type=str,
        help="dataset path for evaluation",
    )
    parser.add_argument(
        "--nb_classes", default=7, type=int, help="number of the classification types"
    )
    parser.add_argument(
        "--imagenet_default_mean_and_std", default=True, action="store_true"
    )

    parser.add_argument(
        "--data_strategy", type=str, default="online"
    )  # online/offline
    parser.add_argument(
        "--output_mode", type=str, default="key_frame"
    )  # key_frame/all_frame
    parser.add_argument("--cut_black", action="store_true")  # True/False
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--sampling_rate", type=int, default=4)
    parser.add_argument("--trimmed", type=int, default=60)
    parser.add_argument("--time_stride", type=int, default=16)
    parser.add_argument(
        "--data_set",
        default="Cholec80",
        choices=["Cholec80", "AutoLaparo", "Cataract101"],
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--data_fps",
        default="1fps",
        choices=["", "5fps", "1fps"],
        type=str,
        help="dataset",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/syangcw/Surgformer/results",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="/home/syangcw/Surgform/results",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_false", dest="auto_resume")
    parser.set_defaults(auto_resume=True)

    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--no_save_ckpt", action="store_false", dest="save_ckpt")
    parser.set_defaults(save_ckpt=True)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--test_best", action="store_true", help="Whether test the best model"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument("--no_amp", action="store_true")
    parser.set_defaults(no_amp=False)
    # 默认使用amp，而不使用deepspeed

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument("--enable_deepspeed", action="store_true", default=False)
    parser.add_argument("--bf16", default=False, action="store_true")

    known_args, _ = parser.parse_known_args()

    if known_args.enable_deepspeed:
        try:
            import deepspeed
            from deepspeed import DeepSpeedConfig

            parser = deepspeed.add_config_arguments(parser)
            ds_init = deepspeed.initialize
        except:
            print("Please 'pip install deepspeed'")
            exit(0)
    else:
        ds_init = None

    return parser.parse_args(), ds_init


def main(args, ds_init):
    print("===========    ", "Init Distribution", "    ===========")
    utils_mamba.init_distributed_mode(args)

    if ds_init is not None:
        print("===========    ", "Use Deepspeed", "    ===========")
        utils_mamba.create_ds_config(args)

    print(args)
    if args.sampling_rate == 0:
        frame_manner = "Exponential_Stride"
    elif args.sampling_rate == -1:
        frame_manner = "Random_Stride"
    elif args.sampling_rate == -2:
        frame_manner = "Incremental_Stride"
    else:
        frame_manner = "Fixed_Stride_" + str(args.sampling_rate)

    args.output_dir = os.path.join(
        args.output_dir,
        "_".join(
            [
                args.model,
                args.data_set,
                str(args.lr),
                str(args.layer_decay),
                args.data_strategy,
                args.output_mode,
                "frame" + str(args.num_frames),
                frame_manner,
            ]
        ),
    )

    args.log_dir = os.path.join(args.output_dir, "log")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.eval:
        txt_file = open(os.path.join(args.output_dir, "hyerparamter.txt"), "w")
        txt_file.write(str(args))
    else:
        txt_file = open(os.path.join(args.output_dir, "val_hyerparamter.txt"), "w")
        txt_file.write(str(args))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils_mamba.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(
        is_train=True, test_mode=False, fps=args.data_fps, args=args
    )  # Cholec80前40个数据集用于训练：2157640

    # 是否在训练时在验证集上测试性能
    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val, _ = build_dataset(
            is_train=False, test_mode=False, fps=args.data_fps, args=args
        )  # Cholec80第41-48视频序列用于验证集：535933
    dataset_test, _ = build_dataset(
        is_train=False, test_mode=True, fps=args.data_fps, args=args
    )  # Cholec80后40个数据集用于测试：2452890

    print("Train Dataset Length: ", len(dataset_train))
    print("Val Dataset Length: ", len(dataset_val))
    print("Test Dataset Length: ", len(dataset_test))

    num_tasks = utils_mamba.get_world_size()
    global_rank = utils_mamba.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        sampler_test = torch.utils.data.DistributedSampler(
            dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Distribute Sampler For Val/Test")
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)  # 顺序采样
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        print("Sequential Sampler For Val/Test")

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils_mamba.TensorboardLogger(log_dir=args.log_dir)
        print("Log dir:", args.log_dir)
    else:
        log_writer = None

    collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        persistent_workers=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True,
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True,
        )
    else:
        data_loader_test = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    model = create_model(
        args.model,
        pretrained=False if args.finetune else True,
        parameters=args.parameters,
        img_size=args.input_size,
        num_classes=args.nb_classes,
        fc_drop_rate=args.fc_drop_rate,
        drop_path_rate=args.drop_path,
        kernel_size=args.tubelet_size,
        num_frames=args.num_frames,
        use_checkpoint=args.use_checkpoint,
        checkpoint_num=args.checkpoint_num,
    )
    txt_file.write(str(model))
    txt_file.close()

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (
        args.num_frames // args.tubelet_size,
        args.input_size // patch_size[0],
        args.input_size // patch_size[1],
    )
    print("Window size: = %s" % str(args.window_size))
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.finetune, map_location="cpu")

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split("|"):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        if "head.weight" in checkpoint_model.keys():
            if args.delete_head:
                print("Removing head from pretrained checkpoint")
                del checkpoint_model["head.weight"]
                del checkpoint_model["head.bias"]
            elif checkpoint_model["head.weight"].shape[0] == 710:
                if args.nb_classes == 400:
                    checkpoint_model["head.weight"] = checkpoint_model["head.weight"][
                        : args.nb_classes
                    ]
                    checkpoint_model["head.bias"] = checkpoint_model["head.bias"][
                        : args.nb_classes
                    ]
                elif args.nb_classes in [600, 700]:
                    # download from https://drive.google.com/drive/folders/17cJd2qopv-pEG8NSghPFjZo1UUZ6NLVm
                    map_path = f"k710/label_mixto{args.nb_classes}.json"
                    print(f"Load label map from {map_path}")
                    with open(map_path) as f:
                        label_map = json.load(f)
                    checkpoint_model["head.weight"] = checkpoint_model["head.weight"][
                        label_map
                    ]
                    checkpoint_model["head.bias"] = checkpoint_model["head.bias"][
                        label_map
                    ]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith("backbone."):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith("encoder."):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if "deit" in args.model or "videomamba" in args.model:
            pos_embed_checkpoint = checkpoint_model["pos_embed"]
            embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
            num_patches = model.patch_embed.num_patches  #
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches**0.5)

            if orig_size != new_size:
                print(
                    "Position interpolate from %dx%d to %dx%d"
                    % (orig_size, orig_size, new_size, new_size)
                )
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> B, H, W, C -> B, C, H, W
                pos_tokens = pos_tokens.reshape(
                    -1, orig_size, orig_size, embedding_size
                ).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode="bicubic",
                    align_corners=False,
                )
                # B, C, H, W -> B, H, W, C ->  B, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    -1, new_size, new_size, embedding_size
                )
                pos_tokens = pos_tokens.flatten(1, 2)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model["pos_embed"] = new_pos_embed

            # we use 8 frames for pretraining
            temporal_pos_embed = checkpoint_model["temporal_pos_embedding"]
            orig_t_size = args.orig_t_size // model.patch_embed.tubelet_size
            new_t_size = args.num_frames // model.patch_embed.tubelet_size
            # height (== width) for the checkpoint position embedding
            if orig_t_size != new_t_size:
                print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
                temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
                temporal_pos_embed = torch.nn.functional.interpolate(
                    temporal_pos_embed,
                    size=(new_t_size,),
                    mode="linear",
                    align_corners=False,
                )
                temporal_pos_embed = temporal_pos_embed.permute(0, 2, 1)
                checkpoint_model["temporal_pos_embedding"] = temporal_pos_embed

        elif "pos_embed" in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model["pos_embed"]
            embedding_size = pos_embed_checkpoint.shape[-1]  # channel dim
            num_patches = model.patch_embed.num_patches  #
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches  # 0/1

            # we use 8 frames for pretraining
            orig_t_size = args.orig_t_size // model.patch_embed.tubelet_size
            new_t_size = args.num_frames // model.patch_embed.tubelet_size
            # height (== width) for the checkpoint position embedding
            orig_size = int(
                ((pos_embed_checkpoint.shape[-2] - num_extra_tokens) // (orig_t_size))
                ** 0.5
            )
            # height (== width) for the new position embedding
            new_size = int((num_patches // new_t_size) ** 0.5)

            if orig_t_size != new_t_size:
                print(f"Temporal interpolate from {orig_t_size} to {new_t_size}")
                tmp_pos_embed = pos_embed_checkpoint.view(
                    1, orig_t_size, -1, embedding_size
                )
                tmp_pos_embed = tmp_pos_embed.permute(0, 2, 3, 1).reshape(
                    -1, embedding_size, orig_t_size
                )
                tmp_pos_embed = torch.nn.functional.interpolate(
                    tmp_pos_embed, size=new_t_size, mode="linear"
                )
                tmp_pos_embed = tmp_pos_embed.view(1, -1, embedding_size, new_t_size)
                tmp_pos_embed = tmp_pos_embed.permute(0, 3, 1, 2).reshape(
                    1, -1, embedding_size
                )
                checkpoint_model["pos_embed"] = tmp_pos_embed
                pos_embed_checkpoint = tmp_pos_embed

            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print(
                    "Position interpolate from %dx%d to %dx%d"
                    % (orig_size, orig_size, new_size, new_size)
                )
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(
                    -1, new_t_size, orig_size, orig_size, embedding_size
                )
                pos_tokens = pos_tokens.reshape(
                    -1, orig_size, orig_size, embedding_size
                ).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens,
                    size=(new_size, new_size),
                    mode="bicubic",
                    align_corners=False,
                )
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(
                    -1, new_t_size, new_size, new_size, embedding_size
                )
                pos_tokens = pos_tokens.flatten(1, 3)  # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model["pos_embed"] = new_pos_embed

        utils_mamba.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

    model.to(device)

    model_ema = None
    if args.model_ema:
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils_mamba.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    args.lr = args.lr * total_batch_size * args.num_frames / 256
    args.min_lr = args.min_lr * total_batch_size * args.num_frames / 256
    args.warmup_lr = args.warmup_lr * total_batch_size * args.num_frames / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    num_layers = model_without_ddp.get_num_layers()
    if args.layer_decay < 1.0:
        assigner = LayerDecayValueAssigner(
            list(
                args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)
            )
        )
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    skip_weight_decay_list = model.no_weight_decay()
    print("Skip weight decay list: ", skip_weight_decay_list)

    amp_autocast = contextlib.nullcontext()
    loss_scaler = "none"
    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model,
            args.weight_decay,
            skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None,
        )
        model, optimizer, _, _ = ds_init(
            args=args,
            model=model,
            model_parameters=optimizer_params,
            dist_init_required=not args.distributed,
        )

        print(
            "model.gradient_accumulation_steps() = %d"
            % model.gradient_accumulation_steps()
        )
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=False
            )
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args,
            model_without_ddp,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )

        if not args.no_amp:
            print(f"Use bf16: {args.bf16}")
            dtype = torch.bfloat16 if args.bf16 else torch.float16
            amp_autocast = torch.cuda.amp.autocast(dtype=dtype)
            loss_scaler = NativeScaler()

    print("Use step level LR scheduler!")
    lr_schedule_values = utils_mamba.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        start_warmup_value=args.warmup_lr,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils_mamba.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        num_training_steps_per_epoch,
    )
    print(
        "Max WD = %.7f, Min WD = %.7f"
        % (max(wd_schedule_values), min(wd_schedule_values))
    )

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    utils_mamba.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_ema=model_ema,
    )

    if args.eval:
        preds_file = os.path.join(args.output_dir, str(global_rank) + ".txt")
        test_stats = final_test(
            data_loader_test,
            model,
            device,
            preds_file,
            amp_autocast,
            ds=args.enable_deepspeed,
            no_amp=args.no_amp,
            bf16=args.bf16,
            maxk=5 if args.nb_classes >= 5 else 1,
        )
        torch.distributed.barrier()
        if global_rank == 0:
            print("Start merging results...")
            final_top1, final_top5 = merge(args.output_dir, num_tasks)
            print(
                f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%"
            )
            log_stats = {"Final top-1": final_top1, "Final Top-5": final_top5}
            if args.output_dir and utils_mamba.is_main_process():
                with open(
                    os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            amp_autocast,
            args.clip_grad,
            model_ema,
            mixup_fn,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            update_freq=args.update_freq,
            no_amp=args.no_amp,
            bf16=args.bf16,
        )
        if args.output_dir and args.save_ckpt:
            # if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            #     utils.save_model(
            #         args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #         loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
            utils_mamba.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
                model_name="latest",
                model_ema=model_ema,
            )
        if data_loader_val is not None:
            test_stats = validation_one_epoch(
                data_loader_val,
                model,
                device,
                amp_autocast,
                ds=args.enable_deepspeed,
                no_amp=args.no_amp,
                bf16=args.bf16,
                maxk=5 if args.nb_classes >= 5 else 1,
            )
            timestep = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(
                f"[{timestep}] Accuracy of the network on the {len(dataset_val)} val videos: {test_stats['acc1']:.1f}%"
            )
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                max_epoch = epoch
                if args.output_dir and args.save_ckpt:
                    utils_mamba.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        model_name="best",
                        model_ema=model_ema,
                    )

            print(
                f"Max accuracy: {max_accuracy:.2f}%" + "   Max Epoch: " + str(max_epoch)
            )
            if log_writer is not None:
                log_writer.update(val_acc1=test_stats["acc1"], head="perf", step=epoch)
                log_writer.update(val_acc5=test_stats["acc5"], head="perf", step=epoch)
                log_writer.update(val_loss=test_stats["loss"], head="perf", step=epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"val_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        if args.output_dir and utils_mamba.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    preds_file = os.path.join(args.output_dir, str(global_rank) + ".txt")
    if args.test_best:
        print("Auto testing the best model")
        args.eval = True
        utils_mamba.auto_load_model(
            args=args,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            model_ema=model_ema,
        )
    test_stats = final_test(
        data_loader_test,
        model,
        device,
        preds_file,
        amp_autocast,
        ds=args.enable_deepspeed,
        no_amp=args.no_amp,
        bf16=args.bf16,
        maxk=5 if args.nb_classes >= 5 else 1,
    )
    torch.distributed.barrier()
    if global_rank == 0:
        print("Start merging results...")
        final_top1, final_top5 = merge(args.output_dir, num_tasks)
        print(
            f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%"
        )
        log_stats = {"Final top-1": final_top1, "Final Top-5": final_top5}
        if args.output_dir and utils_mamba.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    opts, ds_init = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
