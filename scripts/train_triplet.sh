# /home/syangcw/Surgformer/pretrain_params/timesformer_base_patch16_224_K400.pyth
# /home/syangcw/Surgformer/pretrain_params/mae_pretrain_vit_base.pth
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
--nproc_per_node=2 \
--master_port 12324 \
downstream_triplet/run_triplet_training.py \
--batch_size 8 \
--epochs 30 \
--save_ckpt_freq 10 \
--model  surgformer_HTA_KCA \
--pretrained_path /home/syangcw/Surgformer/pretrain_params/timesformer_base_patch16_224_K400.pyth \
--lr 5e-4 \
--layer_decay 0.75 \
--warmup_epochs 3 \
--data_path /home/syangcw/datasets/CholecT50 \
--eval_data_path /home/syangcw/datasets/CholecT50 \
--nb_classes 100 \
--data_strategy online \
--output_mode key_frame \
--num_frames 16 \
--sampling_rate 4 \
--data_set CholecT50 \
--data_fps 1fps \
--output_dir /home/syangcw/Surgformer/results/CholecT50/RDV \
--log_dir /home/syangcw/Surgformer/results/CholecT50/RDV \
--num_workers 10 \
--dist_eval \
--enable_deepspeed \
--no_auto_resume