# [MICCAI 2024] Surgformer: Surgical Transformer with Hierarchical Temporal Attention for Surgical Phase Recognition


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<!-- ![GitHub last commit](https://img.shields.io/github/last-commit/isyangshu/Surgformer?style=flat-square) -->
<!-- ![GitHub issues](https://img.shields.io/github/issues/isyangshu/Surgformer?style=flat-square) -->
<!-- ![GitHub stars](https://img.shields.io/github/stars/isyangshu/Surgformer?style=flat-square) -->
<!-- [![Arxiv Page](https://img.shields.io/badge/Arxiv-2403.06800-red?style=flat-square)](https://arxiv.org/pdf/2403.06800.pdf) -->


## Abstract

> Existing state-of-the-art methods for surgical phase recognition either rely on the extraction of spatial-temporal features at short-range temporal resolution or adopt the sequential extraction of the spatial and temporal features across the entire temporal resolution. However, these methods have limitations in modeling spatial-temporal dependency and addressing spatial-temporal redundancy: 1) These methods fail to effectively model spatial-temporal dependency, due to the lack of long-range information or joint spatial-temporal modeling. 2) These methods utilize dense spatial features across the entire temporal resolution, resulting in significant spatial-temporal redundancy. In this paper, we propose the Surgical Transformer (Surgformer) to address the issues of spatial-temporal modeling and redundancy in an end-to-end manner, which employs divided spatial-temporal attention and takes a limited set of sparse frames as input. Moreover, we propose a novel Hierarchical Temporal Attention (HTA) to capture both global and local information within varied temporal resolutions from a target frame-centric perspective. Distinct from conventional temporal attention that primarily emphasizes dense long-range similarity, HTA not only captures long-term information but also considers local latent consistency among informative frames. HTA then employs pyramid feature aggregation to effectively utilize temporal information across diverse temporal resolutions, thereby enhancing the overall temporal representation. Extensive experiments on two challenging benchmark datasets verify that our proposed Surgformer performs favorably against the state-of-the-art methods.

## NOTES

**2024-10-01**: Without any adjustment and surgical parameters, we win the 2nd in OSS challenge and 3rd in PhaKIR.

**2024-06-26**: We release the full version of Surgformer.

**2024-05-14**: Our paper is early accepted for MICCAI 2024.

## Installation
* Environment: CUDA 11.4 / Python 3.8
* Device: NVIDIA GeForce RTX 3090
* Create a virtual environment
```shell
> conda env create -f Surgformer.yml
```

## How to Train
### Prepare your data
1. Download raw video data from [Cholec80](https://camma.unistra.fr/datasets/) and [AutoLaparo](https://autolaparo.github.io/);
> You need to fill the request form to access both datasets.
2. Use the pre-processing tools provided by us to extract frames and generate pickle files for training.
```python
# Extract frames form raw videos
python datasets/data_preprosses/extract_frames_ch80.py
python datasets/data_preprosses/extract_frames_autolaparo.py

# Generate .pkl for training
python datasets/data_preprosses/generate_labels_ch80.py
python datasets/data_preprosses/generate_labels_autolaparo.py

# Note that you can change the size of each frame to reduce I/O load.
```
3. You can also use the cutting tool provided by [TMRNet](https://github.com/YuemingJin/TMRNet) to cut black margin for surgical videos in Cholec80, which may help to get better performance.
```python
# Cut black margin
python datasets/data_preprosses/frame_cutmargin.py
```

The final structure of datasets should be as following:

Note that you can change `fps` in the Step.2 to generate more frames.
```bash
data/
    └──Cholec80/
        └──frames/
            └──video01
                ├──00000.png
                ├──00001.png
                └──...
            ├──...    
            └──video80
        └──frames_cutmargin/
        └──labels/
            └──train
                ├── 1pstrain.pickle
                └── ...
            └──test
                ├── 1psval_test.pickle
                └── ...
    └──AutoLaparo/
        └──frames/
            └──01
                ├──00000.png
                ├──00001.png
                └──...
            ├──...    
            └──21
        └──labels_pkl/
            └──train
            ├──val
            └──test
```
### Pretrained Parameters

We use the parameters of [TimeSformer](https://github.com/facebookresearch/TimeSformer) trained on [K400 dataset](https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=0) with frames 8 and spatial size 224 as initialization parameters.

> You can try to use [ImageNet(MAE)](https://huggingface.co/timm/vit_base_patch16_224.mae), which may help to get better performance.

### Training
We provide the script for training [train_phase.sh](https://github.com/isyangshu/Surgformer/blob/master/scripts/train_phase.sh).

run the following code for training

```shell
sh scripts/train.sh
```
> You need to modify **data_path**, **eval_data_path**, **output_dir** and **log_dir** according to your own setting.

> Optional settings \
> **Model**: surgformer_base surgformer_HTA surgformer_HTA_KCA \
> **Dataset**: Cholec80 AutoLaparo

### Test
> Currently, the test and evaluation codes we provide are only applicable to two-GPU inference.

1. run the following code for testing, and get **0.txt** and **1.txt**;

```shell
sh scripts/test.sh
```

2. Merge the files and generate separate txt file for each video;
```python
python datasets/convert_results/convert_cholec80.py
python datasets/convert_results/convert_autolaparo.py
```

3. Use [Matlab Evaluation Code](https://github.com/isyangshu/Surgformer/tree/master/evaluation_matlab) to compute metrics;

## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE/tree/main?tab=readme-ov-file)
- [TMRNet](https://github.com/YuemingJin/TMRNet)
- [SelfSupSurg](https://github.com/CAMMA-public/SelfSupSurg)
- [TimeSformer](https://github.com/facebookresearch/TimeSformer)

## License & Citation 
If you find our work useful in your research, please consider citing our paper at:

```text
coming soon
```
This code is available for non-commercial academic purposes. If you have any question, feel free to email [Shu YANG](syangcw@connect.ust.hk).
