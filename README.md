# Surgformer: Surgical Transformer with Hierarchical Temporal Attention for Surgical Phase Recognition


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![GitHub last commit](https://img.shields.io/github/last-commit/isyangshu/Surgformer?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/isyangshu/Surgformer?style=flat-square)
![GitHub stars](https://img.shields.io/github/stars/isyangshu/Surgformer?style=flat-square)
[![Arxiv Page](https://img.shields.io/badge/Arxiv-2403.06800-red?style=flat-square)](https://arxiv.org/pdf/2403.06800.pdf)


## Abstract

> Existing state-of-the-art methods for surgical phase recognition either rely on the extraction of spatial-temporal features at short-range temporal resolution or adopt the sequential extraction of the spatial and temporal features across the entire temporal resolution. However, these methods have limitations in modeling spatial-temporal dependency and addressing spatial-temporal redundancy: 1) These methods fail to effectively model spatial-temporal dependency, due to the lack of long-range information or joint spatial-temporal modeling. 2) These methods utilize dense spatial features across the entire temporal resolution, resulting in significant spatial-temporal redundancy. In this paper, we propose the Surgical Transformer (Surgformer) to address the issues of spatial-temporal modeling and redundancy in an end-to-end manner, which employs divided spatial-temporal attention and takes a limited set of sparse frames as input. Moreover, we propose a novel Hierarchical Temporal Attention (HTA) to capture both global and local information within varied temporal resolutions from a target frame-centric perspective. Distinct from conventional temporal attention that primarily emphasizes dense long-range similarity, HTA not only captures long-term information but also considers local latent consistency among informative frames. HTA then employs pyramid feature aggregation to effectively utilize temporal information across diverse temporal resolutions, thereby enhancing the overall temporal representation. Extensive experiments on two challenging benchmark datasets verify that our proposed Surgformer performs favorably against the state-of-the-art methods.

## NOTES

**2024-04-30**: We released the full version of Surgformer.

## Installation
* Environment: CUDA 11.8 / Python 3.10
* Create a virtual environment
```shell
> conda create -n Surgformer python=3.10 -y
> conda activate Surgformer
```
* Install Pytorch 2.0.1
```shell
> pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
> pip install packaging
```
* Other requirements
```shell
> pip install timm==0.4.12
> pip install deepspeed==0.14.0
> pip install tensorboardx
```
For Hybird Model
* Install causal-conv1d
```shell
> pip install causal-conv1d==1.1.1
```
* Install Mamba
```shell
> git clone git@github.com:isyangshu/Surgformer.git
> cd mamba
> pip install .
```

## Repository Details

<!-- * `csv`:  Complete Cbioportal files, including the features path and data splits with 5-fold cross-validation. 
* `datasets`: The code for Dataset, you can just replace the path in Line-25. -->
* `mamba`: including the original Mamba, Bi-Mamba from Vim and our proposed SRMamba.
* `models`: Support the following model:
  - [Mean pooling](https://github.com/isyangshu/MambaMIL/tree/main/models/Mean_Max_MIL.py) 
  - [Max pooling](https://github.com/isyangshu/MambaMIL/tree/main/models/Mean_Max_MIL.py) 
  - [ABMIL](https://github.com/isyangshu/MambaMIL/tree/main/models/ABMIL.py)  
  - [TransMIL](https://github.com/isyangshu/MambaMIL/tree/main/models/TransMIL.py)
  - [S4MIL](https://github.com/isyangshu/MambaMIL/tree/main/models/S4MIL.py)
  - [Our MambaMIL](https://github.com/isyangshu/MambaMIL/tree/main/models/MambaMIL.py)
<!-- * `results`: the results on 12 datasets, including BLCA BRCA CESC CRC GBMLGG KIRC LIHC LUAD LUSC PAAD SARC UCEC. -->
* `splits`: Splits for reproducation.
* `train_scripts`: We provide train scripts for cancer subtyping and survival prediction.

## How to Train
### Prepare your data
1. Download diagnostic WSIs from [TCGA](https://portal.gdc.cancer.gov/) and [BRACS](https://www.bracs.icar.cnr.it/) 
2. Use the WSI processing tool provided by [CLAM](https://github.com/mahmoodlab/CLAM) to extract resnet-50 and [PLIP](https://github.com/PathologyFoundation/plip/tree/main) pretrained feature for each 512 $\times$ 512 patch (20x), which we then save as `.pt` files for each WSI. So, we get one `pt_files` folder storing `.pt` files for all WSIs of one study.

The final structure of datasets should be as following:
```bash
DATA_ROOT_DIR/
    └──pt_files/
        └──resnet50/
            ├── slide_1.pt
            ├── slide_2.pt
            └── ...
        └──plip/
            ├── slide_1.pt
            ├── slide_2.pt
            └── ...
        └──others/
            ├── slide_1.pt
            ├── slide_2.pt
            └── ...
```
### Survival Prediction
We provide train scripts for survival prediction [ALL_512_surivial_k_fold.sh](https://github.com/isyangshu/MambaMIL/tree/main/train_scripts/ALL_512_survival_k_fold.sh).

Below are the supported models and datasets:

```bash
model_names='max_mil mean_mil att_mil trans_mil s4_mil mamba_mil'
backbones="resnet50 plip"
cancers='BLCA BRCA COADREAD KIRC KIRP LUAD STAD UCEC'
```

run the following code for training

```shell
sh ./train_scripts/ALL_512_surivial_k_fold.sh
```

### Cancer Subtyping
We provide train scripts for TCGA NSCLC cancer subtyping [LUAD_LUSC_512_subtyping.sh](https://github.com/isyangshu/MambaMIL/tree/main/train_scripts/LUAD_LUSC_512_subtyping.sh) and BReAst Carcinoma Subtyping [BRACS.sh](https://github.com/isyangshu/MambaMIL/tree/main/train_scripts/train_scripts/BRACS.sh).

Below are the supported models:

```bash
model_names='max_mil mean_mil att_mil trans_mil s4_mil mamba_mil'
backbones="resnet50 plip"
```

run the following code for training TCGA NSCLC cancer subtyping 

```shell
sh ./train_scripts/LUAD_LUSC_512_subtyping.sh
```
run the following code for training BReAst Carcinoma Subtyping 

```shell
sh ./train_scripts/BRACS.sh
```

## Acknowledgements
Huge thanks to the authors of following open-source projects:
- [CLAM](https://github.com/mahmoodlab/CLAM)



## License & Citation 
If you find our work useful in your research, please consider citing our paper at:

```text
@article{yang2024mambamil,
  title={MambaMIL: Enhancing Long Sequence Modeling with Sequence Reordering in Computational Pathology},
  author={Yang, Shu and Wang, Yihui and Chen, Hao},
  journal={arXiv preprint arXiv:2403.06800},
  year={2024}
}
```
This code is available for non-commercial academic purposes. If you have any question, feel free to email [Shu YANG](yangshu@connect.ust.hk) and [Yihui WANG](ywangrm@connect.ust.hk).
# Surgformer
