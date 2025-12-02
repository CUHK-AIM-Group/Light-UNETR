# [TPAMI' 2025] Harnessing Lightweight Transformer with Contextual Synergic Enhancement for Efficient 3D Medical Image Segmentation

**Authors:** Xinyu Liu, Zhen Chen, Wuyang Li, Chenxin Li, Yixuan Yuan

## Overview

This repository contains the implementation of our Light-UNETR for efficient 3D medical image segmentation with contextual synergic enhancement (CSE).

## Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU
- torch version 2.4.1

### Setup

**Clone the repository:**
```bash
git clone https://github.com/CUHK-AIM-Group/code_cse.git
cd code_cse
```

**Create conda environment:**
```bash
conda create -n lightunetr python=3.12
conda activate lightunetr
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

##  Supported Datasets

- **LA (Left Atrium)**: Left atrium segmentation from cardiac MRI
- **Pancreas-CT**: Pancreas segmentation from abdominal CT scans  
- **BraTS 2019**: Brain tumor segmentation from multimodal MRI

## 📁 Data Preparation

### Download Links

- **LA**: Download from [LA data](https://github.com/yulequan/UA-MT/tree/master/data)
- **Pancreas**: Download from [Pancreas](https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT) and preprocess following [here](https://github.com/grant-jpg/FUSSNet)
- **BraTS 2019**: Download from [BraTS 2019](https://github.com/HiLab-git/SSL4MIS/tree/master/data/BraTS2019)

### Data Structure

After preprocessing, organize your data in the following structure:

```
datasets/
├── brats/
│   ├── data/
│   │   ├── BraTS19_2013_0_1.h5
│   │   └── ...
│   ├── test.list
│   ├── train.list
│   ├── train_lab25.list
│   └── train_unlab25.list
├── la/
│   ├── 2018LA_Seg_Training Set/
│   │   ├── 0RZDK210BSMWAA6467LU/
│   │   │   └── mri_norm2.h5
│   │   └── ...
│   ├── test.list
│   ├── train.list
│   ├── train_lab16.list
│   ├── train_lab4.list
│   ├── train_lab8.list
│   ├── train_unlab16.list
│   ├── train_unlab4.list
│   └── train_unlab8.list
└── pancreas/
    ├── data/
    │   ├── data0001.h5
    │   └── ...
    ├── test.list
    ├── train.list
    ├── train_lab12.list
    ├── train_lab6.list
    ├── train_unlab12.list
    └── train_unlab6.list
```


## 🎯 Semi-Supervised Learning with CSE

### 🚀 Training Commands

**Semi-supervised training with different label numbers:**

```bash

# LA dataset with 4 labeled samples  
python ./code_cse/train_cse_withval.py --dataset LA --exp train_cse --model lightunetr --labelnum 4 --gpu 0

# LA dataset with 8 labeled samples  
python ./code_cse/train_cse_withval.py --dataset LA --exp train_cse --model lightunetr --labelnum 8 --gpu 0

# Pancreas dataset with 6 labeled samples
python ./code_cse/train_cse_withval.py --dataset pancreas --exp train_cse --model lightunetr --labelnum 6 --gpu 0

# Pancreas dataset with 12 labeled samples
python ./code_cse/train_cse_withval.py --dataset pancreas --exp train_cse --model lightunetr --labelnum 12 --gpu 0

# BraTS dataset with 25 labeled samples
python ./code_cse/train_cse_withval.py --dataset brats --exp train_cse --model lightunetr --labelnum 25 --gpu 0
```

**Fully supervised training (upper bound):**

```bash
# LightUNETR models
python ./code_cse/train_supervised.py --dataset LA --exp train_supervised --model lightunetr --gpu 0
python ./code_cse/train_supervised.py --dataset brats --exp train_supervised --model lightunetr --gpu 0
python ./code_cse/train_supervised.py --dataset pancreas --exp train_supervised --model lightunetr --gpu 0

# LightUNETR-Large models  
python ./code_cse/train_supervised.py --dataset LA --exp train_supervised --model lightunetr_large --gpu 1
python ./code_cse/train_supervised.py --dataset brats --exp train_supervised --model lightunetr_large --gpu 2
python ./code_cse/train_supervised.py --dataset pancreas --exp train_supervised --model lightunetr_large --gpu 3
```

### ⚙️ Arguments

- `--dataset`: Choose from `pancreas`, `LA`, or `brats`
- `--exp`: Experiment name for logging and checkpoints
- `--model`: Model architecture (`lightunetr` or `lightunetr_large`)
- `--labelnum`: Number of labeled samples for semi-supervised learning
- `--gpu`: GPU device ID

## �📊 Fully Supervised Learning

For fully supervised training on other datasets, please refer to `./fullysup`.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Model Zoo

Pre-trained models are available on Hugging Face:

| Dataset | Labeled num | Model | Download Link |
|---------|-------------|-------|---------------|
| BraTS 2019 | 25 labels | LightUNETR | [lightunetr_best_model_brats_25lab.pth](https://huggingface.co/jeffrey423/lightunetr-models/blob/main/lightunetr_best_model_brats_25lab.pth) |
| LA | 4 labels | LightUNETR | [lightunetr_best_model_la_4lab.pth](https://huggingface.co/jeffrey423/lightunetr-models/blob/main/lightunetr_best_model_la_4lab.pth) |
| LA | 8 labels | LightUNETR | [lightunetr_best_model_la_8lab.pth](https://huggingface.co/jeffrey423/lightunetr-models/blob/main/lightunetr_best_model_la_8lab.pth) |
| Pancreas | 6 labels | LightUNETR | [lightunetr_best_model_pancreas_6lab.pth](https://huggingface.co/jeffrey423/lightunetr-models/blob/main/lightunetr_best_model_pancreas_6lab.pth) |
| Pancreas | 12 labels | LightUNETR | [lightunetr_best_model_pancreas_12lab.pth](https://huggingface.co/jeffrey423/lightunetr-models/blob/main/lightunetr_best_model_pancreas_12lab.pth) |

### Usage

Download the desired model and use it with the test script:

```bash
# Example: Test BraTS model
python test_cse.py --dataset brats --model lightunetr --checkpoint lightunetr_best_model_brats_25lab.pth --gpu 0
```

## 📖 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{liu2025harnessing,
  title={Harnessing Lightweight Transformer with Contextual Synergic Enhancement for Efficient 3D Medical Image Segmentation},
  author={Liu, Xinyu and Chen, Zhen and Li, Wuyang and Li, Chenxin and Yuan, Yixuan},
  year={2025}
}
```

