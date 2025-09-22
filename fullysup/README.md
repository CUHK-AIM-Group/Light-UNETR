# Fully Supervised Learning

This directory contains the implementation for fully supervised training on additional datasets using Light-UNETR and Light-UNETR-L models.

## 📁 Data Structure

Download AbdomenCT-1K dataset from [here](https://github.com/JunMa11/AbdomenCT-1K) (Three parts in total).

Download Head and Neck Tumor dataset from [here](https://zenodo.org/records/11199559).

Each dataset should be organized following this structure:

```
datasets/
├── AbdomenCT-1K/
│   ├── imagesTr/
│   │   ├── Case_00001_0000.nii.gz
│   │   ├── Case_00002_0000.nii.gz
│   │   └── ...
│   └── labelsTr/
│       ├── Case_00001_0000.nii.gz
│       ├── Case_00002_0000.nii.gz
│       └── ...
└── HeadNeck/
    ├── imagesTr/
    │   ├── 002_0000.nii.gz
    │   ├── 003_0000.nii.gz
    │   └── ...
    └── labelsTr/
        ├── 002_0000.nii.gz
        ├── 003_0000.nii.gz
        └── ...
```


## ⚙️ Configuration

### Training Configuration

Each dataset has its own configuration file:

- `config_abdomen_lightunetr.yml`: Configuration for AbdomenCT-1K with LightUNETR
- `config_abdomen_lightunetr_large.yml`: Configuration for AbdomenCT-1K with LightUNETR-Large  
- `config_headneck_lightunetr.yml`: Configuration for Head and Neck dataset

## 🔧 Usage

1. **Prepare your data**: Ensure your datasets are organized according to the paths specified in the configuration files

2. **Update configuration**: Modify the `data_root` paths in the respective config files to match your data location

3. **Run training**: Execute the appropriate training script for your dataset and model choice

## 🚀 Training Scripts

### Multi-GPU Training

All training scripts support distributed training with:
- 2 GPUs by default (`CUDA_VISIBLE_DEVICES=0,1`)
- PyTorch distributed training via `torchrun`
- NCCL optimizations for multi-GPU communication

### AbdomenCT-1K Dataset

**Light-UNETR (Standard Model):**
```bash
# Multi-GPU training with 2 GPUs
bash train_abdomenct1k_lightunetr.sh
```

**Light-UNETR-Large (Large Model):**
```bash
# Multi-GPU training with 2 GPUs  
bash train_abdomenct1k_lightunetr_large.sh
```

### Head and Neck Tumor Dataset

Please modify the --model args within the bash script before training.

```bash
# Multi-GPU training with 2 GPUs
bash train_headneck.sh
```

## 📁 Project Structure

```
fullysup/
├── src/          # Implementation of the models
├── main_abdomenct1k_lightunetr.py    # AbdomenCT-1K training (LightUNETR)
├── main_abdomenct1k_lightunetr_large.py # AbdomenCT-1K training (LightUNETR-Large)
├── main_headneck.py              # Head & Neck training
├── config_abdomen_lightunetr.yml     # AbdomenCT-1K config (LightUNETR)
├── config_abdomen_lightunetr_large.yml # AbdomenCT-1K config (LightUNETR-Large)
├── config_headneck_lightunetr.yml    # Head & Neck config
├── train_abdomenct1k_lightunetr.sh   # AbdomenCT-1K training script (LightUNETR)
├── train_abdomenct1k_lightunetr_large.sh # AbdomenCT-1K training script (LightUNETR-Large)
└── train_headneck.sh             # Head & Neck training script
```


## 📋 Requirements

The same requirements as the main project apply. See the main `requirements.txt` for dependencies.

## 🔗 Back to Main Project

For semi-supervised learning experiments, please refer to the [main README](../README.md).