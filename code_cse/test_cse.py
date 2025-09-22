"""
Testing Script for CSE LightUNETR

This script evaluates trained LightUNETR and LightUNETR-L models on 
medical image segmentation test sets.

Supported Datasets:
- pancreas: Pancreas segmentation
- LA: Left Atrium segmentation  
- brats: Brain tumor segmentation
"""

import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import random
import math
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import test_3d_patch_fixmatch
from dataloaders.pancreas import StrongWeakPancreas, StrongWeakLA_PancreasStyle, StrongWeakBrats_PancreasStyle
from networks.build_network import select_model
from utils.test_3d_patch_fixmatch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pancreas', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='fixmatch_masking', help='exp_name')
parser.add_argument('--model', type=str, default='lightunetr', 
                    help='model_name: lightunetr (LightUNETR), lightunetr_large (LightUNETR-L)')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint to load')
args = parser.parse_args()

# Dataset setup
if args.dataset in ['pancreas', 'Pancreas', 'PANCREAS']:
    args.dataset = 'pancreas'
elif args.dataset in ['brats', 'BRATS', 'Brats', 'Brats19', 'brats19']:
    args.dataset = 'brats'
elif args.dataset in ['la', 'LA']:
    args.dataset = 'LA'
else:
    raise NotImplementedError("Dataset {} not supported, supported datasets are: pancreas, brats, la".format(args.dataset))

if args.dataset == 'LA':
    args.root_path = '/data/xyliu/UA-MT/data/'
    patch_size = (112, 112, 80)
    num_classes = 2
    stride_xy = 18
    stride_z = 4
elif args.dataset == 'pancreas':
    args.root_path = '/data/xyliu/codes'
    patch_size = (96, 96, 96)
    num_classes = 2
    stride_xy = 16
    stride_z = 4
elif args.dataset == 'brats':
    args.root_path = '/data/xyliu/brats2019'
    patch_size = (96, 96, 96)
    num_classes = 2
    stride_xy = 64
    stride_z = 64

emb_length = math.ceil(patch_size[0] / 32) * math.ceil(patch_size[1] / 32) * math.ceil(patch_size[2] / 32)
print("Patch size: {}, Embedding length: {}".format(patch_size, emb_length))

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ["OMP_NUM_THREADS"] = "8"
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
cudnn.benchmark = False
cudnn.deterministic = True

def create_test_loader():
    if args.dataset == 'LA':
        testset = StrongWeakLA_PancreasStyle(args.root_path, split='test', aug_times=5)
    elif args.dataset == 'pancreas':
        testset = StrongWeakPancreas(args.root_path, split='test', aug_times=5)
    elif args.dataset == 'brats':
        testset = StrongWeakBrats_PancreasStyle(args.root_path, split='test', aug_times=5)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    return test_loader

if __name__ == "__main__":
    # Setup logging
    log_dir = os.path.dirname(args.checkpoint)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, "test_log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("Arguments:")
    for arg in vars(args):
        logging.info("\t{}: {}".format(arg, getattr(args, arg)))
    logging.info("\n")

    # Model
    model = select_model(args, num_classes, emb_length, patch_size)
    checkpoint = torch.load(args.checkpoint, map_location='cuda:0')
    model.load_state_dict(checkpoint, strict=True)
    model.cuda()
    model.eval()

    # Prepare test image list
    if args.dataset == 'LA':
        with open(args.root_path + '/test.list', 'r') as f:
            test_image_list = f.readlines()
        test_image_list = ["/data/xyliu/UA-MT/data/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in test_image_list]
    else:
        with open(args.root_path + '/test.list', 'r') as f:
            test_image_list = f.readlines()
        test_image_list = [args.root_path + '/data/' + item.replace('\n', '') + '.h5' for item in test_image_list]

    # Test save path
    test_save_path = os.path.join(log_dir, "{}_predictions".format(args.model))
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    # Run test
    logging.info("Testing...")
    avg_metric = test_all_case(
        model, test_image_list, num_classes=num_classes,
        patch_size=patch_size, stride_xy=stride_xy, stride_z=stride_z,
        save_result=True, test_save_path=test_save_path,
        metric_detail=1, nms=1
    )
    logging.info("Test finished. Average metric: {}".format(avg_metric))
