"""
Supervised Training Script for CSE LightUNETR

This script implements fully supervised learning for training LightUNETR and 
LightUNETR-L models on medical image segmentation tasks.

Supported Datasets:
- pancreas: Pancreas segmentation
- LA: Left Atrium segmentation  
- brats: Brain tumor segmentation
"""

import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import random
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from utils import losses, ramps, test_3d_patch_fixmatch
from dataloaders.pancreas import StrongWeakPancreas, StrongWeakLA_PancreasStyle, StrongWeakBrats_PancreasStyle
#
from networks.build_network import select_model, save_model_info
#
from utils.test_3d_patch_fixmatch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pancreas', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='fully_supervised', help='exp_name')
parser.add_argument('--model', type=str, default='lightunetr', 
                    help='model_name: lightunetr (LightUNETR), lightunetr_large (LightUNETR-L)')
parser.add_argument('--max_iteration', type=int,  default=30000, help='maximum iteration to train')
parser.add_argument('--labeled_bs', type=int, default=1, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='7', help='GPU to use')
parser.add_argument(
    "--conf_thresh",
    type=float,
    default=0.8,
    help="confidence threshold for using pseudo-labels",
)
parser.add_argument('--ema', action='store_true') # Whether to use EMA model
# optimizer
parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--weight_decay', type=float, default=0.0001)
# lr schedule
parser.add_argument('--lr_schedule', type=str, default='cosine', choices=['cosine', 'multistep'])
parser.add_argument('--lr_warmup', type=int, default=0)
# others

args = parser.parse_args()

# make upper if la, LA; Make lower if pancreas, Pancreas, PANCREAS, brats, BRATS, Brats, Brats19, brats19
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


snapshot_path = "./experiments/lightunetr/{}_fullysupervised_seed{}/{}_{}_bs{}_labbs{}".format(args.dataset, args.seed, args.exp, args.model, args.batch_size, args.labeled_bs)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ["OMP_NUM_THREADS"] = "8"
max_iterations = args.max_iteration
base_lr = args.base_lr

# if args.deterministic:
cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    else:
        print('Warning: dir exists! renaming the folder to {} + timestamp'.format(snapshot_path))
        snapshot_path = snapshot_path + '_{}'.format(time.strftime("%Y%m%d-%H%M%S"))
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    # log args in a friendly format
    logging.info("Arguments:")
    for arg in vars(args):
        logging.info("\t{}: {}".format(arg, getattr(args, arg)))
    logging.info("\n")
    
    model = select_model(args, num_classes, emb_length, patch_size)
    save_model_info(model, patch_size, snapshot_path)

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)

    def create_dataloader():
        if args.dataset == 'LA':
            train_labset = StrongWeakLA_PancreasStyle(args.root_path, split='train', aug_times = 5)
        elif args.dataset == 'pancreas':
            train_labset = StrongWeakPancreas(args.root_path, split='train', aug_times = 5)
        elif args.dataset == 'brats':
            train_labset = StrongWeakBrats_PancreasStyle(args.root_path, split='train', aug_times = 5)

        if args.dataset == 'LA':
            testset = StrongWeakLA_PancreasStyle(args.root_path, split='test', aug_times = 5)
        elif args.dataset == 'pancreas':
            testset = StrongWeakPancreas(args.root_path, split='test', aug_times = 5)
        elif args.dataset == 'brats':
            testset = StrongWeakBrats_PancreasStyle(args.root_path, split='test', aug_times = 5)
        
        train_loader_labunlab = DataLoader(train_labset, batch_size=args.batch_size-args.labeled_bs, 
                                       shuffle=True, num_workers=0, 
                                       drop_last=False)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

        logging.info("{} iterations for supervised training per epoch.".format(len(train_loader_labunlab)))
        logging.info("{} samples for test.\n".format(len(test_loader)))
        return train_loader_labunlab, test_loader

    train_loader_labunlab, test_loader = create_dataloader()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(train_loader_labunlab)))
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(train_loader_labunlab) + 1
    lr_ = base_lr
    
    for epoch_num in range(max_epoch):
        for step, labeled_batch in enumerate(train_loader_labunlab):
            start = time.time()
            weak_batch, label_batch = labeled_batch['image_weak'].cuda(), labeled_batch['label'].cuda()

            weak_batch, label_batch = (
                weak_batch.cuda(),
                label_batch.cuda(),
            )
            
            model.train()
            outputs_weak = model(weak_batch)
            # if tuple use the first element
            if isinstance(outputs_weak, tuple):
                outputs_weak = outputs_weak[0]
            outputs_weak_soft = torch.softmax(outputs_weak, dim=1)

            assert outputs_weak.shape[-3:] == outputs_weak.shape[-3:]
            assert label_batch.shape[-3:] == label_batch.shape[-3:]
            sup_loss = ce_loss(outputs_weak, label_batch[:].long(),) + dice_loss(outputs_weak_soft, label_batch.unsqueeze(1))

            loss = sup_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            writer.add_scalar('1_Loss/sup_loss', sup_loss, iter_num)
            logging.info('iteration %d : loss : %03f, loss_sup: %03f' % (iter_num, loss.item(), sup_loss.item()))
            
            # change lr
            if args.lr_schedule == 'multistep':
                if iter_num % 2500 == 0:
                    lr_ = base_lr * 0.1 ** (iter_num // 2500)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
            elif args.lr_schedule == 'cosine':
                warmup_iterations = 500 # Hard code it to 500 now. May change it later.
                if iter_num >= warmup_iterations:
                    lr_ = base_lr * (1 + math.cos(math.pi * (iter_num - warmup_iterations) / (max_iterations - warmup_iterations))) / 2
                else:
                    warmup_factor = iter_num / warmup_iterations
                    lr_ = base_lr * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_


            if iter_num % 200 == 0:
                model.eval()
                if args.dataset == 'LA':
                    valeq = test_3d_patch_fixmatch.var_all_case_LA
                elif args.dataset == 'pancreas':
                    valeq = test_3d_patch_fixmatch.var_all_case_Pancreas
                elif args.dataset == 'brats':
                    valeq = test_3d_patch_fixmatch.var_all_case_Brats
                dice_sample = valeq(model,
                                    num_classes=num_classes,
                                    patch_size=patch_size, 
                                    stride_xy=stride_xy, 
                                    stride_z=stride_z)
                logging.info("Dice score at {}-th iteration is {}".format(iter_num, round(dice_sample, 4)))
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
                    logging.info("Dice score of best model is {}".format(best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_best_path))

                    # do testing
                    test_save_path = snapshot_path + "/{}_predictions/".format(args.model)

                    if not os.path.exists(test_save_path):
                        os.makedirs(test_save_path)
                    with open(args.root_path + '/test.list', 'r') as f:
                        test_image_list = f.readlines()
                    if args.dataset != 'LA':
                        test_image_list = [args.root_path + '/data/' + item.replace('\n', '') + '.h5' for item in test_image_list]
                    else:
                        test_image_list = [args.root_path + "/2018LA_Seg_Training Set/" + item.replace('\n', '') + "/mri_norm2.h5" for item in test_image_list]
                    avg_metric = test_all_case(model, test_image_list, num_classes=num_classes,
                                        patch_size=patch_size, stride_xy=stride_xy, stride_z=stride_z,
                                        save_result=True, test_save_path=test_save_path,
                                        metric_detail=1, nms=1)
                writer.add_scalar('4_Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('4_Var_dice/Best_dice', best_dice, iter_num)
                model.train()


            if iter_num >= max_iterations:
                break

        if iter_num >= max_iterations:
            logging.info("Finish training at iteration {}".format(iter_num))
            logging.info("Best average metric is {}".format(avg_metric))
            break

    writer.close()