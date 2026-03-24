"""
Semi-supervised Training Script for CSE LightUNETR

This script implements semi-supervised learning
for training LightUNETR and LightUNETR-L models on medical image segmentation tasks.

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
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from utils import losses, ramps, test_3d_patch_fixmatch
from dataloaders.pancreas import StrongWeakPancreas, StrongWeakLA_PancreasStyle, StrongWeakBrats_PancreasStyle
from dataloaders.masking import Masking
#
from networks.build_network import select_model, save_model_info
#
from utils.BCP_utils import context_mask, mix_loss, parameter_sharing, attention_mask, soft_attention_mask
from utils.test_3d_patch_fixmatch import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='LA', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='train_cse', help='exp_name')
parser.add_argument('--model', type=str, default='lightunetr', 
                    help='model_name: lightunetr (LightUNETR), lightunetr_large (LightUNETR-L)')
parser.add_argument('--max_iteration', type=int,  default=15000, help='maximum iteration to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=4, help='trained samples')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
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
parser.add_argument('--ada_u_weight', type=float, default=1.0, help='loss weight for unlabeled data')
parser.add_argument('--mask_order', type=str, default='descend', choices=['ascend', 'descend'])
parser.add_argument('--mix_unlab_weight', type=float, default=0.5)
##
parser.add_argument('--mask_cont2', type=str,  default='mask', choices=['mask', 'depthwise_mask', 'voxelwise_mask', 'none', 'center_mask'])
parser.add_argument('--mask_block_size', default=16, type=int)
parser.add_argument('--mask_ratio', default=0.5, type=float)
parser.add_argument('--mask_color_jitter_s', default=0, type=float) # 0.2
parser.add_argument('--mask_color_jitter_p', default=0, type=float) # 0.2
parser.add_argument('--mask_blur', default=False, type=bool) # True

parser.add_argument('--norm', type=str, default='bn', choices=['bn', 'gn'])
parser.add_argument('--jitter', type=int, default=0, help='How much to jitter the mask range')
parser.add_argument('--attn_mask_correct', type=int, default=3, help='Whether to correct masking', choices=[1, 2, 3])
parser.add_argument('--attn_mask_method', type=str, default='soft', choices=['soft', 'hard'])
parser.add_argument('--thresh_method', type=str, default='patch_wise', choices=['patch_wise', 'none'])

# lambda
parser.add_argument('--lambda_mix_loss', type=float, default=4.0, help='loss weight for mix loss')

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
    assert args.labelnum in [4, 8, 16, 32], "For LA dataset, labelnum should be 4 or 8, but got {}".format(args.labelnum)
elif args.dataset == 'pancreas':
    args.root_path = '/data/xyliu/codes'
    patch_size = (96, 96, 96)
    num_classes = 2
    stride_xy = 16
    stride_z = 4
    assert args.labelnum in [6, 12, 24, 48], "For Pancreas dataset, labelnum should be 4 or 8, but got {}".format(args.labelnum)
elif args.dataset == 'brats':
    args.root_path = '/data/xyliu/brats2019'
    patch_size = (96, 96, 96)
    num_classes = 2
    stride_xy = 64
    stride_z = 64
    assert args.labelnum in [25, 50, 100], "For Brats dataset, labelnum should be 25, but got {}".format(args.labelnum)

emb_length = math.ceil(patch_size[0] / 32) * math.ceil(patch_size[1] / 32) * math.ceil(patch_size[2] / 32) # special for the position embedding length
print("Patch size: {}, Embedding length: {}".format(patch_size, emb_length))


snapshot_path = "./experiments/lightunetr/{}_bs{}_seed{}/{}_{}_bs{}_labbs{}".format(args.dataset, args.labelnum, args.seed, args.exp, args.model, args.batch_size, args.labeled_bs)

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


def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = 0.99
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def get_pseudo_label(out, thres=0.5, nms=0):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).type(torch.int64)
    masks = masks[:, 1, :, :].contiguous()
    if nms == 1:
        masks = LargestCC_pancreas(masks)
    return masks

from skimage.measure import label
def LargestCC_pancreas(segmentation):
    N = segmentation.shape[0]
    batch_list = []
    for n in range(N):
        n_prob = segmentation[n].detach().cpu().numpy()
        labels = label(n_prob)
        if labels.max() != 0:
            largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        else:
            largestCC = n_prob
        batch_list.append(largestCC)
    batch_list = np.array(batch_list)
    return torch.Tensor(batch_list).cuda()

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

    labelnum = args.labelnum
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)

    def create_dataloader():
        if labelnum == 6:
            train_labset = StrongWeakPancreas(args.root_path, split='lab6', aug_times = 5)
            train_unlabset = StrongWeakPancreas(args.root_path, split='unlab6_noval', aug_times = 5)
        elif labelnum == 12:
            train_labset = StrongWeakPancreas(args.root_path, split='lab12', aug_times = 5)
            train_unlabset = StrongWeakPancreas(args.root_path, split='unlab12_noval', aug_times = 5)
        elif labelnum == 4:
            train_labset = StrongWeakLA_PancreasStyle(args.root_path, split='lab4', aug_times = 5)
            train_unlabset = StrongWeakLA_PancreasStyle(args.root_path, split='unlab4_noval', aug_times = 5)
        elif labelnum == 8:
            train_labset = StrongWeakLA_PancreasStyle(args.root_path, split='lab8', aug_times = 5)
            train_unlabset = StrongWeakLA_PancreasStyle(args.root_path, split='unlab8_noval', aug_times = 5)
        elif labelnum == 16:
            train_labset = StrongWeakLA_PancreasStyle(args.root_path, split='lab16', aug_times = 5)
            train_unlabset = StrongWeakLA_PancreasStyle(args.root_path, split='unlab16', aug_times = 5)
        elif labelnum == 32:
            train_labset = StrongWeakLA_PancreasStyle(args.root_path, split='lab32', aug_times = 5)
            train_unlabset = StrongWeakLA_PancreasStyle(args.root_path, split='unlab32', aug_times = 5)
        elif labelnum == 25:
            train_labset = StrongWeakBrats_PancreasStyle(args.root_path, split='lab25', aug_times = 5)
            train_unlabset = StrongWeakBrats_PancreasStyle(args.root_path, split='unlab25', aug_times = 5)
        elif labelnum == 50:
            train_labset = StrongWeakBrats_PancreasStyle(args.root_path, split='lab50', aug_times = 5)
            train_unlabset = StrongWeakBrats_PancreasStyle(args.root_path, split='unlab50', aug_times = 5)
        elif labelnum == 100:
            train_labset = StrongWeakBrats_PancreasStyle(args.root_path, split='lab100', aug_times = 5)
            train_unlabset = StrongWeakBrats_PancreasStyle(args.root_path, split='unlab100', aug_times = 5)
        else:
            raise NotImplementedError

        if args.dataset == 'LA':
            valset = StrongWeakLA_PancreasStyle(args.root_path, split='val', aug_times = 5)
            testset = StrongWeakLA_PancreasStyle(args.root_path, split='test', aug_times = 5)
        elif args.dataset == 'pancreas':
            valset = StrongWeakPancreas(args.root_path, split='val', aug_times = 5)
            testset = StrongWeakPancreas(args.root_path, split='test', aug_times = 5)
        elif args.dataset == 'brats':
            valset = StrongWeakBrats_PancreasStyle(args.root_path, split='val', aug_times = 5)
            testset = StrongWeakBrats_PancreasStyle(args.root_path, split='test', aug_times = 5)
        
        trainlab_loader = DataLoader(train_labset, batch_size=args.labeled_bs, shuffle=True, num_workers=0, drop_last=True if args.dataset == 'brats' else False)
        trainunlab_loader = DataLoader(train_unlabset, batch_size=args.batch_size-args.labeled_bs, shuffle=True, num_workers=0, drop_last=True if args.dataset == 'brats' else False)
        test_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

        logging.info("{} iterations for lab per epoch.".format(len(trainlab_loader)))
        logging.info("{} iterations for unlab per epoch.".format(len(trainunlab_loader)))
        logging.info("{} samples for test.\n".format(len(test_loader)))
        return trainlab_loader, trainunlab_loader, test_loader

    trainlab_loader, trainunlab_loader, test_loader = create_dataloader()

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    
    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainlab_loader)))
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainlab_loader) + 1
    lr_ = base_lr
    iterator = range(max_epoch)
    if args.mask_cont2 == 'mask':
        masking = Masking(
            block_size=args.mask_block_size,
            ratio=args.mask_ratio,
            color_jitter_s=args.mask_color_jitter_s,
            color_jitter_p=args.mask_color_jitter_p,
            blur=args.mask_blur,)
    else:
        raise NotImplementedError
    # init tau
    tau = torch.zeros((1, 1, 1, 1)).cuda()
    
    for epoch_num in range(15000):
        for step, (labeled_batch, unlabeled_batch) in enumerate(zip(trainlab_loader,trainunlab_loader)):
            start = time.time()
            weak_batch_l, label_l = labeled_batch['image_weak'].cuda(), labeled_batch['label'].cuda()
            weak_batch_u, strong_batch_u = unlabeled_batch['image_weak'].cuda(), unlabeled_batch['image_strong'].cuda()
            weak_batch = torch.cat([weak_batch_l, weak_batch_u], dim=0) # label + unlabel
            strong_batch = torch.cat([torch.zeros_like(strong_batch_u), strong_batch_u], dim=0) # 0 + unlabel
            label_batch = torch.cat([label_l, torch.zeros_like(label_l)], dim=0)

            weak_batch, strong_batch, label_batch = (
                weak_batch.cuda(),
                strong_batch.cuda(),
                label_batch.cuda(),
            )
            # to avoid using unlabeled data masks, make them to be 0
            label_batch[args.labeled_bs :] = 0
            
            outputs_weak, att_map = model(weak_batch)
            outputs_weak_soft = torch.softmax(outputs_weak, dim=1)
            logits_u_aug, label_u_aug = torch.max(outputs_weak_soft, dim=1)
            strong_batch = masking(strong_batch)
            outputs_strong, _ = model(strong_batch)
            outputs_strong_soft = torch.softmax(outputs_strong, dim=1)
            end1 = time.time()

            label_batch[args.labeled_bs :] = 0

            model.train()
            
            # -------------------------------------------- #
            # ------ main training process for model ----- #
            # -------------------------------------------- #

            sup_loss = ce_loss(outputs_weak[: args.labeled_bs], label_batch[:][: args.labeled_bs].long(),) + \
                dice_loss(outputs_weak_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1))

            if args.thresh_method == 'none':
                unsup_loss, pseduo_high_ratio = losses.compute_unsupervised_loss_by_threshold(
                    outputs_strong[args.labeled_bs :],
                    label_u_aug[args.labeled_bs :],
                    logits_u_aug[args.labeled_bs :],
                    thresh=args.conf_thresh,
                )
            elif args.thresh_method == 'patch_wise':
                unsup_loss, pseduo_high_ratio, tau = losses.compute_unsupervised_loss_by_patch_wise_threshold(
                    outputs_strong[args.labeled_bs :],
                    label_u_aug[args.labeled_bs :],
                    logits_u_aug[args.labeled_bs :],
                    iter_num,
                    init_thresh=args.conf_thresh,
                    path_size=(16, 16, 16), 
                    ema=0.99, 
                    tau=tau,
                )
                end2 = time.time()

            loss = sup_loss + args.ada_u_weight * unsup_loss

            loss_l = torch.zeros(1).cuda()
            loss_u = torch.zeros(1).cuda()

            img_a = weak_batch_l
            lab_a = label_l
            unimg_a = weak_batch_u
            assert len(img_a) == len(unimg_a) == args.labeled_bs == args.batch_size - args.labeled_bs == len(lab_a)
            with torch.no_grad():
                unoutput_a, att_map_a = model(unimg_a)
                plab_a = get_pseudo_label(unoutput_a, nms=1)
                end3 = time.time()
            mask_method = soft_attention_mask if args.attn_mask_method == 'soft' else attention_mask
            unimg_mask_a, unloss_mask_a, mask_region_a = mask_method(unimg_a,
                                                mask_ratio=2/3,
                                                attention_map=att_map_a,
                                                shape=patch_size,
                                                order=args.mask_order,
                                                jitter=args.jitter,
                                                )
            assert args.attn_mask_correct == 3
            loss_mask_a = torch.ones_like(unloss_mask_a).cuda() - unloss_mask_a
            img_mask_a = torch.ones_like(unimg_mask_a).cuda() - unimg_mask_a
            mix_img = img_a * img_mask_a + unimg_a * (1 - img_mask_a)
            outputs_mix, _ = model(mix_img)
            loss_mix = mix_loss(outputs_mix, lab_a, plab_a, loss_mask_a, u_weight=args.mix_unlab_weight)
            loss += args.lambda_mix_loss * loss_mix
            end4 = time.time()

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1

            writer.add_scalar('1_Loss/sup_loss', sup_loss, iter_num)
            writer.add_scalar('1_Loss/unsup_loss', unsup_loss, iter_num)
            assert args.thresh_method == 'patch_wise'
            logging.info('iteration %d : loss : %03f, loss_sup: %03f, loss_unsup: %03f, loss_mix: %03f, high_ratio: %03f, max_tau: %03f, min_tau: %03f' % \
                            (iter_num, loss.item(), sup_loss.item(), unsup_loss.item(), loss_mix.item(), pseduo_high_ratio, torch.max(tau).item(), torch.min(tau).item()))

            
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
                                    stride_z=stride_z,
                                    val_list="val")
                logging.info("Dice score for VAL at {}-th iteration is {}".format(iter_num, round(dice_sample, 4)))
                if dice_sample > best_dice:
                    best_dice = round(dice_sample, 4)
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
                    logging.info("Average metric for TEST SET is {}".format(avg_metric))
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