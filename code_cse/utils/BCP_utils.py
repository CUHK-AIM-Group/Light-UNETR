import numpy as np
import torch.nn as nn
import torch
import random
import math
from utils.losses import mask_DiceLoss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

DICE = mask_DiceLoss(nclass=2)
CE = nn.CrossEntropyLoss(reduction='none')
DICE_5 = mask_DiceLoss(nclass=5)

def attention_mask(img, mask_ratio, attention_map, shape=(112, 112, 80), order='ascend', n_patches_per_dim=3, jitter=0):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    # upsampling attention map to the same size as image
    attention_map = nn.functional.interpolate(attention_map, size=(img_x, img_y, img_z), mode='trilinear', align_corners=True)
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    # split into 27 overlapping patches
    n_w, n_h, n_z = n_patches_per_dim, n_patches_per_dim, n_patches_per_dim
    stride_x, stride_y, stride_z = int((img.shape[2] - patch_pixel_x) / (n_w - 1)), int((img.shape[3] - patch_pixel_y) / (n_h - 1)), int((img.shape[4] - patch_pixel_z) / (n_z - 1))
    attn_val = {}
    for i in range(n_w):
        for j in range(n_h):
            for k in range(n_z):
                start_x, start_y, start_z = i*stride_x, j*stride_y, k*stride_z
                end_x, end_y, end_z = start_x+patch_pixel_x, start_y+patch_pixel_y, start_z+patch_pixel_z
                total_attn = attention_map[:, :, start_x:end_x, start_y:end_y, start_z:end_z].sum()
                attn_val['{}_{}_{}'.format(start_x, start_y, start_z)] = total_attn
    attn_val = sorted(attn_val.items(), key=lambda x: x[1], reverse=False if order=='ascend' else True)
    # print('attn_val: ', attn_val)
    w = int(attn_val[0][0].split('_')[0])
    h = int(attn_val[0][0].split('_')[1])
    z = int(attn_val[0][0].split('_')[2])
    # random jitter by 10 pixels
    if jitter > 0:
        jitter = int(jitter)
        w = np.random.randint(w-jitter, w+jitter)
        h = np.random.randint(h-jitter, h+jitter)
        z = np.random.randint(z-jitter, z+jitter)
    # print('w: {} to {}, h: {} to {}, z: {} to {}'.format(w, w+patch_pixel_x, h, h+patch_pixel_y, z, z+patch_pixel_z))
    mask_region = 'w: {}-{}, h: {}-{}, z: {}-{}'.format(w, w+patch_pixel_x, h, h+patch_pixel_y, z, z+patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0 # mask out the lowest attention patch
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long(), mask_region


def soft_attention_mask(img, mask_ratio, attention_map, shape=(112, 112, 80), order='ascend', n_patches_per_dim=3, jitter=0):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    # upsampling attention map to the same size as image
    attention_map = nn.functional.interpolate(attention_map.detach().clone(), size=(img_x, img_y, img_z), mode='trilinear', align_corners=True)
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    # split into 27 overlapping patches
    n_w, n_h, n_z = n_patches_per_dim, n_patches_per_dim, n_patches_per_dim
    stride_x, stride_y, stride_z = int((img.shape[2] - patch_pixel_x) / (n_w - 1)), int((img.shape[3] - patch_pixel_y) / (n_h - 1)), int((img.shape[4] - patch_pixel_z) / (n_z - 1))
    attn_val = {}
    for i in range(n_w):
        for j in range(n_h):
            for k in range(n_z):
                start_x, start_y, start_z = i*stride_x, j*stride_y, k*stride_z
                end_x, end_y, end_z = start_x+patch_pixel_x, start_y+patch_pixel_y, start_z+patch_pixel_z
                total_attn = attention_map[:, :, start_x:end_x, start_y:end_y, start_z:end_z].sum()
                attn_val['{}_{}_{}'.format(start_x, start_y, start_z)] = total_attn.item()
    # attn_val = sorted(attn_val.items(), key=lambda x: x[1], reverse=False if order=='ascend' else True)
    # make the attention values as probabilities by softmax
    key_l = list(attn_val.keys())
    val_l = list(attn_val.values())
    val_l = [(x - min(val_l)) / (max(val_l) - min(val_l)) for x in val_l]
    val_l = torch.softmax(torch.tensor(val_l), dim=0).tolist()
    attn_val = dict(zip(key_l, val_l))
    # random select a patch based on the probability
    selected_patch = random.choices(list(attn_val.keys()), weights=list(attn_val.values()), k=1)
    w = int(selected_patch[0].split('_')[0])
    h = int(selected_patch[0].split('_')[1])
    z = int(selected_patch[0].split('_')[2])
    # random jitter by 10 pixels
    if jitter > 0:
        jitter = int(jitter)
        w = np.random.randint(w-jitter, w+jitter)
        h = np.random.randint(h-jitter, h+jitter)
        z = np.random.randint(z-jitter, z+jitter)
    # print('w: {} to {}, h: {} to {}, z: {} to {}'.format(w, w+patch_pixel_x, h, h+patch_pixel_y, z, z+patch_pixel_z))
    mask_region = 'w: {}-{}, h: {}-{}, z: {}-{}'.format(w, w+patch_pixel_x, h, h+patch_pixel_y, z, z+patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0 # mask out the lowest attention patch
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long(), mask_region

def context_mask(img, mask_ratio, shape=(112, 112, 80)):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, shape[0] - patch_pixel_x)
    h = np.random.randint(0, shape[1] - patch_pixel_y)
    z = np.random.randint(0, shape[2] - patch_pixel_z)
    mask_region = 'w: {}-{}, h: {}-{}, z: {}-{}'.format(w, w+patch_pixel_x, h, h+patch_pixel_y, z, z+patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long()

def random_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*2/3), int(img_y*2/3), int(img_z*2/3)
    mask_num = 27
    mask_size_x, mask_size_y, mask_size_z = int(patch_pixel_x/3)+1, int(patch_pixel_y/3)+1, int(patch_pixel_z/3)
    size_x, size_y, size_z = int(img_x/3), int(img_y/3), int(img_z/3)
    for xs in range(3):
        for ys in range(3):
            for zs in range(3):
                w = np.random.randint(xs*size_x, (xs+1)*size_x - mask_size_x - 1)
                h = np.random.randint(ys*size_y, (ys+1)*size_y - mask_size_y - 1)
                z = np.random.randint(zs*size_z, (zs+1)*size_z - mask_size_z - 1)
                mask[w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
                loss_mask[:, w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
    return mask.long(), loss_mask.long()

def concate_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    z_length = int(img_z * 8 / 27)
    z = np.random.randint(0, img_z - z_length -1)
    mask[:, :, z:z+z_length] = 0
    loss_mask[:, :, :, z:z+z_length] = 0
    return mask.long(), loss_mask.long()

def mix_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask) * image_weight 
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def mix_loss5(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    dice_loss = DICE_5(net3_output, img_l, mask) * image_weight 
    dice_loss += DICE_5(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def sup_loss(output, label):
    label = label.type(torch.int64)
    dice_loss = DICE(output, label)
    loss_ce = torch.mean(CE(output, label))
    loss = (dice_loss + loss_ce) / 2
    return loss

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

@torch.no_grad()
def update_ema_students(model1, model2, ema_model, alpha):
    for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
        ema_param.data.mul_(alpha).add_(((1 - alpha)/2) * param1.data).add_(((1 - alpha)/2) * param2.data)

@torch.no_grad()
def parameter_sharing(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = param.data

class BBoxException(Exception):
    pass

def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    max += 1
    return min, max


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 2)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 0)

    return np.array(((min_x, max_x),
                     (min_y, max_y),
                     (min_z, max_z)))

def get_bbox_mask(mask):
    batch_szie, x_dim, y_dim, z_dim = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
    mix_mask = torch.ones(batch_szie, 1, x_dim, y_dim, z_dim).cuda()
    for i in range(batch_szie):
        curr_mask = mask[i, ...].squeeze()
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bbox_3d(curr_mask)
        mix_mask[i, :, min_x:max_x, min_y:max_y, min_z:max_z] = 0
    return mix_mask.long()
