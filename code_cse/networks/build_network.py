"""
Network Builder for LightUNETR Models

This module provides an interface for selecting and creating LightUNETR models.
"""

import os
import sys
import copy
import torch 
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis

from networks.lightunetr import LightUNETR
from networks.lightunetr_large import LightUNETRLarge


def select_model(args, num_classes, emb_length, patch_size, in_channels=1):
    """
    Select and create a model based on the arguments.
    
    Args:
        args: Arguments containing model selection
        num_classes: Number of output classes
        emb_length: Embedding length (for compatibility)
        patch_size: Input patch size (for compatibility)
        in_channels: Number of input channels
        
    Returns:
        Selected model
    """
    if args.model == 'lightunetr':
        model = LightUNETR(
            in_channels,
            num_classes,
            embedding_dim=emb_length,
        ).cuda()
    elif args.model == 'lightunetr_large':
        model = LightUNETRLarge(
            in_channels, 
            num_classes, 
            embedding_dim=emb_length,
        ).cuda()
    
    return model


def compute_flops(model, tensor):
    """Compute FLOPs for the given model and input tensor."""
    new_model = copy.deepcopy(model)
    flops = FlopCountAnalysis(new_model, tensor)
    del new_model
    return flops.total() / 1e9


def compute_params(model):
    """Compute total parameters and parameters excluding head."""
    n_parameters = 0
    n_parameters_rm_head = 0
    for name, p in model.named_parameters():
        if 'head' in name:
            n_parameters += p.numel()
        else:
            n_parameters += p.numel()
            n_parameters_rm_head += p.numel()
    return n_parameters / 1.0e6, n_parameters_rm_head / 1.0e6


def save_model_info(model, patch_size, snapshot_path):
    """Save model information including FLOPs and parameters to a text file."""
    x1 = torch.randn(1, 1, patch_size[0], patch_size[1], patch_size[2]).cuda()
    n_flops = compute_flops(model, x1)
    n_params, n_parameters_rm_head = compute_params(model)

    with open(os.path.join(snapshot_path, 'model.txt'), 'w') as f:
        print(model, file=f)
        print("WARNING: The FLOPs count is ONLY an approximation. please use thop (https://github.com/Lyken17/pytorch-OpCounter) for a more precise measurement.", file=f)
        print(f'flops with input size {x1.shape}: {n_flops:.2f} G', file=f)
        print(f'params with input size {x1.shape}: {n_params:.2f} M', file=f)
        print(f'params without head with input size {x1.shape}: {n_parameters_rm_head:.2f} M', file=f)
    del x1


# Usage example:
# model = select_model(args, num_classes, emb_length, patch_size)
# save_model_info(model, patch_size, snapshot_path)
