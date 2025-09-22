import sys
import os
import torch
import random
from networks.build_network import select_model
# test build network
if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.model = 'lightunetr_large'  # Example model name

    args = Args()
    num_classes = 3
    emb_length = 27
    patch_size = (96, 96, 96)
    in_channels = 1
    snapshot_path = './'

    model = select_model(args, num_classes, emb_length, patch_size, in_channels)
    y = model(torch.randn(1, in_channels, *patch_size).cuda())
    print(f"Output shape: {y.shape}")
    print(f"Model selected: {args.model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    # save_model_info(model, patch_size, snapshot_path)