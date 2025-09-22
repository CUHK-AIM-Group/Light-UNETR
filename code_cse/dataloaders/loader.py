import json
import os
from typing import Dict, Hashable, Mapping, Tuple

import monai
import numpy as np
import torch
from easydict import EasyDict
from monai.utils import ensure_tuple_rep

from typing import Mapping, Hashable, Dict
from monai.transforms import MapTransform
import torch
import numpy as np
from monai.config import KeysCollection, NdarrayOrTensor

def load_abdomenct1k_dataset_images(root):
    img_path = os.path.join(root, "imagesTr")
    label_path = os.path.join(root, "labelsTr_samename")
    images_list = []
    for img_name in os.listdir(img_path):
        if img_name.endswith(".nii.gz"):
            image = os.path.join(img_path, img_name)
            label = os.path.join(label_path, img_name.replace("image", "label"))
            images_list.append({"image": image, "label": label})
    # sort images_list by image name
    images_list.sort(key=lambda x: os.path.basename(x["image"]))
    # sort labels_list by label name
    for image in images_list:
        image["label"] = image["label"].replace("image", "label")
    return images_list


import monai
from monai.config import KeysCollection
from easydict import EasyDict

def get_AbdomenCT1k_transforms():
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            
            monai.transforms.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5, upper=99.5,
                b_min=0, b_max=1, clip=True,
            ),
            monai.transforms.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(96,96,96),
                mode="constant",
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96,96,96),
                num_samples=2,
                pos=1, neg=1,
                image_key="image",
                image_threshold=0,
            ),
            
            monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            monai.transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            monai.transforms.RandRotated(
                keys=["image", "label"], prob=0.3
            ),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            
            monai.transforms.ScaleIntensityRangePercentilesd(
                keys=["image"],
                lower=0.5, upper=99.5,
                b_min=0, b_max=1, clip=True,
            ),
            
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    return train_transform, val_transform

def get_dataloader(dataroot):
    train_images = load_abdomenct1k_dataset_images(dataroot)
    train_transform, val_transform = get_AbdomenCT1k_transforms()

    # see if cache_dataset key exist in the config
    val_dataset = monai.data.PersistentDataset(
        data=train_images[int(len(train_images) * 0.8) :],
        transform=val_transform,
        cache_dir=os.path.join(dataroot, "cache"),
    )
    train_num = int(len(train_images) * 0.8)
    train_labnum = int(train_num * 0.1)
    train_labdataset = monai.data.PersistentDataset(
        data=train_images[: train_labnum],
        transform=train_transform,
        cache_dir=os.path.join(dataroot, "cache"),
    )
    train_unlabdataset = monai.data.PersistentDataset(
        data=train_images[train_labnum:train_num],
        transform=train_transform,
        cache_dir=os.path.join(dataroot, "cache"),
    )
    train_labloader = monai.data.DataLoader(
        train_labdataset,
        num_workers=8,
        batch_size=1,
        shuffle=True,
    )
    train_unlabloader = monai.data.DataLoader(
        train_unlabdataset,
        num_workers=8,
        batch_size=1,
        shuffle=True,
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=8,
        batch_size=1,
        shuffle=False,
    )

    return train_labloader, train_unlabloader, val_loader
