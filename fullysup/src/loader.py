import json
import os
from typing import Dict, Hashable, Mapping, Tuple

import monai
import numpy as np
import torch
from easydict import EasyDict
from monai.utils import ensure_tuple_rep


class ConvertToMultiChannelBasedOnBratsClassesd(monai.transforms.MapTransform):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        keys: monai.config.KeysCollection,
        is2019: bool = False,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.is2019 = is2019

    def converter(self, img: monai.config.NdarrayOrTensor):
        # TC WT ET
        # if img has channel dim, squeeze it
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        if self.is2019:
            result = [
                (img == 2) | (img == 3),
                (img == 1) | (img == 2) | (img == 3),
                (img == 2),
            ]
        else:
            # TC WT ET
            result = [
                (img == 1) | (img == 4),
                (img == 1) | (img == 4) | (img == 2),
                img == 4,
            ]
            # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
            # label 4 is ET
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]
    ) -> Dict[Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

from typing import Mapping, Hashable, Dict
from monai.transforms import MapTransform
import torch
import numpy as np
from monai.config import KeysCollection, NdarrayOrTensor

class ConvertToMultiChannelAbdomenCTd(MapTransform):
    """
    Explicitly convert AbdomenCT-1k labels into multi-channel binary masks.
    Labels:
        background: 0 (ignored)
        liver: 1
        kidney: 2
        spleen: 3
        pancreas: 4
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        # explicitly predefined organ-label lookup stored internally
        labels = {
            "background": 0,
            "liver": 1,
            "kidney": 2,
            "spleen": 3,
            "pancreas": 4,
        }
        # explicitly exclude background
        self.labels = {k: v for k, v in labels.items() if v != 0}

    def converter(self, img: NdarrayOrTensor):
        # explicitly squeeze unnecessary channel dimension clearly
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        # explicitly create binary masks per organ
        result = [(img == label_id) for label_id in self.labels.values()]

        # explicitly stack masks into multi-channel segmentation mask
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


class ConvertToMultiChannelHeadNeckd(MapTransform):
    """
    Explicitly convert HeadNeck labels into multi-channel binary masks.
    Labels:
        background: 0 (ignored)
        head tumor: 1
        neck tumor: 2
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        # explicitly predefined organ-label lookup stored internally
        labels = {
            "background": 0,
            "head": 1,
            "neck": 2
        }
        # explicitly exclude background
        self.labels = {k: v for k, v in labels.items() if v != 0}

    def converter(self, img: NdarrayOrTensor):
        # explicitly squeeze unnecessary channel dimension clearly
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)

        # explicitly create binary masks per organ
        result = [(img == label_id) for label_id in self.labels.values()]

        # explicitly stack masks into multi-channel segmentation mask
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d

class ConvertToMultiChannelBasedOnBratsClassesd_for_MSD(monai.transforms.MapTransform):
    """
    TC WT ET
    Dictionary-based wrapper of :py:class:`monai.transforms.ConvertToMultiChannelBasedOnBratsClasses`.
    Convert labels to multi channels based on brats18 classes:
    label 1 is the necrotic and non-enhancing tumor core
    label 2 is the peritumoral edema
    label 4 is the GD-enhancing tumor
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).
    """

    backend = [monai.utils.TransformBackends.TORCH, monai.utils.TransformBackends.NUMPY]

    def __init__(
        self, keys: monai.config.KeysCollection, allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)

    def converter(self, img: monai.config.NdarrayOrTensor):
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        result = [(img == 1), (img == 2)]
        return (
            torch.stack(result, dim=0)
            if isinstance(img, torch.Tensor)
            else np.stack(result, axis=0)
        )

    def __call__(
        self, data: Mapping[Hashable, monai.config.NdarrayOrTensor]
    ) -> Dict[Hashable, monai.config.NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.converter(d[key])
        return d


def load_brats2021_dataset_images(root):
    images_path = os.listdir(root)
    images_list = []
    for path in images_path:
        image_path = root + "/" + path + "/" + path
        flair_img = image_path + "_flair.nii.gz"
        t1_img = image_path + "_t1.nii.gz"
        t1ce_img = image_path + "_t1ce.nii.gz"
        t2_img = image_path + "_t2.nii.gz"
        seg_img = image_path + "_seg.nii.gz"
        images_list.append(
            {"image": [flair_img, t1_img, t1ce_img, t2_img], "label": seg_img}
        )
    return images_list


def load_brats2019_dataset_images(root):
    root_dir = root + "/dataset.json"
    # load files
    with open(root_dir, encoding="utf-8") as a:
        # read files
        images_list = json.load(a)["training"]
        for image in images_list:
            image["image"] = image["image"].replace("./", root + "/")
            image["label"] = image["label"].replace("./", root + "/")
    return images_list


def load_abdomenct1k_dataset_images(root):
    img_path = os.path.join(root, "imagesTr")
    label_path = os.path.join(root, "labelsTr")
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

def load_headneck_dataset_images(root):
    img_path = os.path.join(root, "imagesTr")
    label_path = os.path.join(root, "labelsTr")
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

def get_Brats_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(
                keys=["label"], is2019=config.trainer.is_brats2019
            ),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.SpatialPadD(
                keys=["image", "label"],
                spatial_size=(255, 255, config.trainer.image_size),
                method="symmetric",
                mode="constant",
            ),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.CenterSpatialCropD(
                keys=["image", "label"],
                roi_size=ensure_tuple_rep(config.trainer.image_size, 3),
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                num_samples=2,
                spatial_size=ensure_tuple_rep(config.trainer.image_size, 3),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=0
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=1
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=2
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
            monai.transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            monai.transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys="image"),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(
                keys="label", is2019=config.trainer.is_brats2019
            ),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.NormalizeIntensityd(
                keys="image", nonzero=True, channel_wise=True
            ),
        ]
    )
    return train_transform, val_transform


def load_dataset_images(root):
    root_dir = root + "/dataset.json"
    # load files
    with open(root_dir, encoding="utf-8") as a:
        # read flies
        images_list = json.load(a)["training"]
        for image in images_list:
            image["image"] = image["image"].replace("./", root + "/")
            image["label"] = image["label"].replace("./", root + "/")
    return images_list


def get_MSD_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd_for_MSD(keys=["label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            # foreground crop
            monai.transforms.CropForegroundd(
                keys=["image", "label"], source_key="image"
            ),
            # intensity limit
            monai.transforms.ScaleIntensityRanged(
                keys=["image", "label"],
                a_min=0.0,
                a_max=230.0,
                b_min=0.0,
                b_max=230.0,
                clip=True,
            ),
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                num_samples=2,
                spatial_size=monai.utils.ensure_tuple_rep(config.trainer.image_size, 3),
                pos=1,
                neg=1,
                image_key="image",
                image_threshold=0,
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=0
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=1
            ),
            monai.transforms.RandFlipd(
                keys=["image", "label"], prob=0.5, spatial_axis=2
            ),
            monai.transforms.RandAxisFlipd(keys=["image", "label"], prob=0.5),
            monai.transforms.RandRotated(keys=["image", "label"], prob=0.25),
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
            ConvertToMultiChannelBasedOnBratsClassesd_for_MSD(keys=["label"]),
            monai.transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            monai.transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            monai.transforms.CropForegroundd(
                keys=["image", "label"], source_key="image"
            ),
            # 强度限制
            monai.transforms.ScaleIntensityRanged(
                keys=["image", "label"],
                a_min=0.0,
                a_max=230.0,
                b_min=0.0,
                b_max=230.0,
                clip=True,
            ),
            monai.transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    return train_transform, val_transform


import monai
from monai.config import KeysCollection
from easydict import EasyDict

def get_AbdomenCT1k_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelAbdomenCTd(keys=["label"]),
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
            monai.transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96,96,96),
                num_samples=2,
                pos=1, neg=1,
                image_key="image",
                image_threshold=0,
                allow_smaller=True,
            ),
            monai.transforms.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(96,96,96),
                mode="constant",
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
            ConvertToMultiChannelAbdomenCTd(keys=["label"]),
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


def get_HeadNeck_transforms(
    config: EasyDict,
) -> Tuple[monai.transforms.Compose, monai.transforms.Compose]:
    train_transform = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=["image", "label"]),
            monai.transforms.EnsureChannelFirstd(keys=["image", "label"]),
            monai.transforms.EnsureTyped(keys=["image", "label"]),
            ConvertToMultiChannelHeadNeckd(keys=["label"]),
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
            ConvertToMultiChannelHeadNeckd(keys=["label"]),
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

def get_dataloader(
    config: EasyDict, data_flag: str
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if data_flag == "abdomenct1k":
        train_images = load_abdomenct1k_dataset_images(config.data_root)
        train_transform, val_transform = get_AbdomenCT1k_transforms(config)
    elif data_flag == "headneck":
        train_images = load_headneck_dataset_images(config.data_root)
        train_transform, val_transform = get_HeadNeck_transforms(config)
    else:
        if data_flag == "brain2019":
            train_images = load_brats2019_dataset_images(config.data_root)
            config.trainer.is_brats2019 = True
        else:
            train_images = load_brats2021_dataset_images(config.data_root)
            config.trainer.is_brats2019 = False
        train_transform, val_transform = get_Brats_transforms(config)

    print(f"Using cache dataset for {data_flag}")
    # cache dataset
    val_dataset = monai.data.PersistentDataset(
        data=train_images[int(len(train_images) * config.trainer.train_ratio) :],
        transform=val_transform,
        cache_dir=os.path.join(config.data_root, "cache"),
    )
    train_dataset = monai.data.PersistentDataset(
        data=train_images[: int(len(train_images) * config.trainer.train_ratio)],
        transform=train_transform,
        cache_dir=os.path.join(config.data_root, "cache"),
    )

    train_loader = monai.data.DataLoader(
        train_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    val_loader = monai.data.DataLoader(
        val_dataset,
        num_workers=config.trainer.num_workers,
        batch_size=1,
        shuffle=False,
    )

    return train_loader, val_loader, train_images, train_transform
