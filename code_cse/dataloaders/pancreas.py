import os
import torch
import numpy as np
from torch.utils.data import Dataset
import h5py
from torch.utils.data.sampler import Sampler
from torchvision.transforms import Compose

from .volumentations import RandomGamma, GaussianNoise
from .volumentations import Compose as VCompose


"""Pancreas modified from https://github.com/grant-jpg/FUSSNet"""

class StrongWeakPancreas(Dataset):
    """ Pancreas Dataset """

    def __init__(self, base_dir, split, aug_times=5, args=None): # aug_times=5 follows https://github.com/VivienLu/UPCoL
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.split = split
        self.aug_times = aug_times
        self.args = args

        tr_transform = Compose([
            RandomCrop((96, 96, 96)),
            ToTensor()
        ])
        test_transform = Compose([
            CenterCrop((96, 96, 96)),
            ToTensor()
        ])
            
        if split == 'lab':
            data_path = os.path.join(base_dir,'train_lab.list')
            self.transform = tr_transform
        elif split == 'unlab':
            data_path = os.path.join(base_dir,'train_unlab.list')
            self.transform = tr_transform 
        elif split == 'train':
            data_path = os.path.join(base_dir,'train.list')
            self.transform = tr_transform
        elif split == 'lab6':
            data_path = os.path.join(base_dir,'train_lab6.list')
            self.transform = tr_transform
        elif split == 'unlab6':
            data_path = os.path.join(base_dir,'train_unlab6.list')
            self.transform = tr_transform
        elif split == 'unlab6_noval':
            data_path = os.path.join(base_dir,'train_unlab6_noval.list')
            self.transform = tr_transform
        elif split == 'lab12':
            data_path = os.path.join(base_dir,'train_lab12.list')
            self.transform = tr_transform
        elif split == 'unlab12':
            data_path = os.path.join(base_dir,'train_unlab12.list')
            self.transform = tr_transform
        elif split == 'unlab12_noval':
            data_path = os.path.join(base_dir,'train_unlab12_noval.list')
            self.transform = tr_transform
        elif split == 'lab24':
            data_path = os.path.join(base_dir,'train_lab24.list')
            self.transform = tr_transform
        elif split == 'unlab24':
            data_path = os.path.join(base_dir,'train_unlab24.list')
            self.transform = tr_transform
        elif split == 'lab48':
            data_path = os.path.join(base_dir,'train_lab48.list')
            self.transform = tr_transform
        elif split == 'unlab48':
            data_path = os.path.join(base_dir,'train_unlab48.list')
            self.transform = tr_transform
        elif split == 'val':
            data_path = os.path.join(base_dir,'val.list')
            self.transform = test_transform
        else:
            data_path = os.path.join(base_dir,'test.list')
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [self.data_dir+ "/{}".format(item.strip()) + '.h5' for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

        if self.args:
            if not self.args.no_strong_gamma:
                self.aug = VCompose([
                    RandomGamma(always_apply=True),
                ], p=1)
                print("strong gamma")
            if not self.args.no_strong_noise:
                self.aug = VCompose([
                    RandomGamma(always_apply=True),
                    GaussianNoise(),
                ], p=1)
                print("strong noise")
        else:
            self.aug = VCompose([
                    RandomGamma(always_apply=True),
                    GaussianNoise(),
                ], p=1)
            print("strong gamma and noise")

    def __len__(self):
        if self.split != 'test':
            return len(self.image_list) * self.aug_times
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)

        image_weak = image.copy()
        image_strong = image.copy()
        sample_for_aug = {'image': image_strong, 'mask': label}
        sample_for_aug = self.aug(**sample_for_aug)
        image_strong = sample_for_aug['image']
        samples = image_weak, image_strong, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_weak, image_strong, label = tr_samples
        return {'image_weak':image_weak.float(), 'image_strong':image_strong.float(), 'label':label.long(), 'name':image_path, 'idx':idx % len(self.image_list)}

class StrongWeakLA_PancreasStyle(Dataset):
    """ Pancreas Dataset """

    def __init__(self, base_dir, split, aug_times=5, args=None): # aug_times=5 follows https://github.com/VivienLu/UPCoL
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.split = split
        self.aug_times = aug_times
        self.args = args

        tr_transform = Compose([
            RandomCrop((112, 112, 80)),
            ToTensor()
        ])
        test_transform = Compose([
            CenterCrop((112, 112, 80)),
            ToTensor()
        ])
            
        if split == 'lab':
            data_path = os.path.join(base_dir,'train_lab.list')
            self.transform = tr_transform
        elif split == 'unlab':
            data_path = os.path.join(base_dir,'train_unlab.list')
            self.transform = tr_transform 
        elif split == 'train':
            data_path = os.path.join(base_dir,'train.list')
            self.transform = tr_transform
        elif split == 'lab4':
            data_path = os.path.join(base_dir,'train_lab4.list')
            self.transform = tr_transform
        elif split == 'unlab4':
            data_path = os.path.join(base_dir,'train_unlab4.list')
            self.transform = tr_transform
        elif split == 'unlab4_noval':
            data_path = os.path.join(base_dir,'train_unlab4_noval.list')
            self.transform = tr_transform
        elif split == 'lab8':
            data_path = os.path.join(base_dir,'train_lab8.list')
            self.transform = tr_transform
        elif split == 'unlab8':
            data_path = os.path.join(base_dir,'train_unlab8.list')
            self.transform = tr_transform
        elif split == 'unlab8_noval':
            data_path = os.path.join(base_dir,'train_unlab8_noval.list')
            self.transform = tr_transform
        elif split == 'lab16':
            data_path = os.path.join(base_dir,'train_lab16.list')
            self.transform = tr_transform
        elif split == 'unlab16':
            data_path = os.path.join(base_dir,'train_unlab16.list')
            self.transform = tr_transform
        elif split == 'lab32':
            data_path = os.path.join(base_dir,'train_lab32.list')
            self.transform = tr_transform
        elif split == 'unlab32':
            data_path = os.path.join(base_dir,'train_unlab32.list')
            self.transform = tr_transform
        elif split == 'val':
            data_path = os.path.join(base_dir,'val.list')
            self.transform = test_transform
        else:
            data_path = os.path.join(base_dir,'test.list')
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

        if self.args:
            if not self.args.no_strong_gamma:
                self.aug = VCompose([
                    RandomGamma(always_apply=True),
                ], p=1)
                print("strong gamma")
            if not self.args.no_strong_noise:
                self.aug = VCompose([
                    RandomGamma(always_apply=True),
                    GaussianNoise(),
                ], p=1)
                print("strong noise")
        else:
            self.aug = VCompose([
                    RandomGamma(always_apply=True),
                    GaussianNoise(),
                ], p=1)
            print("strong gamma and noise")

    def __len__(self):
        if self.split != 'test':
            return len(self.image_list) * self.aug_times
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        h5f = h5py.File(self.base_dir + "/2018LA_Seg_Training Set/" + image_path + "/mri_norm2.h5", 'r')
        image, label = h5f['image'][:], h5f['label'][:]

        image_weak = image.copy()
        image_strong = image.copy()
        sample_for_aug = {'image': image_strong, 'mask': label}
        sample_for_aug = self.aug(**sample_for_aug)
        image_strong = sample_for_aug['image']
        samples = image_weak, image_strong, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_weak, image_strong, label = tr_samples
        return {'image_weak':image_weak.float(), 'image_strong':image_strong.float(), 'label':label.long(), 'name':image_path, 'idx':idx % len(self.image_list)}


class StrongWeakLA_PancreasStyle_withAUG(Dataset):
    """ Pancreas Dataset """

    def __init__(self, base_dir, split, aug_times=5, args=None): # aug_times=5 follows https://github.com/VivienLu/UPCoL
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.split = split
        self.aug_times = aug_times
        self.args = args

        tr_transform = Compose([
            # RandomRotFlip(),
            RandomCrop((112, 112, 80)),
            ToTensor()
        ])
        test_transform = Compose([
            CenterCrop((112, 112, 80)),
            ToTensor()
        ])
            
        if split == 'lab':
            data_path = os.path.join(base_dir,'train_lab.list')
            self.transform = tr_transform
        elif split == 'unlab':
            data_path = os.path.join(base_dir,'train_unlab.list')
            self.transform = test_transform 
        elif split == 'train':
            data_path = os.path.join(base_dir,'train.list')
            self.transform = tr_transform
        elif split == 'lab4':
            data_path = os.path.join(base_dir,'train_lab4.list')
            self.transform = tr_transform
        elif split == 'unlab4':
            data_path = os.path.join(base_dir,'train_unlab4.list')
            self.transform = test_transform
        elif split == 'lab8':
            data_path = os.path.join(base_dir,'train_lab8.list')
            self.transform = tr_transform
        elif split == 'unlab8':
            data_path = os.path.join(base_dir,'train_unlab8.list')
            self.transform = test_transform
        elif split == 'lab16':
            data_path = os.path.join(base_dir,'train_lab16.list')
            self.transform = tr_transform
        elif split == 'unlab16':
            data_path = os.path.join(base_dir,'train_unlab16.list')
            self.transform = test_transform
        else:
            data_path = os.path.join(base_dir,'test.list')
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

        if self.args:
            if not self.args.no_strong_gamma:
                self.aug = VCompose([
                    RandomGamma(always_apply=True),
                ], p=1)
                print("strong gamma")
            if not self.args.no_strong_noise:
                self.aug = VCompose([
                    RandomGamma(always_apply=True),
                    GaussianNoise(),
                ], p=1)
                print("strong noise")
        else:
            self.aug = VCompose([
                    RandomGamma(always_apply=True),
                    GaussianNoise(),
                ], p=1)
            print("strong gamma and noise")

    def __len__(self):
        if self.split != 'test':
            return len(self.image_list) * self.aug_times
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        h5f = h5py.File(self.base_dir + "/2018LA_Seg_Training Set/" + image_path + "/mri_norm2.h5", 'r')
        image, label = h5f['image'][:], h5f['label'][:]

        image_weak = image.copy()
        image_strong = image.copy()
        sample_for_aug = {'image': image_strong, 'mask': label}
        sample_for_aug = self.aug(**sample_for_aug)
        image_strong = sample_for_aug['image']
        samples = image_weak, image_strong, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_weak, image_strong, label = tr_samples
        return {'image_weak':image_weak.float(), 'image_strong':image_strong.float(), 'label':label.long(), 'name':image_path, 'idx':idx % len(self.image_list)}



class StrongWeakBrats_PancreasStyle(Dataset):
    """ Pancreas Dataset """

    def __init__(self, base_dir, split, aug_times=5, args=None): # aug_times=5 follows https://github.com/VivienLu/UPCoL
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data')
        self.split = split
        self.aug_times = aug_times
        self.args = args

        tr_transform = Compose([
            RandomCrop((96, 96, 96)),
            ToTensor()
        ])
        test_transform = Compose([
            CenterCrop((96, 96, 96)),
            ToTensor()
        ])
            
        if split == 'lab':
            data_path = os.path.join(base_dir,'train_lab.list')
            self.transform = tr_transform
        elif split == 'unlab':
            data_path = os.path.join(base_dir,'train_unlab.list')
            self.transform = test_transform 
        elif split == 'train':
            data_path = os.path.join(base_dir,'train.list')
            self.transform = tr_transform
        elif split == 'lab25':
            data_path = os.path.join(base_dir,'train_lab25.list')
            self.transform = tr_transform
        elif split == 'lab250':
            data_path = os.path.join(base_dir,'train_lab250.list')
            self.transform = tr_transform
        elif split == 'unlab25':
            data_path = os.path.join(base_dir,'train_unlab25.list')
            self.transform = test_transform
        elif split == 'lab50':
            data_path = os.path.join(base_dir,'train_lab50.list')
            self.transform = tr_transform
        elif split == 'unlab50':
            data_path = os.path.join(base_dir,'train_unlab50.list')
            self.transform = test_transform
        elif split == 'lab100':
            data_path = os.path.join(base_dir,'train_lab100.list')
            self.transform = tr_transform
        elif split == 'unlab100':
            data_path = os.path.join(base_dir,'train_unlab100.list')
            self.transform = test_transform
        elif split == 'val':
            data_path = os.path.join(base_dir,'val.list')
            self.transform = test_transform
        elif split == 'test':
        # else:
            data_path = os.path.join(base_dir,'test.list')
            self.transform = test_transform

        with open(data_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [self.data_dir+ "/{}".format(item.strip()) + '.h5' for item in self.image_list]
        print("Split : {}, total {} samples".format(split, len(self.image_list)))

        if self.args:
            if not self.args.no_strong_gamma:
                self.aug = VCompose([
                    RandomGamma(always_apply=True),
                ], p=1)
                print("strong gamma")
            if not self.args.no_strong_noise:
                self.aug = VCompose([
                    RandomGamma(always_apply=True),
                    GaussianNoise(),
                ], p=1)
                print("strong noise")
        else:
            self.aug = VCompose([
                    RandomGamma(always_apply=True),
                    GaussianNoise(),
                ], p=1)
            print("strong gamma and noise")

    def __len__(self):
        if self.split != 'test':
            return len(self.image_list) * self.aug_times
        else:
            return len(self.image_list)

    def __getitem__(self, idx):
        image_path = self.image_list[idx % len(self.image_list)]
        h5f = h5py.File(image_path, 'r')
        image, label = h5f['image'][:], h5f['label'][:].astype(np.float32)

        image_weak = image.copy()
        image_strong = image.copy()
        sample_for_aug = {'image': image_strong, 'mask': label}
        sample_for_aug = self.aug(**sample_for_aug)
        image_strong = sample_for_aug['image']
        samples = image_weak, image_strong, label
        if self.transform:
            tr_samples = self.transform(samples)
        image_weak, image_strong, label = tr_samples
        return {'image_weak':image_weak.float(), 'image_strong':image_strong.float(), 'label':label.long(), 'name':image_path, 'idx':idx % len(self.image_list)}


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def _get_transform(self, label):
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 1, 0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = label.shape
        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        def do_transform(x):
            if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
                x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            x = x[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return x
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def _get_transform(self, x):
        if x.shape[0] <= self.output_size[0] or x.shape[1] <= self.output_size[1] or x.shape[2] <= self.output_size[2]:
            pw = max((self.output_size[0] - x.shape[0]) // 2 + 1, 0)
            ph = max((self.output_size[1] - x.shape[1]) // 2 + 1, 0)
            pd = max((self.output_size[2] - x.shape[2]) // 2 + 1, 0)
            x = np.pad(x, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
        else:
            pw, ph, pd = 0, 0, 0

        (w, h, d) = x.shape
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        def do_transform(image):
            if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= self.output_size[2]:
                try:
                    image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
                except Exception as e:
                    print(e)
            image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return image
        return do_transform

    def __call__(self, samples):
        transform = self._get_transform(samples[0])
        return [transform(s) for s in samples]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image_weak = sample[0]
        image_weak = image_weak.reshape(1, image_weak.shape[0], image_weak.shape[1], image_weak.shape[2]).astype(np.float32)
        image_strong = sample[1]
        image_strong = image_strong.reshape(1, image_strong.shape[0], image_strong.shape[1], image_strong.shape[2]).astype(np.float32)

        sample = [image_weak, image_strong] + [*sample[2:]]
        return [torch.from_numpy(s.astype(np.float32)) for s in sample]


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image_weak'], sample['label']
        image, label = random_rot_flip(image, label)
        sample['image_weak'] = image
        sample['label'] = label
        if 'image_strong' in sample:
            image_strong = sample['image_strong']
            image_strong, _ = random_rot_flip(image_strong, label)
            sample['image_strong'] = image_strong
        return sample

if __name__ == '__main__':
    base_dir = '/mnt/zhen_chen/xyliu/preprocess/codes'

    labset = StrongWeakPancreas(base_dir, split='lab')
    unlabset = StrongWeakPancreas(base_dir, split='unlab')
    trainset = StrongWeakPancreas(base_dir, split='train')
    testset = StrongWeakPancreas(base_dir, split='test')

    lab_sample = labset[0]
    unlab_sample = unlabset[0]
    train_sample = trainset[0]
    test_sample = testset[0]

    print(len(labset), lab_sample['image_weak'].shape, lab_sample['image_strong'].shape, lab_sample['label'].shape)
    print(len(unlabset), unlab_sample['image_weak'].shape, unlab_sample['image_strong'].shape, unlab_sample['label'].shape)
    print(len(trainset), train_sample['image_weak'].shape, train_sample['image_strong'].shape, train_sample['label'].shape)
    print(len(testset), test_sample['image_weak'].shape, test_sample['image_strong'].shape, test_sample['label'].shape)