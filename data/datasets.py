# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import hashlib
import io
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .dct import DCT_base_Rec_Module

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import kornia.augmentation as K

Perturbations = K.container.ImageSequential(
    K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 3.0), p=0.1),
    K.RandomJPEG(jpeg_quality=(30, 100), p=0.1)
)

transform_before = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: Perturbations(x)[0])
    ]
)
transform_before_test = transforms.Compose([
    transforms.ToTensor(),
    ]
)

transform_train = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

transform_test_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def _list_subdirs(root):
    return [
        name for name in os.listdir(root)
        if os.path.isdir(os.path.join(root, name))
    ]


def _has_binary_layout(root):
    subdirs = set(_list_subdirs(root))
    return '0_real' in subdirs and '1_fake' in subdirs


def _append_binary_dir(root, out_list):
    subdirs = set(_list_subdirs(root))
    missing = {'0_real', '1_fake'} - subdirs
    if missing:
        raise ValueError(f'Expected 0_real/1_fake under {root}, missing: {sorted(missing)}')

    for image_path in os.listdir(os.path.join(root, '0_real')):
        out_list.append({"image_path": os.path.join(root, '0_real', image_path), "label": 0})
    for image_path in os.listdir(os.path.join(root, '1_fake')):
        out_list.append({"image_path": os.path.join(root, '1_fake', image_path), "label": 1})


def _scan_progan_style(root, out_list):
    """Walk a directory in the standard ProGAN/CNNSpot layout and append
    {image_path, label} dicts to out_list. Root may contain subfolders
    that each hold `0_real`/`1_fake`, or directly contain `0_real`/`1_fake`.
    """
    if _has_binary_layout(root):
        _append_binary_dir(root, out_list)
    else:
        for folder_name in _list_subdirs(root):
            _append_binary_dir(os.path.join(root, folder_name), out_list)


def _sha1_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha1()
    with open(path, 'rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sample_key(sample_path, root, mode):
    if mode == 'name':
        return os.path.basename(sample_path)
    if mode == 'relative':
        return os.path.relpath(sample_path, root).replace('\\', '/')
    if mode == 'sha1':
        return _sha1_file(sample_path)
    raise ValueError(f'Unsupported dedup mode: {mode}')


def _parse_reference_roots(reference_arg):
    if not reference_arg:
        return []
    return [item.strip() for item in reference_arg.split(',') if item.strip()]


def _parse_subset_names(subset_arg):
    if not subset_arg:
        return []
    return [item.strip() for item in subset_arg.split(',') if item.strip()]


def _iter_dataset_samples(root, subset_arg=None):
    samples = []
    subsets = _parse_subset_names(subset_arg)

    if not subsets:
        _scan_progan_style(root, samples)
        return samples

    if _has_binary_layout(root):
        root_name = os.path.basename(os.path.normpath(root))
        if root_name in subsets:
            _append_binary_dir(root, samples)
        return samples

    missing = []
    for subset in subsets:
        subset_root = os.path.join(root, subset)
        if not os.path.isdir(subset_root):
            missing.append(subset)
            continue
        _append_binary_dir(subset_root, samples)
    if missing:
        raise ValueError(f'Diffusion subsets not found under {root}: {missing}')
    return samples


def _filter_duplicates(candidate_list, candidate_root, reference_roots, mode):
    if mode == 'none' or not reference_roots:
        return candidate_list, 0

    reference_keys = set()
    for ref_root in reference_roots:
        ref_samples = _iter_dataset_samples(ref_root)
        for sample in ref_samples:
            reference_keys.add(_sample_key(sample['image_path'], ref_root, mode))

    filtered = []
    removed = 0
    for sample in candidate_list:
        sample_key = _sample_key(sample['image_path'], candidate_root, mode)
        if sample_key in reference_keys:
            removed += 1
            continue
        filtered.append(sample)

    return filtered, removed


class TrainDataset(Dataset):
    def __init__(self, is_train, args):

        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        if 'GenImage' in root and root.split('/')[-1] != 'train':
            file_path = root

            if _has_binary_layout(file_path):
                _append_binary_dir(file_path, self.data_list)
            else:
                for folder_name in _list_subdirs(file_path):
                    _append_binary_dir(os.path.join(file_path, folder_name), self.data_list)
        else:

            for filename in _list_subdirs(root):

                file_path = os.path.join(root, filename)

                if _has_binary_layout(file_path):
                    _append_binary_dir(file_path, self.data_list)
                else:
                    for folder_name in _list_subdirs(file_path):
                        _append_binary_dir(os.path.join(file_path, folder_name), self.data_list)

        # Optional: mix in a fraction of Diffusion samples for E3/E4.
        diffusion_path = getattr(args, 'diffusion_path', None)
        mix_ratio = getattr(args, 'mix_ratio', 0.0)
        if is_train and diffusion_path is not None and 0.0 < mix_ratio < 1.0:
            progan_list = self.data_list
            diffusion_subsets = getattr(args, 'diffusion_subsets', None)
            diffusion_list = _iter_dataset_samples(diffusion_path, diffusion_subsets)
            if diffusion_subsets:
                print(f'[TrainDataset] diffusion subsets: {diffusion_subsets}')

            dedup_mode = getattr(args, 'dedup_mode', 'none')
            reference_roots = _parse_reference_roots(getattr(args, 'dedup_reference_path', None))
            diffusion_list, removed = _filter_duplicates(
                diffusion_list, diffusion_path, reference_roots, dedup_mode
            )
            if removed > 0:
                print(f'[TrainDataset] dedup removed {removed} diffusion samples using mode={dedup_mode}')

            n_diff = int(len(progan_list) * mix_ratio / (1.0 - mix_ratio))
            if len(diffusion_list) < n_diff:
                print(f'[TrainDataset] WARNING: diffusion pool size {len(diffusion_list)} '
                      f'< requested {n_diff}, using all available.')
                sampled_diffusion = diffusion_list
            else:
                sampled_diffusion = random.sample(diffusion_list, n_diff)

            self.data_list = progan_list + sampled_diffusion
            total = len(self.data_list)
            ratio = len(sampled_diffusion) / total if total > 0 else 0.0
            print(
                f'[TrainDataset] progan={len(progan_list)}, '
                f'diffusion={len(sampled_diffusion)}, ratio={ratio:.3f}'
            )

        # Shuffle data_list to mix real and fake samples
        if is_train:
            random.shuffle(self.data_list)

        self.dct = DCT_base_Rec_Module()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        sample = self.data_list[index]

        image_path, targets = sample['image_path'], sample['label']

        try:
            image = Image.open(image_path).convert('RGB')
        except:
            print(f'image error: {image_path}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        image = transform_before(image)

        try:
            x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)
        except:
            print(f'image error: {image_path}, c, h, w: {image.shape}')
            return self.__getitem__(random.randint(0, len(self.data_list) - 1))

        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin)
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1)
        x_maxmax1 = transform_train(x_maxmax1)

        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))


class TestDataset(Dataset):
    def __init__(self, is_train, args):

        root = args.data_path if is_train else args.eval_data_path

        self.data_list = []

        file_path = root

        if _has_binary_layout(file_path):
            _append_binary_dir(file_path, self.data_list)
        else:
            for folder_name in _list_subdirs(file_path):
                _append_binary_dir(os.path.join(file_path, folder_name), self.data_list)

        self.dct = DCT_base_Rec_Module()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):

        sample = self.data_list[index]

        image_path, targets = sample['image_path'], sample['label']

        image = Image.open(image_path).convert('RGB')

        image = transform_before_test(image)

        # x_max, x_min, x_max_min, x_minmin = self.dct(image)

        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(image)

        x_0 = transform_train(image)
        x_minmin = transform_train(x_minmin)
        x_maxmax = transform_train(x_maxmax)

        x_minmin1 = transform_train(x_minmin1)
        x_maxmax1 = transform_train(x_maxmax1)

        return torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0], dim=0), torch.tensor(int(targets))
