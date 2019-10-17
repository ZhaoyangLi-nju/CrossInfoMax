import os.path
import random

import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
from torchvision.datasets.folder import find_classes
from torchvision.datasets.folder import make_dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torchvision.transforms import functional as F
import copy
import numbers


class AlignedConcDataset:

    def __init__(self, cfg, data_dir=None, transform=None, labeled=True):
        self.cfg = cfg
        self.transform = transform
        self.data_dir = data_dir
        self.labeled = labeled

        if labeled:
            self.classes, self.class_to_idx = find_classes(self.data_dir)
            self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
            self.imgs = make_dataset(self.data_dir, self.class_to_idx, ['jpg','png'])
        else:
            self.imgs = get_images(self.data_dir, ['jpg', 'png'])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc = Image.open(img_path).convert('RGB')

        # split RGB and Depth as A and B
        w, h = AB_conc.size
        w2 = int(w / 2)
        if w2 > self.cfg.FINE_SIZE:
            A = AB_conc.crop((0, 0, w2, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
            B = AB_conc.crop((w2, 0, w, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
        else:
            A = AB_conc.crop((0, 0, w2, h))
            B = AB_conc.crop((w2, 0, w, h))

        if self.labeled:
            sample = {'A': A, 'B': B, 'img_name': img_name, 'label': label, 'index': index}
        else:
            sample = {'A': A, 'B': B, 'img_name': img_name, 'index':index}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        if self.padding > 0:
            A = F.pad(A, self.padding)
            B = F.pad(B, self.padding)

        # pad the width if needed
        if self.pad_if_needed and A.size[0] < self.size[1]:
            A = F.pad(A, (int((1 + self.size[1] - A.size[0]) / 2), 0))
            B = F.pad(B, (int((1 + self.size[1] - B.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and A.size[1] < self.size[0]:
            A = F.pad(A, (0, int((1 + self.size[0] - A.size[1]) / 2)))
            B = F.pad(B, (0, int((1 + self.size[0] - B.size[1]) / 2)))

        i, j, h, w = self.get_params(A, self.size)
        sample['A'] = F.crop(A, i, j, h, w)
        sample['B'] = F.crop(B, i, j, h, w)

        # _i, _j, _h, _w = self.get_params(A, self.size)
        # sample['A'] = F.crop(A, i, j, h, w)
        # sample['B'] = F.crop(B, _i, _j, _h, _w)

        return sample


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        sample['A'] = F.center_crop(A, self.size)
        sample['B'] = F.center_crop(B, self.size)
        return sample


class FiveCrop(transforms.FiveCrop):

    def __call__(self, sample):

        A, B = sample['A'], sample['B']
        sample['A'] = F.five_crop(A, self.size)
        sample['B'] = F.five_crop(B, self.size)

        result = []
        list_A = F.five_crop(A, self.size)
        list_B = F.five_crop(B, self.size)
        for item in zip(list_A, list_B):
            _sample = copy.deepcopy(sample)
            _sample['A'] = item[0]
            _sample['B'] = item[1]
            result.append(_sample)
            # item[0].show()
            # item[1].show()
        return result

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, sample):
        A, B = sample['A'], sample['B']
        if random.random() > 0.5:
            A = F.hflip(A)
            B = F.hflip(B)

        sample['A'] = A
        sample['B'] = B

        return sample


class Resize(transforms.Resize):

    def __call__(self, sample):

        A, B = sample['A'], sample['B']
        h = self.size[0]
        w = self.size[1]

        sample['A'] = F.resize(A, (h, w))
        sample['B'] = F.resize(B, (h, w))

        return sample


class MultiScale(object):

    def __init__(self, size, scale_times=5):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_times = scale_times

    def __call__(self, sample):
        h = self.size[0]
        w = self.size[1]
        A, B = sample['A'], sample['B']
        # sample['A'] = [
        #    F.resize(A, (h, w)), F.resize(A, (int(h / 2), int(w / 2))),
        #    F.resize(A, (int(h / 4), int(w / 4))), F.resize(A, (int(h / 8), int(w / 8))),
        #    F.resize(A, (int(h / 16), int(w / 16)))
        # ]
        # sample['B'] = [
        #     F.resize(B, (h, w)), F.resize(B, (int(h / 2), int(w / 2))),
        #     F.resize(B, (int(h / 4), int(w / 4))), F.resize(B, (int(h / 8), int(w / 8))),
        #     F.resize(B, (int(h / 16), int(w / 16))), F.resize(B, (int(h / 32), int(w / 32)))
        # ]

        # sample['A'] = [F.resize(A, (int(h / pow(2, i)), int(w / pow(2, i)))) for i in range(self.scale_times)]
        sample['B'] = [F.resize(B, (int(h / pow(2, i)), int(w / pow(2, i)))) for i in range(self.scale_times)]

        return sample


class ToTensor(object):
    def __call__(self, sample):

        A, B = sample['A'], sample['B']

        sample['A'] = F.to_tensor(A)
        if isinstance(B, list):
            # sample['A'] = [F.to_tensor(item) for item in A]
            sample['B'] = [F.to_tensor(item) for item in B]
        else:
            sample['B'] = F.to_tensor(B)

        return sample

class ToTensor_LAB(object):
    def __call__(self, sample):

        A, B = sample['A'], sample['B']

        if isinstance(B, list):
            sample['A'] = [F.to_tensor(item) for item in A]
            sample['B'] = [F.to_tensor(item) for item in B]
        else:
            sample['A'] = F.to_tensor(A)
            sample['B'] = F.to_tensor(B)

        sample['A'] = sample['A'][[0], ...] / 50.0 - 1.0
        sample['B'] = sample['A'][[1, 2], ...] / 110.0

        return sample


class Normalize(transforms.Normalize):

    def __call__(self, sample):
        A, B = sample['A'], sample['B']

        sample['A'] = F.normalize(A, self.mean, self.std)
        if isinstance(B, list):
            # sample['A'] = [F.normalize(item, self.mean, self.std) for item in A]
            sample['B'] = [F.normalize(item, self.mean, self.std) for item in B]
        else:
            sample['B'] = F.normalize(B, self.mean, self.std)

        return sample

class MultiScale(object):

    def __init__(self, size, scale_times=4):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_times = scale_times

    def __call__(self, sample):
        h = self.size[0]
        w = self.size[1]
        A, B = sample['A'], sample['B']

        # sample['A'] = [F.resize(A, (int(h / pow(2, i)), int(w / pow(2, i)))) for i in range(self.scale_times)]
        sample['B'] = [F.resize(B, (int(h / pow(2, i)), int(w / pow(2, i)))) for i in range(self.scale_times)]

        return sample

class Lambda(transforms.Lambda):

    def __call__(self, sample):
        return self.lambd(sample)

class Stack(object):

    def __call__(self, samples):
        list_A = []
        list_B = []
        sample = copy.deepcopy(samples[0])
        for item in samples:
            A, B = item['A'], item['B']
            list_A.append(A)
            list_B.append(B)

        sample['A'] = torch.stack(list_A)
        sample['B'] = torch.stack(list_B)
        return sample

def get_images(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)
    image_names = [d for d in os.listdir(dir)]
    for image_name in image_names:
        if has_file_allowed_extension(image_name, extensions):
            file = os.path.join(dir, image_name)
            images.append(file)
    return images

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)