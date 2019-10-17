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
from skimage import color
import numpy as np

class SPL10_Dataset:

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
        image = Image.open(img_path).convert('RGB')

        # RGB as image
        # w, h = image.size
        # if w > self.cfg.FINE_SIZE:
        #     A = AB_conc.crop((w, h)).resize((self.cfg.LOAD_SIZE, self.cfg.LOAD_SIZE), Image.BICUBIC)
        # else:
        #     A = AB_conc.crop((w, h))

        if self.labeled:
            sample = {'image': image, 'label': label, 'index': index}
        else:
            sample = {'image': image, 'index':index}

        if self.transform:
            sample = self.transform(sample)
        return sample


class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        image = sample['image']

        if self.padding > 0:
            image = F.pad(image, self.padding)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (int((1 + self.size[1] - image.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, int((1 + self.size[0] - image.size[1]) / 2)))

        i, j, h, w = self.get_params(image, self.size)
        sample['image'] = F.crop(image, i, j, h, w)

        # _i, _j, _h, _w = self.get_params(A, self.size)
        # sample['A'] = F.crop(A, i, j, h, w)
        # sample['B'] = F.crop(B, _i, _j, _h, _w)

        return sample


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        image = sample['image']
        sample['image'] = F.center_crop(image, self.size)
        return sample


class FiveCrop(transforms.FiveCrop):

    def __call__(self, sample):

        image = sample['image']
        sample['image'] = F.five_crop(image, self.size)

        result = []
        list_image = F.five_crop(image, self.size)
        for item in zip(list_image):
            _sample = copy.deepcopy(sample)
            _sample['image'] = item[0]
            result.append(_sample)
    
        return result

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, sample):
        image = sample['image']
        if random.random() > 0.5:
            image = F.hflip(image)
        sample['image'] = image

        return sample


class Resize(transforms.Resize):

    def __call__(self, sample):

        image = sample['image']
        h = self.size[0]
        w = self.size[1]

        sample['image'] = F.resize(image, (h, w))

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
        image = sample['image']

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
        sample['image'] = [F.resize(image, (int(h / pow(2, i)), int(w / pow(2, i)))) for i in range(self.scale_times)]

        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample['image'] = F.to_tensor(sample['image'])
        if isinstance(sample['lab'], list):
            for i in range(len(sample['lab'])):
                sample['lab'][i] = F.to_tensor(sample['lab'][i])
        else:
             sample['lab'] = F.to_tensor(sample['lab'])
        # if isinstance(B, list):
        #     # sample['A'] = [F.to_tensor(item) for item in A]
        #     sample['B'] = [F.to_tensor(item) for item in B]
        # else:
        #     sample['B'] = F.to_tensor(B)

        return sample

# class ToTensor_LAB(object):
#     def __call__(self, sample):

#         A, B = sample['A'], sample['B']

#         if isinstance(B, list):
#             sample['A'] = [F.to_tensor(item) for item in A]
#             sample['B'] = [F.to_tensor(item) for item in B]
#         else:
#             sample['A'] = F.to_tensor(A)
#             sample['B'] = F.to_tensor(B)

#         sample['A'] = sample['A'][[0], ...] / 50.0 - 1.0
#         sample['B'] = sample['A'][[1, 2], ...] / 110.0

#         return sample


class Normalize(transforms.Normalize):

    def __call__(self, sample):
        sample['image'] = (F.normalize(sample['image'], self.mean, self.std)).float()
        if isinstance(sample['lab'], list):
            for i in range(len(sample['lab'])):
                sample['lab'][i] = (F.normalize(sample['lab'][i], self.mean, self.std)).float()
        else:
            sample['lab'] = (F.normalize(sample['lab'], self.mean, self.std)).float()
        # if isinstance(B, list):
        #     # sample['A'] = [F.normalize(item, self.mean, self.std) for item in A]
        #     sample['B'] = [F.normalize(item, self.mean, self.std) for item in B]
        # else:
        #     sample['B'] = F.normalize(B, self.mean, self.std)
        return sample

# class MultiScale(object):#needing change more time

#     def __init__(self, size, scale_times=4):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#         self.scale_times = scale_times

#     def __call__(self, sample):
#         h = self.size[0]
#         w = self.size[1]
#         A, B = sample['A'], sample['B']

#         # sample['A'] = [F.resize(A, (int(h / pow(2, i)), int(w / pow(2, i)))) for i in range(self.scale_times)]
#         sample['B'] = [F.resize(B, (int(h / pow(2, i)), int(w / pow(2, i)))) for i in range(self.scale_times)]

#         return sample

class Lambda(transforms.Lambda):

    def __call__(self, sample):
        return self.lambd(sample)




class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, sample):
        if isinstance(sample['image'], list):
            LAB=[]
            for i in range(len(sample['image'])):
                if i == 0:
                    image = sample['image'][0]
                img = sample['image'][i]
                img = np.asarray(img, np.uint8)
                img = color.rgb2lab(img)
                LAB.append(img)
            sample={'image':image,'lab':LAB,'label':sample['label']}
        else:
            image = sample['image']
            img = sample['image']
            img = np.asarray(img, np.uint8)
            img = color.rgb2lab(img)
            LAB = img
            sample={'image':image,'lab':LAB,'label':sample['label']}

        return sample
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)