import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'none':
        transform_list = []

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomVerticalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def transform_pair(opt, A, B):
    if opt.resize_or_crop == 'resize_and_crop':
        A = A.resize((opt.loadSize, opt.loadSize), Image.BICUBIC)
        B = B.resize((opt.loadSize, opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, opt.loadSize - opt.fineSize - 1))
        h_offset = random.randint(0, max(0, opt.loadSize - opt.fineSize - 1))

        A = A[:, h_offset:h_offset + opt.fineSize, w_offset:w_offset + opt.fineSize]
        B = B[:, h_offset:h_offset + opt.fineSize, w_offset:w_offset + opt.fineSize]

    elif opt.resize_or_crop == 'none':
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
    else:
        raise ValueError("For paired data, only support resize_and_crop or none.")

    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

    if (not opt.no_flip) and random.random() < 0.5:
        idx = [i for i in range(A.size(2) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        A = A.index_select(2, idx)
        B = B.index_select(2, idx)

    return A, B


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
