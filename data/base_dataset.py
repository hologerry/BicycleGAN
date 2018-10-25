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


def transform_pair(opt, A, B, C=None):
    if opt.resize_or_crop == 'resize_and_crop':
        A = A.resize((opt.loadSize, opt.loadSize), Image.BICUBIC)
        B = B.resize((opt.loadSize, opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        w_offset = random.randint(0, max(0, opt.loadSize - opt.fineSize - 1))
        h_offset = random.randint(0, max(0, opt.loadSize - opt.fineSize - 1))

        A = A[:, h_offset:h_offset + opt.fineSize,
              w_offset:w_offset + opt.fineSize]
        B = B[:, h_offset:h_offset + opt.fineSize,
              w_offset:w_offset + opt.fineSize]
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        new_C = []
        if C is not None:
            assert(isinstance(C, list))
            for one_C in C:
                one_C = one_C.resize(
                    (opt.loadSize, opt.loadSize), Image.BICUBIC)
                one_C = transforms.ToTensor()(one_C)
                one_C = transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(one_C)
                one_C = one_C[:, h_offset:h_offset +
                              opt.fineSize, w_offset:w_offset + opt.fineSize]
                new_C.append(one_C)
            C = torch.cat(new_C)
    elif opt.resize_or_crop == 'none':
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
        if C is not None:
            assert(isinstance(C, list))
            C = list(map(lambda c: transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(c)), C))
            C = torch.cat(C)
    else:
        raise ValueError(
            "For paired data, only support resize_and_crop or none --resise_or_crop mode")

    if (not opt.no_flip) and random.random() < 0.5:
        idx = [i for i in range(A.size(2) - 1, -1, -1)]
        idx = torch.LongTensor(idx)
        A = A.index_select(2, idx)
        B = B.index_select(2, idx)
        if C is not None:
            C = C.index_select(2, idx)

    if C is not None:
        return A, B, C
    else:
        return A, B


def transform_fusion(opt, A, B, C, Colors, Shapes=None):
    if not opt.resize_or_crop == 'none':
        raise ValueError(
            "Only support none mode for resize_or_crop on base_gray_color dataset")
    assert(isinstance(Shapes, list))
    assert(isinstance(Colors, list))
    A = transforms.ToTensor()(A)
    B = transforms.ToTensor()(B)
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
    B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)
    Colors = list(map(lambda c: transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(c)), Colors))
    Colors = torch.cat(Colors)
    if Shapes is not None:
        Shapes = list(map(lambda s: transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(s)), Shapes))
        Shapes = torch.cat(Shapes)

        return A, B, C, Colors, Shapes
    else:
        return A, B, C, Colors

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
