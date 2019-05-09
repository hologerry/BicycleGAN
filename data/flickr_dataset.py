import os.path
import random

from PIL import Image, ImageFilter
import torch

from data.base_dataset import BaseDataset, transform_multi
from data.image_folder import make_dataset


class FlickrDataset(BaseDataset):
    """Pretrain dataset
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)
        self.ABC_paths = sorted(make_dataset(self.dir_ABC))
        self.idxs = [i for i in range(len(self.ABC_paths))]

    def __getitem__(self, index):
        ABC_path = self.ABC_paths[index]
        ABC = Image.open(ABC_path).convert('RGB')
        w3, h = ABC.size
        w = int(w3 / 3)
        A = ABC.crop((0, 0, w, h))
        B = ABC.crop((w, 0, w+w, h))
        C = ABC.crop((w+w, 0, w+w+w, h))

        Bases = []
        Shapes = []
        Colors = []

        Style_paths = []

        blur_Shapes = []
        blur_Colors = []

        random.shuffle(self.idxs)
        random_idxs = self.idxs[:self.opt.nencode]
        for idx in random_idxs:
            if self.opt.phase == 'val':
                idx += 39618  # train size
            s_path = self.ABC_paths[idx]
            Style_paths.append(s_path)
            Bases.append(Image.open(s_path).convert('RGB').crop((0, 0, w, h)))
            Shapes.append(Image.open(s_path).convert('RGB').crop((w, 0, w+w, h)))
            Colors.append(Image.open(s_path).convert('RGB').crop((w+w, 0, w+w+w, h)))

            blur_Shapes.append(
                Image.open(s_path).convert('RGB').crop((w, 0, w+w, h)).filter(
                    ImageFilter.GaussianBlur(radius=(random.random()*2+2)))
                )

            blur_Colors.append(
                Image.open(s_path).convert('RGB').crop((w+w, 0, w+w+w, h)).filter(
                    ImageFilter.GaussianBlur(radius=(random.random()*2+2)))
                )

        A, B, C, Bases, Shapes, Colors, blur_Shapes, blur_Colors = \
            transform_multi(self.opt, A, B, C, Bases, Shapes, Colors, blur_Shapes, blur_Colors)
        C_l = C
        label = torch.tensor(1.0)
        B_G = B
        C_G = C

        # A is the reference, B is the gray shape, C is the gradient
        return {'A': A, 'B': B, 'B_G': B_G, 'C': C, 'C_G': C_G, 'C_l': C_l, 'label': label,
                'Bases': Bases, 'Shapes': Shapes, 'Colors': Colors,
                'blur_Shapes': blur_Shapes, 'blur_Colors': blur_Colors,
                'ABC_path': ABC_path, 'Style_paths': Style_paths,
                }

    def __len__(self):
        return len(self.ABC_paths)

    def name(self):
        return 'FlickrDataset'
