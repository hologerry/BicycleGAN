import os.path
import random

from PIL import Image

from data.base_dataset import BaseDataset, transform_fusion
from data.image_folder import make_dataset


class CnMultiFusionDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def rreplace(self, s, old, new, occurrence):
        li = s.rsplit(old, occurrence)
        return new.join(li)

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)
        self.ABC_paths = sorted(make_dataset(self.dir_ABC))
        # self.chars = list(range(639))
        # guarantee consistent for test
        self.shuffled_gb639list = [79, 279, 633, 632, 508, 378, 283, 462, 574, 251,
                                   267, 102, 181, 100, 415, 94, 261, 330, 501, 434,
                                   418, 160, 119, 207, 44, 493, 290, 590, 73, 613,
                                   244, 618, 516, 40, 442, 48, 218, 61, 585, 411,
                                   518, 174, 543, 324, 424, 312, 302, 466, 541, 379,
                                   111, 557, 256, 564, 338, 129, 118, 214, 395, 11]
        assert(opt.few_size <= 60)
        self.chars = self.shuffled_gb639list[:opt.few_size]

    def __getitem__(self, index):
        ABC_path = self.ABC_paths[index]
        ABC = Image.open(ABC_path).convert('RGB')
        w3, h = ABC.size
        w = int(w3 / 3)
        A = ABC.crop((0, 0, w, h))
        B = ABC.crop((w, 0, w+w, h))
        C = ABC.crop((w+w, 0, w+w+w, h))
        Shapes = []
        Shape_paths = []
        Colors = []
        Color_paths = []
        if self.opt.nencode > 1:
            target_char = ABC_path.split('_')[-1].split('.')[0]
            ABC_path_c = ABC_path
            # for shapes
            random.shuffle(self.chars)
            chars_random = self.chars[:self.opt.nencode]
            for char in chars_random:
                s_path = self.rreplace(ABC_path_c, target_char, str(
                    char), 1)  # /path/to/img/X_XX_XXX.png
                Shape_paths.append(s_path)
                Shapes.append(Image.open(s_path).convert(
                    'RGB').crop((w, 0, w+w, h)))
            # for colors
            random.shuffle(self.chars)
            chars_random = self.chars[:self.opt.nencode]
            for char in chars_random:
                c_path = self.rreplace(ABC_path_c, target_char, str(
                    char), 1)  # /path/to/img/X_XX_XXX.png
                Color_paths.append(c_path)
                Colors.append(Image.open(c_path).convert(
                    'RGB').crop((w+w, 0, w+w+w, h)))

        else:
            Shapes.append(B)
            Shape_paths.append(ABC_path)
            Colors.append(C)
            Color_paths.append(ABC_path)
        A, B, C, Shapes, Colors = transform_fusion(
            self.opt, A, B, C, Shapes, Colors)

        # A is the reference, B is the gray shape, C is the gradient
        return {'A': A, 'B': B, 'C': C, 'Shapes': Shapes, 'Colors': Colors,
                'ABC_path': ABC_path, 'Shape_paths': Shape_paths, 'Color_paths': Color_paths}

    def __len__(self):
        return len(self.ABC_paths)

    def name(self):
        return 'CnMultiFusionDataset'
