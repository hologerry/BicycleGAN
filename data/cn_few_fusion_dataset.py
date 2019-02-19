import os.path
import random

from PIL import Image, ImageFilter

from data.base_dataset import BaseDataset, transform_few_with_label
from data.image_folder import make_dataset


class CnFewFusionDataset(BaseDataset):
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
        # self.chars = list(range(500))  # only use 500 of 639 to train, and the remain 139 as test set
        # guarantee consistent for test
        # so just shuffle 500 once
        self.shuffled_gb639list = [445, 10, 116, 113, 133, 365,
                                   297, 368, 207, 241, 450, 486,
                                   252, 301, 308, 121, 382, 269,
                                   112, 356, 418, 197, 381, 205,
                                   430, 236, 149, 435, 291, 455]
        assert(opt.few_size <= len(self.shuffled_gb639list))
        self.chars = self.shuffled_gb639list[:opt.few_size]

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

        target_char = int(ABC_path.split('_')[-1].split('.')[0])
        ABC_path_c = ABC_path
        label = 0.0
        if target_char in self.chars:
            label = 1.0
        # for shapes
        random.shuffle(self.chars)
        chars_random = self.chars[:self.opt.nencode]
        for char in chars_random:
            s_path = self.rreplace(ABC_path_c, str(target_char), str(char), 1)  # /path/to/img/XXXX_XX_XXX.png
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

        A, B, B_G, C, C_G, C_l, Bases, Shapes, Colors, blur_Shapes, blur_Colors = \
            transform_few_with_label(self.opt, A, B, C, label, Bases, Shapes, Colors, blur_Shapes, blur_Colors)

        # A is the reference, B is the gray shape, C is the gradient
        return {'A': A, 'B': B, 'B_G': B_G, 'C': C, 'C_G': C_G, 'C_l': C_l, 'label': label,
                'Bases': Bases, 'Shapes': Shapes, 'Colors': Colors,
                'blur_Shapes': blur_Shapes, 'blur_Colors': blur_Colors,
                'ABC_path': ABC_path, 'Style_paths': Style_paths,
                }

    def __len__(self):
        return len(self.ABC_paths)

    def name(self):
        return 'CnFewFusionDataset'
