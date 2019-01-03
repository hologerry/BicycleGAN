import os.path
import random

from PIL import Image

from data.base_dataset import BaseDataset, transform_fusion, transform_vgg
from data.image_folder import make_dataset


class UnpairedFewFusionDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)
        self.ABC_paths = sorted(make_dataset(self.dir_ABC))
        # self.few_alphas = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
        self.few_alphas = ['0', '1', '2', '3', '4']
        assert(self.opt.nencode == (len(self.few_alphas)-1))

    def __getitem__(self, index):
        ABC_path = self.A_paths[index]
        ABC = Image.open(ABC_path).convert('RGB')
        w3, h = ABC.size
        w = int(w3 / 3)
        A = ABC.crop((0, 0, w, h))
        B = ABC.crop((w, 0, w+w, h))
        C = ABC.crop((w+w, 0, w+w+w, h))
        Shapes = []
        Colors = []
        Style_paths = []
        ABC_path_list = list(ABC_path)
        # for shapes
        random.shuffle(self.few_alphas)
        chars_random = self.few_alphas[:self.opt.nencode]
        for char in chars_random:
            ABC_path_list[-5] = char  # /path/to/img/XXXX_X_X.png
            style_path = "".join(ABC_path_list)
            style_path = style_path.replace('train', 'style')
            Style_paths.append(style_path)
            Shapes.append(Image.open(style_path).convert('RGB').crop((w, 0, w+w, h)))
            Colors.append(Image.open(style_path).convert('RGB').crop((w+w, 0, w+w+w, h)))

        vgg_Shapes, vgg_Colors = transform_vgg(Shapes, Colors)
        A, B, C, Shapes, Colors = transform_fusion(self.opt, A, B, C, Shapes, Colors)

        # A is the reference, B is the gray shape, C is the gradient
        return {'A': A, 'B': B, 'C': C, 'Shapes': Shapes, 'Colors': Colors,
                'ABC_path': ABC_path, 'Style_paths': Style_paths, 'vgg_Shapes': vgg_Shapes, 'vgg_Colors':vgg_Colors}

    def __len__(self):
        return len(self.ABC_paths)

    def name(self):
        return 'UnpairedFewFusionDataset'
