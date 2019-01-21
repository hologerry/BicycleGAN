import os.path
import random

from PIL import Image

from data.base_dataset import BaseDataset, transform_triple_with_label, transform_vgg
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
        with open(os.path.join(opt.dataroot, "few_dict.txt")) as f:
            self.few_dict = f.readlines()
        assert(self.opt.nencode == (len(self.few_alphas)-1))

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
        ABC_path_list = list(ABC_path)
        target_font = int(ABC_path.split("/")[-1].split("_")[0])
        target_char = ABC_path_list[-5]
        label = 0.0
        if target_char in self.few_dict[target_font-11000].strip():
            label = 1.0
        # for shapes
        random.shuffle(self.few_alphas)
        chars_random = self.few_alphas[:self.opt.nencode]
        for char in chars_random:
            ABC_path_list[-5] = char  # /path/to/img/XXXX_X.png
            phase_path = "".join(ABC_path_list)
            style_path = phase_path.replace(self.opt.phase, 'style')
            Style_paths.append(style_path)
            Bases.append(Image.open(style_path).convert('RGB').crop((0, 0, w, h)))
            Shapes.append(Image.open(style_path).convert('RGB').crop((w, 0, w+w, h)))
            Colors.append(Image.open(style_path).convert('RGB').crop((w+w, 0, w+w+w, h)))

        B = 0.299 * C[..., 0] + 0.587 * C[..., 1] + 0.114 * C[..., 2]
        B = B.unsqueeze(2)
        B = torch.cat([B, B, B], dim=1)

        vgg_Shapes, vgg_Colors = transform_vgg(Shapes, Colors)

        vgg_Shapes = 0.299 * vgg_Colors[..., 0] + 0.587 * vgg_Colors[..., 1] + 0.114 * vgg_Colors[..., 2]
        vgg_Shapes = vgg_Shapes.unsqueeze(2)
        vgg_Shapes = torch.cat([vgg_Shapes, vgg_Shapes, vgg_Shapes], dim=1)

        # A, B, C, Shapes, Colors = transform_fusion(self.opt, A, B, C, Shapes, Colors)
        A, B, B_G, C, C_G, C_l, label, Bases, Shapes, Colors = \
            transform_triple_with_label(self.opt, A, B, C, label, Bases, Shapes, Colors)

        # A is the reference, B is the gray shape, C is the gradient
        return {'A': A, 'B': B, 'B_G': B_G, 'C': C, 'C_G': C_G, 'C_l': C_l, 'label': label,
                'Bases': Bases, 'Shapes': Shapes, 'Colors': Colors,
                'ABC_path': ABC_path, 'Style_paths': Style_paths,
                'vgg_Shapes': vgg_Shapes, 'vgg_Colors': vgg_Colors}

    def __len__(self):
        return len(self.ABC_paths)

    def name(self):
        return 'UnpairedFewFusionDataset'
