import os.path
import random

from PIL import Image

from data.base_dataset import BaseDataset, transform_fusion
from data.image_folder import make_dataset


class FewFusionDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)
        self.ABC_paths = sorted(make_dataset(self.dir_ABC))
        assert(self.opt.nencode == 4)
        self.few_alphas = ['0', '1', '2', '3', '4']

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
            if self.opt.phase == 'train':
                ABC_path_list = list(ABC_path)
                target_char = ABC_path_list[-5]
                # for shapes
                random.shuffle(self.few_alphas)
                chars_random = [x for x in self.few_alphas if x != target_char]
                for char in chars_random:
                    ABC_path_list[-5] = char  # /path/to/img/XXXX_X_X.png
                    s_path = "".join(ABC_path_list)
                    Shape_paths.append(s_path)
                    Shapes.append(Image.open(s_path).convert('RGB').crop((w, 0, w+w, h)))
                # for colors
                random.shuffle(self.few_alphas)
                chars_random = [x for x in self.few_alphas if x != target_char]
                for char in chars_random:
                    ABC_path_list[-5] = char  # /path/to/img/XXXX_X_X.png
                    c_path = "".join(ABC_path_list)
                    Color_paths.append(c_path)
                    Colors.append(Image.open(c_path).convert('RGB').crop((w+w, 0, w+w+w, h)))
            else:
                ABC_path_train = ABC_path.replace(self.opt.phase, 'train')
                ABC_path_list = list(ABC_path_train)
                target_char = ABC_path_list[-5]
                # for shapes
                random.shuffle(self.few_alphas)
                chars_random = self.few_alphas[:self.opt.nencode]
                for char in chars_random:
                    ABC_path_list[-5] = char  # /path/to/img/XXXX_X_X.png
                    s_path = "".join(ABC_path_list)
                    Shape_paths.append(s_path)
                    Shapes.append(Image.open(s_path).convert('RGB').crop((w, 0, w+w, h)))
                # for colors
                random.shuffle(self.few_alphas)
                chars_random = self.few_alphas[:self.opt.nencode]
                for char in chars_random:
                    ABC_path_list[-5] = char  # /path/to/img/XXXX_X_X.png
                    c_path = "".join(ABC_path_list)
                    Color_paths.append(c_path)
                    Colors.append(Image.open(c_path).convert('RGB').crop((w+w, 0, w+w+w, h)))

        else:
            Shapes.append(B)
            Shape_paths.append(ABC_path)
            Colors.append(C)
            Color_paths.append(ABC_path)
        A, B, C, Shapes, Colors = transform_fusion(self.opt, A, B, C, Shapes, Colors)

        # A is the reference, B is the gray shape, C is the gradient
        return {'A': A, 'B': B, 'C': C, 'Shapes': Shapes, 'Colors': Colors,
                'ABC_path': ABC_path, 'Shape_paths': Shape_paths, 'Color_paths': Color_paths}

    def __len__(self):
        return len(self.ABC_paths)

    def name(self):
        return 'FewFusionDataset'
