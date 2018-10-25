import os.path
import random

from PIL import Image

from data.base_dataset import BaseDataset, transform_pair
from data.image_folder import make_dataset


class MultiAlignedDataset(BaseDataset):
    # Deprecated !!!
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_ABC = os.path.join(opt.dataroot, opt.phase)
        self.ABC_paths = sorted(make_dataset(self.dir_AB))
        self.alphabets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def __getitem__(self, index):
        ABC_path = self.ABC_paths[index]
        ABC = Image.open(ABC_path).convert('RGB')
        w3, h = ABC.size
        w = int(w3 / 3)
        A = ABC.crop((0, 0, w, h))
        B = ABC.crop((w+w, 0, w+w+w, h))
        C = ABC.crop((w, 0, w + w, h))

        D = []
        D_paths = []
        if self.opt.nencode > 1:
            ABC_path_list = list(ABC_path)
            # for colors
            random.shuffle(self.alphabets)
            chars_random = self.alphabets[:self.opt.nencode]
            for char in chars_random:
                ABC_path_list[-5] = char  # /path/to/img/XXXX_X_X.png
                d_path = "".join(ABC_path_list)
                D_paths.append(d_path)
                D.append(Image.open(d_path).convert('RGB').crop((w+w, 0, w+w+w, h)))
        else:
            D.append(C)
            D_paths.append(ABC_path)

        A, B, C, D = transform_fusion(self.opt, A, B, C, D)

        # A is the reference, B is the gray shape, C is the gradient
        return {'A': A, 'B': B, 'C': C, 'D': D,
                'ABC_path': ABC_path,  'D_paths': D_paths}

    def __len__(self):
        return len(self.ABC_paths)

    def name(self):
        return 'MultiAlignedDataset'
