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
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.alphabets = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))
        C = []
        C_paths = []
        if self.opt.nencode > 1:
            AB_path_list = list(AB_path)
            random.shuffle(self.alphabets)
            chars_random = self.alphabets[:self.opt.nencode]
            for char in chars_random:
                AB_path_list[-8] = char  # /path/to/img/XXX_X_XX.jpg
                c_path = "".join(AB_path_list)
                C_paths.append(c_path)
                C.append(Image.open(c_path).convert('RGB').crop((w2, 0, w, h)))
        else:
            C.append(B)
            C_paths.append(AB_path)
        A, B, C = transform_pair(self.opt, A, B, C)

        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'C': C,
                'A_path': AB_path, 'B_path': AB_path, 'C_paths': C_paths}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'MultiAlignedDataset'
