from util import util
import os


# reference to hyper-parameters.md
# use int to denote the value
upsample = ['basic', 'bilinear']
nl = ['relu', 'lrelu', 'elu']
use_attention = [False, True]
init_type = ['normal', 'xavier', 'kaiming', 'orthogonal']

beta1 = [0.5, 0.6, 0.7, 0.8, 0.9]
lr = [0.01, 0.02, 0.001, 0.002, 0.0001, 0.0002]
# lr_policy = ['lambda', 'step', 'plateau']
lr_policy = ['lambda', 'step']  # plateau is not supported due to the step() has to be after validate()
lr_decay_iters = [100, 500, 1000]

lambda_L1 = list(range(1, 21))
lambda_L1_B = list(range(1, 21))
lambda_CX = list(range(1, 21))
lambda_CX_B = list(range(1, 21))
lambda_GAN = list(range(1, 11))
lambda_GAN_B = list(range(1, 11))


def get_range_list():
    range_list = []
    # model related
    # range_list.append(2)  # norm
    range_list.append(2)  # upsample
    range_list.append(3)  # nl
    range_list.append(2)  # use_attention
    range_list.append(4)  # init_type
    # optimizer related
    range_list.append(5)  # beta1
    range_list.append(6)  # lr
    range_list.append(2)  # lr_policy
    range_list.append(3)  # lr_decay_iters
    # lambda parameters
    range_list.append(20)  # l1
    range_list.append(20)  # l1 b
    range_list.append(20)  # cx
    range_list.append(20)  # cx b
    range_list.append(10)  # gan
    range_list.append(10)  # gan b

    return range_list


def convert_hp_to_dict(hps, dim):
    hp_opt = {}
    assert(isinstance(hps, list))
    assert(len(hps) == dim)
    hp_opt['upsample'] = upsample[hps[0]]
    hp_opt['nl'] = nl[hps[1]]
    hp_opt['use_attention'] = use_attention[hps[2]]
    hp_opt['init_type'] = init_type[hps[3]]
    hp_opt['beta1'] = beta1[hps[4]]
    hp_opt['lr'] = lr[hps[5]]
    hp_opt['lr_policy'] = lr_policy[hps[6]]
    hp_opt['lr_decay_iters'] = lr_decay_iters[hps[7]]
    hp_opt['lambda_L1'] = lambda_L1[hps[8]]*5.0
    hp_opt['lambda_L1_B'] = lambda_L1_B[hps[9]]*5.0
    hp_opt['lambda_CX'] = lambda_CX[hps[10]]*5.0
    hp_opt['lambda_CX_B'] = lambda_CX_B[hps[11]]*5.0
    hp_opt['lambda_GAN'] = lambda_GAN[hps[12]]*5.0
    hp_opt['lambda_GAN_B'] = lambda_GAN_B[hps[13]]*5.0

    print(hp_opt)

    return hp_opt


def print_options(opt):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
