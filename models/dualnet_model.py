import torch
from .base_model import BaseModel
from . import networks


class DualNetModel(BaseModel):
    def name(self):
        return 'DualNetModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_GAN2', 'D', 'D2', 'G_L1', 'G_L1_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # It is up to the direction AtoB or BtoC or AtoC
        self.dirsection = opt.direction

        # DualNet model only support AtoC now, BtoC and AtoB need to do
        # BicycleGAN model s            upports all
        assert(self.dirsection == 'AtoC')
        self.visual_names = ['real_A', 'real_B', 'real_C', 'fake_C']
        # specify the models you want to save to the disk.
        # The program will call base_model.save_networks and base_model.load_networks
        # D for color
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        # D2 for shape
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        # use_D2 = False
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, self.opt.nencode, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, where_add=self.opt.where_add, upsample=opt.upsample)
        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        use_sigmoid = opt.gan_mode == 'dcgan'
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type,
                                          num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds,
                                           gpu_ids=self.gpu_ids)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(mse_loss=not use_sigmoid).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(
                    self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)

    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        self.real_A = input['A'].to(self.device)  # A is the base font
        self.real_B = input['B'].to(self.device)  # B is the gray shape
        self.real_C = input['C'].to(self.device)  # C is the color font
        self.real_Shapes = input['Shapes'].to(self.device)
        self.real_Colors = input['Colors'].to(self.device)  # Colors is multiple color characters

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def encode(self, input_image):
        mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def test(self, encode=True):
        # for dualnet always use encode
        with torch.no_grad():
            self.fake_C = self.netG(self.real_A, self.real_Colors)
            return self.real_A, self.fake_C, self.real_C

    def forward(self):
        # generate fake_C
        self.fake_C = self.netG(self.real_A, self.real_Colors)
        # fake_C rgb  ->  fake_B gray
        self.fake_B_one = self.fake_C[:, 0, ...] * 0.299 \
            + self.fake_C[:, 1, ...] * 0.587 + self.fake_C[:, 2, ...] * 0.114
        self.fake_B_one.unsqueeze_(1)
        self.fake_B = self.fake_B_one.repeat(1, 3, 1, 1)
        self.real_B_one = self.real_B[:, 0, ...] * 0.299 \
            + self.real_B[:, 1, ...] * 0.587 + self.real_B[:, 2, ...] * 0.114
        if self.opt.conditional_D:   # tedious conditoinal data
            self.fake_data_B = torch.cat([self.real_A, self.fake_B], 1)
            self.real_data_B = torch.cat([self.real_A, self.real_B], 1)
            self.fake_data_C = torch.cat([self.real_A, self.fake_C], 1)
            self.real_data_C = torch.cat([self.real_A, self.real_C], 1)
        else:
            self.fake_data_B = self.fake_B
            self.real_data_B = self.real_B
            self.fake_data_C = self.fake_C
            self.real_data_C = self.real_C

    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_B
        pred_fake = netD(fake.detach())
        # real
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]

    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_G(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_C, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(
                self.fake_data_B, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(
                self.fake_data_B, self.netD2, self.opt.lambda_GAN2)

        # 2, reconstruction |fake_C-real_C| |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_C, self.real_C) * self.opt.lambda_L1
            self.loss_G_L1_B = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1_B
        else:
            self.loss_G_L1 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_G_L1_B
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad(self.netD, True)
        # update D
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_C, self.fake_data_C)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(self.netD2, self.real_data_B, self.fake_data_B)
            self.optimizer_D2.step()

    def update_G(self):
        # update dual net G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # update G second time
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G()
        self.update_D()
