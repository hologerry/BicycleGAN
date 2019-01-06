import torch
from .base_model import BaseModel
from . import networks
from .vgg import VGG19


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
        self.loss_names = ['G_L1', 'G_L1_B', 'G_CX', 'G_CX_B', 'G_MSE', 'G_GAN', 'G_GAN_B', 'D', 'D_B',
                           'G_L1_val', 'G_L1_B_val', 'patch_G']
        self.loss_G_L1_val = 0.0
        self.loss_G_L1_B_val = 0.0
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # It is up to the direction AtoB or BtoC or AtoC
        self.dirsection = opt.direction

        # DualNet model only support AtoC now, BtoC and AtoB need to do
        # BicycleGAN model supports all
        assert(self.dirsection == 'AtoC')
        self.visual_names = ['real_A', 'real_B', 'fake_B', 'real_C', 'fake_C']
        # specify the models you want to save to the disk.
        # The program will call base_model.save_networks and base_model.load_networks
        # D for color
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        # D_B for shape
        use_D_B = opt.isTrain and opt.lambda_GAN_B > 0.0
        # use_D_B = False
        use_R = opt.isTrain and opt.lambda_GAN_R > 0.0
        use_R = False
        self.use_R = use_R

        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.nz, opt.ngf, self.opt.nencode, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout, init_type=opt.init_type,
                                      gpu_ids=self.gpu_ids, where_add=self.opt.where_add, upsample=opt.upsample)
        print(self.netG)
        D_output_nc = (opt.input_nc + opt.output_nc) if opt.conditional_D else opt.output_nc
        use_sigmoid = opt.gan_mode == 'dcgan'
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type,
                                          num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)
            print(self.netD)
        if use_D_B:
            self.model_names += ['D_B']
            self.netD_B = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD_B, norm=opt.norm, nl=opt.nl,
                                            use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds,
                                            gpu_ids=self.gpu_ids)

        if use_R:
            self.model_names += ['R']
            self.netR = networks.define_R(D_output_nc, opt.ndf, netR=opt.netR, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type,
                                          gpu_ids=self.gpu_ids)
            print(self.netR)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(mse_loss=not use_sigmoid).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()

            # Contextual Loss
            self.criterionCX = networks.CXLoss(sigma=0.5).to(self.device)
            self.vgg19 = VGG19().to(self.device)
            self.vgg19.load_model(self.opt.vgg)
            self.vgg19.eval()
            self.vgg_layers = ['conv3_2', 'conv4_2']

            # patch based loss
            self.patchLoss = networks.PatchLoss(self.device, self.opt).to(self.device)

            # Discriminative region proposal
            if self.use_R:
                self.proposal = networks.Proposal(opt)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D_B:
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_B)
            if use_R:
                self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_R)

    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input, blk_epoch=False):
        self.real_A = input['A'].to(self.device)  # A is the base font
        self.real_B = input['B'].to(self.device)  # B is the gray shape
        self.real_C = input['C'].to(self.device)  # C is the color font
        self.real_Shapes = input['Shapes'].to(self.device)
        self.real_Colors = input['Colors'].to(self.device)  # Colors is multiple color characters
        self.vgg_Shapes = input['vgg_Shapes'].to(self.device)
        self.vgg_Colors = input['vgg_Colors'].to(self.device)
        # current epoch is black epoch
        if blk_epoch:
            self.real_Colors = self.real_Shapes
            self.real_C = self.real_B

    def test(self):
        with torch.no_grad():
            self.fake_C, self.fake_B = self.netG(self.real_A, self.real_Colors)
            return self.real_A, self.fake_B, self.real_B, self.fake_C, self.real_C

    def validate(self):
        with torch.no_grad():
            self.fake_C, self.fake_B = self.netG(self.real_A, self.real_Colors)
            self.loss_G_L1_val = 0.0
            self.loss_G_L1_B_val = 0.0
            if self.opt.lambda_L1 > 0.0:
                self.loss_G_L1_val = self.criterionL1(self.fake_C, self.real_C) * self.opt.lambda_L1
                self.loss_G_L1_B_val = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1_B
            return self.real_A, self.fake_B, self.real_B, self.fake_C, self.real_C, \
                    self.loss_G_L1_B_val, self.loss_G_L1_val

    def train(self):
        for name in self.model_names:
            model_name = 'net' + name
            getattr(self, model_name).train()

    def forward(self):
        # generate fake_C
        self.fake_C, self.fake_B = self.netG(self.real_A, self.real_Colors)
        # vgg
        self.vgg_fake_C = self.vgg19(self.fake_C)
        self.vgg_real_C = self.vgg19(self.real_C)
        self.vgg_fake_B = self.vgg19(self.fake_B)
        self.vgg_real_B = self.vgg19(self.real_B)

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

    def backward_R(self, netR, netD, real_data, fake_data, real, fake):
        score_map = netD(fake_data.detach())
        # r means region
        if self.opt.mask_operation:
            masked_fake_data, masked_fake, real_data_r, fake_data_r, real_r, fake_r \
                = self.proposal(score_map, real_data, fake_data, real, fake)
        else:
            real_data_r, fake_data_r, real_r, fake_r = self.proposal(score_map, real_data, fake_data, real, fake)

        self.masked_fake = masked_fake
        self.real_r = real_r
        self.fake_r = fake_r
        # print("masked fake data size", masked_fake_data.size())
        pred_fake = netR(masked_fake_data.detach())
        pred_real = netR(real_data)
        # print("Reviser ouput size", pred_fake.size())
        loss_R_fake, _ = self.criterionGAN(pred_fake, False)
        loss_R_real, _ = self.criterionGAN(pred_real, True)
        # Combined loss
        loss_R = loss_R_fake + loss_R_real
        loss_R.backward()
        return loss_R, [loss_R_fake, loss_R_real]

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

    def backward_G_GAN(self, fake, net=None, ll=0.0):
        if ll > 0.0:
            pred_fake = net(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll

    def backward_G(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_C, self.netD, self.opt.lambda_GAN)
        self.loss_G_GAN_B = self.backward_G_GAN(self.fake_data_B, self.netD_B, self.opt.lambda_GAN_B)
        if self.use_R:
            self.loss_G_GAN_R = self.backward_G_GAN(self.fake_data_C, self.netR, self.opt.lambda_GAN_R)

        # 2, reconstruction |fake_C-real_C| |fake_B-real_B|
        self.loss_G_L1 = 0.0
        self.loss_G_L1_B = 0.0
        if self.opt.lambda_L1 > 0.0 or self.opt.lambda_L1_B > 0.0:
            self.loss_G_L1 = self.criterionL1(self.fake_C, self.real_C) * self.opt.lambda_L1
            self.loss_G_L1_B = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1_B

        # 3, contextual loss
        self.loss_G_CX = 0.0
        self.loss_G_CX_B = 0.0
        if self.opt.lambda_CX > 0.0:
            for l in self.vgg_layers:
                # symmetric contextual loss
                self.loss_G_CX += self.criterionCX(self.vgg_real_C[l], self.vgg_fake_C[l]) * self.opt.lambda_CX
                self.loss_G_CX_B += self.criterionCX(self.vgg_real_B[l], self.vgg_fake_B[l]) * self.opt.lambda_CX_B

        # 4, L2 losss
        self.loss_G_MSE = 0.0
        if self.opt.lambda_L2 > 0.0:
            self.loss_G_MSE = self.criterionMSE(self.fake_C, self.real_C) * self.opt.lambda_L2


        # 5. patch loss
        self.loss_patch_G = self.patchLoss(self.fake_C, self.real_B, self.vgg_Shapes, self.vgg_Colors) * self.opt.lambda_patch

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN_B + self.loss_G_L1 + self.loss_G_L1_B \
            + self.loss_G_CX + self.loss_G_CX_B + self.loss_G_MSE \
            + self.loss_patch_G
            
        self.loss_G.backward(retain_graph=True)

    def update_R(self):
        self.set_requires_grad(self.netR, True)
        if self.opt.lambda_GAN_R > 0.0:
            self.optimizer_R.zero_grad()
            self.loss_R, self.losses_R = self.backward_R(self.netR, self.netD, self.real_data_C, self.fake_data_C,
                                                         self.real_C, self.fake_C)
            self.optimizer_R.step()

    def update_D(self):
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netD_B, True)
        # update D
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_C, self.fake_data_C)
            self.optimizer_D.step()

        if self.opt.lambda_GAN_B > 0.0:
            self.optimizer_D_B.zero_grad()
            self.loss_D_B, self.losses_D_B = self.backward_D(self.netD_B, self.real_data_B, self.fake_data_B)
            self.optimizer_D_B.step()

    def update_G(self):
        # update dual net G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netD_B, False)
        if self.use_R:
            self.set_requires_grad(self.netR, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G()
        self.update_D()

        if self.use_R:
            self.forward()
            self.update_R()
            self.update_G()
