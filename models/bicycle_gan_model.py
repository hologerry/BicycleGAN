import torch
from .base_model import BaseModel
from . import networks


class BiCycleGANModel(BaseModel):
    def name(self):
        return 'BiCycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        if opt.isTrain:
            assert opt.batch_size % 2 == 0  # load two images at one time.

        BaseModel.initialize(self, opt)
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'G_L2', 'z_L1', 'kl']

        # get the direction AtoB or BtoC or AtoC
        self.direction = opt.direction

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        # It is up to the direction AtoB or BtoC or AtoC
        if self.opt.dataset_mode == 'multi_fusion':
            if self.direction == 'AtoC' or self.direction == 'BtoC':
                self.visual_names = ['real_A', 'real_B', 'real_C', 'fake_C_encoded']
            else:
                self.visual_names = ['real_A', 'real_B', 'real_C', 'fake_B_encoded']
        else:
            self.visual_names = ['real_A', 'real_B', 'fake_B_random', 'fake_B_encoded']

        # specify the models you want to save to the disk.
        # The program will call base_model.save_networks and base_model.load_networks
        use_D = opt.isTrain and opt.lambda_GAN > 0.0
        use_D2 = opt.isTrain and opt.lambda_GAN2 > 0.0 and not opt.use_same_D
        # Encoder is used for other datasets or ABC shapes encode
        use_E = opt.isTrain or not opt.no_encode
        # Encoder2 is used for ABC colors encode
        # use_E2 = opt.dataset_mode == 'multi_fusion' and (opt.direction == 'AtoC' or opt.direction == 'BtoC')
        use_E2 = False
        use_vae = True

        use_attention = opt.use_attention
        use_spectral_norm_G = opt.use_spectral_norm_G
        use_spectral_norm_D = opt.use_spectral_norm_D

        self.nzG = opt.nz*2 if use_E2 else opt.nz
        self.model_names = ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, self.nzG, opt.ngf, netG=opt.netG,
                                      norm=opt.norm, nl=opt.nl, use_dropout=opt.use_dropout,
                                      use_attention=use_attention, use_spectral_norm=use_spectral_norm_G,
                                      init_type=opt.init_type, gpu_ids=self.gpu_ids,
                                      where_add=self.opt.where_add, upsample=opt.upsample)

        D_output_nc = opt.input_nc + opt.output_nc if opt.conditional_D else opt.output_nc
        use_sigmoid = opt.gan_mode == 'dcgan'
        if use_D:
            self.model_names += ['D']
            self.netD = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD, norm=opt.norm, nl=opt.nl,
                                          use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds,
                                          use_spectral_norm=use_spectral_norm_D, gpu_ids=self.gpu_ids)
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = networks.define_D(D_output_nc, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                           use_sigmoid=use_sigmoid, init_type=opt.init_type, num_Ds=opt.num_Ds,
                                           use_spectral_norm=use_spectral_norm_D, gpu_ids=self.gpu_ids)

        if use_E:
            self.model_names += ['E']
            self.netE = networks.define_E(opt.input_nc*opt.nencode, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm,
                                          nl=opt.nl, init_type=opt.init_type, gpu_ids=self.gpu_ids, vaeLike=use_vae)
        if use_E2:
            self.model_names += ['E2']
            self.netE2 = networks.define_E(opt.input_nc*opt.nencode, opt.nz, opt.nef, netE=opt.netE, norm=opt.norm,
                                           nl=opt.nl, init_type=opt.init_type, gpu_ids=self.gpu_ids, vaeLike=use_vae)

        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                mse_loss=not use_sigmoid).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionZ = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(
                    self.netE.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E)
            if use_E2:
                self.optimizer_E2 = torch.optim.Adam(
                    self.netE2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_E2)

            if use_D:
                self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(
                    self.netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D2)

    def is_train(self):
        return self.opt.isTrain and self.real_A.size(0) == self.opt.batch_size

    def set_input(self, input):
        if self.opt.dataset_mode == 'multi_fusion':
            self.real_A = input['A'].to(self.device)
            self.real_B = input['B'].to(self.device)  # B is the gray shape
            self.real_C = input['C'].to(self.device)
            self.real_Shapes = input['Shapes'].to(self.device)
            self.real_Colors = input['Colors'].to(self.device)
            self.image_paths = input['ABC_path']
        else:
            # other datasets
            AtoB = self.opt.direction == 'AtoB'
            self.real_A = input['A' if AtoB else 'B'].to(self.device)
            self.real_B = input['B' if AtoB else 'A'].to(self.device)
            self.image_paths = input['A_path' if AtoB else 'B_path']

    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.to(self.device)

    def encode(self, input_image, is_E2=False):
        if is_E2:
            mu, logvar = self.netE2.forward(input_image)
        else:
            mu, logvar = self.netE.forward(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar

    def test(self, z0=None, encode=False, use_E2=False):
        with torch.no_grad():
            if self.opt.dataset_mode == 'multi_fusion':
                if encode:  # use encoded z
                    zshape, _ = self.netE(self.real_Shapes)
                    zcolor, _ = self.netE2(self.real_Colors)
                    z0 = torch.cat([zshape, zcolor], 1)
                if z0 is None:
                    z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
                self.fake_out = self.netG(self.real_A, z0)
                return self.real_A, self.fake, self.real
            else:
                if encode:  # use encoded z
                    z0, _ = self.netE(self.real_B)
                if z0 is None:
                    z0 = self.get_z_random(self.real_A.size(0), self.opt.nz)
                self.fake_B = self.netG(self.real_A, z0)
                return self.real_A, self.fake_B, self.real_B

    def forward(self):
        # compute encoded or random B on whole batch
        if self.opt.dataset_mode == 'multi_fusion':
            if self.opt.direction == 'AtoC':
                z_encoded_shape, mu_shape, logvar_shape = self.encode(self.real_Shapes)
                z_encoded_color, mu_color, logvar_color = self.encode(self.real_Colors, is_E2=True)
                self.z_encoded = torch.cat([z_encoded_shape, z_encoded_color], 1)
                self.mu = torch.cat([mu_shape, mu_color], 1)
                self.logvar = torch.cat([logvar_shape, logvar_color], 1)
                # get random z
                self.z_random = self.get_z_random(self.real_A.size(0), self.nzG)
                # generate fake_C_encoded
                self.fake_C_encoded = self.netG(self.real_A, self.z_encoded)
                # generate fake_B_random
                self.fake_C_random = self.netG(self.real_A, self.z_random)

            elif self.opt.direction == 'AtoB':
                self.z_encoded, self.mu, self.logvar = self.encode(self.real_Shapes)
                # get random z
                self.z_random = self.get_z_random(self.real_A.size(0), self.nzG)
                # generate fake_C_encoded
                self.fake_B_encoded = self.netG(self.real_A, self.z_encoded)
                # generate fake_B_random
                self.fake_B_random = self.netG(self.real_A, self.z_random)
            elif self.opt.direction == 'BtoC':
                self.z_encoded, self.mu, self.logvar = self.encode(self.real_Colors, is_E2=True)
                # get random z
                self.z_random = self.get_z_random(self.real_B.size(0), self.nzG)
                # generate fake_C_encoded
                self.fake_C_encoded = self.netG(self.real_B, self.z_encoded)
                # generate fake_B_random
                self.fake_C_random = self.netG(self.real_B, self.z_random)
        else:
            self.z_encoded, self.mu, self.logvar = self.encode(self.real_B)
            # get random z
            self.z_random = self.get_z_random(self.real_B.size(0), self.nzG)
            # generate fake_C_encoded
            self.fake_B_encoded = self.netG(self.real_A, self.z_encoded)
            # generate fake_B_random
            self.fake_B_random = self.netG(self.real_A, self.z_random)

        if self.opt.conditional_D:   # tedious conditoinal data
            if self.opt.dataset_mode == 'multi_fusion':
                if self.opt.direction == 'AtoC':
                    self.fake_data_encoded = torch.cat(
                        [self.real_A, self.fake_C_encoded], 1)
                    self.real_data_encoded = torch.cat(
                        [self.real_A, self.real_C_encoded], 1)
                    self.fake_data_random = torch.cat(
                        [self.real_A, self.fake_C_random], 1)
                    self.real_data_random = torch.cat(
                        [self.real_A, self.real_C_random], 1)
                elif self.opt.direction == 'AtoB':
                    self.fake_data_encoded = torch.cat(
                        [self.real_A, self.fake_B_encoded], 1)
                    self.real_data_encoded = torch.cat(
                        [self.real_A, self.real_B_encoded], 1)
                    self.fake_data_random = torch.cat(
                        [self.real_A, self.fake_B_random], 1)
                    self.real_data_random = torch.cat(
                        [self.real_B, self.real_B_random], 1)
                elif self.opt.direction == 'BtoC':
                    self.fake_data_encoded = torch.cat(
                        [self.real_B, self.fake_C_encoded], 1)
                    self.real_data_encoded = torch.cat(
                        [self.real_B, self.real_C_encoded], 1)
                    self.fake_data_random = torch.cat(
                        [self.real_B, self.fake_C_random], 1)
                    self.real_data_random = torch.cat(
                        [self.real_B, self.real_C_random], 1)
            else:
                self.fake_data_encoded = torch.cat(
                    [self.real_A, self.fake_B_encoded], 1)
                self.real_data_encoded = torch.cat(
                    [self.real_A, self.real_B_encoded], 1)
                self.fake_data_random = torch.cat(
                    [self.real_A, self.fake_B_random], 1)
                self.real_data_random = torch.cat(
                    [self.real_A, self.real_B_random], 1)
        else:
            if self.opt.dataset_mode == 'multi_fusion':
                if self.opt.direction == 'AtoC' or self.opt.direction == 'BtoC':
                    self.fake_data_encoded = self.fake_C_encoded
                    self.fake_data_random = self.fake_C_random
                    self.real_data_encoded = self.real_C
                    self.real_data_random = self.real_C
                elif self.opt.direction == 'AtoB':
                    self.fake_data_encoded = self.fake_B_encoded
                    self.fake_data_random = self.fake_B_random
                    self.real_data_encoded = self.real_B
                    self.real_data_random = self.real_B
            else:
                self.fake_data_encoded = self.fake_B_encoded
                self.fake_data_random = self.fake_B_random
                self.real_data_encoded = self.real_B
                self.real_data_random = self.real_B

        # compute z_predict
        if self.opt.lambda_z > 0.0:
            if self.opt.dataset_mode == 'multi_fusion':
                if self.opt.direction == 'AtoC':
                    mu2_shape, logvar2_shape = self.netE(
                        self.fake_C_random.repeat(1, self.opt.nencode, 1, 1))  # mu2 is a point estimate
                    mu2_color, logvar2_color = self.netE2(
                        self.fake_C_random.repeat(1, self.opt.nencode, 1, 1))  # mu2 is a point estimate
                    self.mu2 = torch.cat([mu2_shape, mu2_color], 1)
                    self.logvar2 = torch.cat([logvar_shape, logvar2_color], 1)
                elif self.opt.direction == 'AtoB':
                    self.mu2, logvar2 = self.netE(
                        self.fake_B_random.repeat(1, self.opt.nencode, 1, 1))  # mu2 is a point estimate
                elif self.opt.direction == 'BtoC':
                    self.mu2, logvar2 = self.netE2(
                        self.fake_C_random.repeat(1, self.opt.nencode, 1, 1))  # mu2 is a point estimate
            else:
                self.mu2, logvar2 = self.netE(
                    self.fake_B_random.repeat(1, self.opt.nencode, 1, 1))  # mu2 is a point estimate

    def backward_D(self, netD, real, fake):
        # Fake, stop backprop to the generator by detaching fake_out
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

    def backward_EG(self):
        # 1. G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(
            self.fake_data_encoded, self.netD, self.opt.lambda_GAN)
        if self.opt.use_same_D:
            self.loss_G_GAN2 = self.backward_G_GAN(
                self.fake_data_random, self.netD, self.opt.lambda_GAN2)
        else:
            self.loss_G_GAN2 = self.backward_G_GAN(
                self.fake_data_random, self.netD2, self.opt.lambda_GAN2)
        # 2. KL loss
        if self.opt.lambda_kl > 0.0:
            kl_element = self.mu.pow(2).add_(
                self.logvar.exp()).mul_(-1).add_(1).add_(self.logvar)
            self.loss_kl = torch.sum(
                kl_element).mul_(-0.5) * self.opt.lambda_kl
        else:
            self.loss_kl = 0
        # 3. reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            if self.opt.dataset_mode == 'multi_fusion':
                if self.opt.direction == 'AtoC' or self.opt.direction == 'BtoC':
                    self.loss_G_L1 = self.criterionL1(
                        self.fake_C_encoded, self.real_C) * self.opt.lambda_L1
                elif self.opt.direction == 'AtoB':
                    self.loss_G_L1 = self.criterionL1(
                        self.fake_B_encoded, self.real_B) * self.opt.lambda_L1
            else:
                self.loss_G_L1 = self.criterionL1(
                    self.fake_B_encoded, self.real_B) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = 0.0

        if self.opt.lambda_L2 > 0.0:
            if self.opt.dataset_mode == 'multi_fusion':
                if self.opt.direction == 'AtoC' or self.opt.direction == 'BtoC':
                    self.loss_G_L2 = self.criterionL2(
                        self.fake_C_encoded, self.real_C) * self.opt.lambda_L1
                elif self.opt.direction == 'AtoB':
                    self.loss_G_L2 = self.criterionL2(
                        self.fake_B_encoded, self.real_B) * self.opt.lambda_L1
            else:
                self.loss_G_L2 = self.criterionL2(
                    self.fake_B_encoded, self.real_B) * self.opt.lambda_L1
        else:
            self.loss_G_L2 = 0.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_G_L2 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad(self.netD, True)
        # update D1
        if self.opt.lambda_GAN > 0.0:
            self.optimizer_D.zero_grad()
            self.loss_D, self.losses_D = self.backward_D(
                self.netD, self.real_data_encoded, self.fake_data_encoded)
            if self.opt.use_same_D:
                self.loss_D2, self.losses_D2 = self.backward_D(
                    self.netD, self.real_data_random, self.fake_data_random)
            self.optimizer_D.step()

        if self.opt.lambda_GAN2 > 0.0 and not self.opt.use_same_D:
            self.optimizer_D2.zero_grad()
            self.loss_D2, self.losses_D2 = self.backward_D(
                self.netD2, self.real_data_random, self.fake_data_random)
            self.optimizer_D2.step()

    def backward_G_alone(self):
        # 3. reconstruction |(E(G(A, z_random)))-z_random|
        if self.opt.lambda_z > 0.0:
            self.loss_z_L1 = torch.mean(
                torch.abs(self.mu2 - self.z_random)) * self.opt.lambda_z
            self.loss_z_L1.backward()
        else:
            self.loss_z_L1 = 0.0

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad(self.netD, False)
        self.optimizer_E.zero_grad()
        self.optimizer_E2.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()
        self.optimizer_G.step()
        self.optimizer_E.step()
        self.optimizer_E2.step()
        # update G only
        if self.opt.lambda_z > 0.0:
            self.optimizer_G.zero_grad()
            self.optimizer_E.zero_grad()
            self.optimizer_E2.zero_grad()
            self.backward_G_alone()
            self.optimizer_G.step()

    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()
