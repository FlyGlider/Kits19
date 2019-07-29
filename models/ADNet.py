import sys
sys.path.append('../')
from models.resnet import *
from models.attention_resnet import *
from models.base_model import BaseModel
from models import networks
from utils import ImagePool
import matplotlib.pyplot as plt
from skimage.io import imsave
from torchgan.losses import WassersteinDiscriminatorLoss, WassersteinGeneratorLoss, WassersteinGradientPenalty
# from torchgan.layers import 

class ADNet(BaseModel):
    def name(self):
        return 'ADNet'
    def initialize(self, args, mode=2):
        BaseModel.initialize(self, args)
        self.isTrain = True
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['g_gan', 'g_ce', 'd']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
    
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        if mode == 2:
            self.netG =  AttentionResNet2D(BasicBlock_2d, [1, 1, 1, 1], args.nclass, True).to(self.device)
        else:
            self.netG = ResNet(BasicBlock, [1, 1, 1], args.nclass, True).to(self.device)

        if self.isTrain:
            use_sigmoid = False
            if mode == 2:
                self.netD = networks.define_D(2, 8, 'image', use_sigmoid=use_sigmoid, init_type='xavier', gpu_ids=self.gpu_ids, device=self.device)


        if self.isTrain:
            # define loss functions
            self.criterion_g = WassersteinGeneratorLoss(reduction='mean')
            self.criterion_d = WassersteinDiscriminatorLoss()
            self.criterion_d_penalty = WassersteinGradientPenalty()
            # self.criterion_gan = networks.GANLoss(use_lsgan=False).to(self.device)
            self.criterion_ce = torch.nn.NLLLoss().to(self.device)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=args.lr, betas=(0.9, 0.999))
            self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(),
                                                lr=args.lr)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, img, seg):
        self.img = img.to(self.device)
        self.seg = seg.to(self.device)
    
    def forward(self):
        self.fake_seg = self.netG(self.img)
        self.fake_exp = torch.exp(self.fake_seg)
    
    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        seg = self.seg
        fake_exp = self.fake_exp
        pred_fake = self.netD(fake_exp.detach())
        # self.loss_D_fake = self.criterion_gan(pred_fake, False)

        # Real
        one_hot = torch.zeros(fake_exp.size()).to(self.device).scatter_(1, self.seg.unsqueeze(1), 1)
        pred_real = self.netD(one_hot)
        # self.loss_D_real = self.criterion_gan(pred_real, True)

        # Combined loss
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_d = self.criterion_d(pred_real, pred_fake)
        eps = torch.rand(1).item()
        interpolate = eps * one_hot + (1 - eps) * fake_exp
        d_interpolate = self.netD(interpolate)
        loss_d_penalty = self.criterion_d_penalty(interpolate, d_interpolate)
        loss_d_weighted_loss = self.criterion_d_penalty.lambd * loss_d_penalty
        loss_d = loss_d_weighted_loss + self.loss_d * 0.002
        
        loss_d.backward(retain_graph=True)
    
    def backward_G(self):
        # First, G(A) should fake the discriminator
        pred_fake = self.netD(self.fake_exp)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_g_ce = self.criterion_ce(self.fake_seg, self.seg)
        self.loss_g_gan = self.criterion_g(pred_fake)
        self.loss_g = self.loss_g_gan * 0.002 + self.loss_g_ce * 1

        self.loss_g.backward()
    
    def optimize_parameters(self, epoch):
        self.forward()
        
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


