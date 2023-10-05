from .layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, ResUnetDecoderJF, ResUnetDecoderJFNoSkip
from torch import nn
import torch
from abc import abstractmethod
from einops import rearrange
from ..utils import ModelModule

class GenericModel(ModelModule):
    def __init__(self, params, training_params) -> None:
        super(GenericModel, self).__init__(training_params)
        self.opt_input = params['opt_bands'] # len(params['train_opt_imgs'][0]) * params['opt_bands'] + 1
        self.sar_input = params['sar_bands'] #len(params['train_sar_imgs'][0]) * params['sar_bands'] + 1

        self.n_opt_imgs = len(params['train_opt_imgs'][0])
        self.n_sar_imgs = len(params['train_sar_imgs'][0])
        self.n_classes = params['n_classes']
        self.depths = params['resunet_depths']

        self.pre_conv_opt = None
        self.pre_conv_sar = None

    def get_opt(self, x):
        #return rearrange(x[0], 'b i c h w -> b (i c) h w')
        x_ret = rearrange(x[0], 'b i c h w -> b c i h w')
        return rearrange(self.pre_conv_opt(x_ret), 'b i c h w -> b (c i) h w')
    
    def get_sar(self, x):
        #return rearrange(x[1], 'b i c h w -> b (i c) h w')
        x_ret = rearrange(x[1], 'b i c h w -> b c i h w')
        return rearrange(self.pre_conv_sar(x_ret), 'b i c h w -> b (c i) h w')


class GenericResunet(GenericModel):
    def prepare_model(self, in_channels):
        self.encoder = ResUnetEncoder(in_channels + 1, self.depths)
        self.decoder = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)

    @abstractmethod
    def prepare_input(self, x):
        pass
    
    def forward(self, x):
        x = self.prepare_input(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x


class ResUnetOpt(GenericResunet):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.opt_input)
        self.pre_conv_opt = nn.Conv3d(self.opt_input, self.opt_input, (self.n_opt_imgs, 3, 3), 1, (0, 1, 1))

    def prepare_input(self, x):
        #x_img = torch.cat(x[0], dim=1)
        x_img = self.get_opt(x)
        x = torch.cat((x_img, x[2]), dim=1)
        return x
    
   
class ResUnetSAR(GenericResunet):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.sar_input)
        self.pre_conv_sar = nn.Conv3d(self.sar_input, self.sar_input, (self.n_sar_imgs, 3, 3), 1, (0, 1, 1))

    def prepare_input(self, x):
        #x_img = torch.cat(x[1], dim=1)
        x_img = self.get_sar(x)
        x = torch.cat((x_img, x[2]), dim=1)
        return x
    
class ResUnetEF(GenericResunet):    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.opt_input + self.sar_input + 1)
        self.pre_conv_opt = nn.Conv3d(self.opt_input, self.opt_input, (self.n_opt_imgs, 3, 3), 1, (0, 1, 1))
        self.pre_conv_sar = nn.Conv3d(self.sar_input, self.sar_input, (self.n_sar_imgs, 3, 3), 1, (0, 1, 1))

    def prepare_input(self, x):
        x_img_0 = self.get_opt(x)
        x_img_1 = self.get_sar(x)
        x = torch.cat((x_img_0, x_img_1, x[2]), dim=1)
        return x
    

class ResUnetJF(GenericModel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.encoder_0 = ResUnetEncoder(self.opt_input+1, self.depths)
        self.encoder_1 = ResUnetEncoder(self.sar_input+1, self.depths)
        self.decoder = ResUnetDecoderJF(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)
        self.pre_conv_opt = nn.Conv3d(self.opt_input, self.opt_input, (self.n_opt_imgs, 3, 3), 1, (0, 1, 1))
        self.pre_conv_sar = nn.Conv3d(self.sar_input, self.sar_input, (self.n_sar_imgs, 3, 3), 1, (0, 1, 1))


    def forward(self, x):
        x_img_0 = self.get_opt(x)
        x_0 = torch.cat((x_img_0, x[2]), dim=1)

        x_img_1 = self.get_sar(x)
        x_1 = torch.cat((x_img_1, x[2]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)
        x = []
        for i in range(len(x_0)):
            x_cat = torch.cat((x_0[i], x_1[i]), dim=1)
            x.append(x_cat)

        x = self.decoder(x)
        x = self.classifier(x)

        return x
    
class ResUnetJFNoSkip(GenericModel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        self.encoder_0 = ResUnetEncoder(self.opt_input+1, self.depths)
        self.encoder_1 = ResUnetEncoder(self.sar_input+1, self.depths)
        self.decoder = ResUnetDecoderJF(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)
        self.pre_conv_opt = nn.Conv3d(self.opt_input, self.opt_input, (self.n_opt_imgs, 3, 3), 1, (0, 1, 1))
        self.pre_conv_sar = nn.Conv3d(self.sar_input, self.sar_input, (self.n_sar_imgs, 3, 3), 1, (0, 1, 1))


    def forward(self, x):
        x_img_0 = self.get_opt(x)
        x_0 = torch.cat((x_img_0, x[2]), dim=1)

        x_img_1 = self.get_sar(x)
        x_1 = torch.cat((x_img_1, x[2]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)
        x = torch.cat((x_0[-1], x_1[-1]), dim=1)

        x = self.decoder(x)
        x = self.classifier(x)

        return x
    
class ResUnetLF(GenericModel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        self.encoder_0 = ResUnetEncoder(self.opt_input+1, self.depths)
        self.encoder_1 = ResUnetEncoder(self.sar_input+1, self.depths)
        self.decoder_0 = ResUnetDecoder(self.depths)
        self.decoder_1 = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(2*self.depths[0], self.n_classes)
        self.pre_conv_opt = nn.Conv3d(self.opt_input, self.opt_input, (self.n_opt_imgs, 3, 3), 1, (0, 1, 1))
        self.pre_conv_sar = nn.Conv3d(self.sar_input, self.sar_input, (self.n_sar_imgs, 3, 3), 1, (0, 1, 1))

    def forward(self, x):
        x_img_0 = self.get_opt(x)
        x_0 = torch.cat((x_img_0, x[2]), dim=1)

        x_img_1 = self.get_sar(x)
        x_1 = torch.cat((x_img_1, x[2]), dim=1)

        x_0 = self.encoder_0(x_0)
        x_1 = self.encoder_1(x_1)

        x_0 = self.decoder_0(x_0)
        x_1 = self.decoder_1(x_1)

        x = torch.cat((x_0, x_1), dim=1)

        x = self.classifier(x)

        return x

