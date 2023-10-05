from .layers import ResUnetEncoder, ResUnetDecoder, ResUnetClassifier, ResUnetDecoderJF, ResUnetDecoderJFNoSkip, ResUnetRegressionClassifier
from torch import nn
import torch
from abc import abstractmethod
from einops import rearrange
from ..utils import ModelModule, ModelModuleMultiTask

class GenericModel(ModelModule):
    def __init__(self, params, training_params) -> None:
        super(GenericModel, self).__init__(training_params)
        self.opt_input = len(params['train_opt_imgs'][0]) * params['opt_bands'] + 1
        self.sar_input = len(params['train_sar_imgs'][0]) * params['sar_bands'] + 1

        #self.opt_imgs = len(params['train_opt_imgs'][0])
        #self.sar_imgs = len(params['train_sar_imgs'][0])
        self.n_classes = params['n_classes']
        self.depths = params['resunet_depths']

    def get_opt(self, x):
        return rearrange(x[0], 'b i c h w -> b (i c) h w')
    
    def get_sar(self, x):
        return rearrange(x[1], 'b i c h w -> b (i c) h w')

class GenericResunet(GenericModel):
    def prepare_model(self, in_channels):
        self.encoder = ResUnetEncoder(in_channels, self.depths)
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

    def prepare_input(self, x):
        #x_img = torch.cat(x[0], dim=1)
        x_img = self.get_opt(x)
        x = torch.cat((x_img, x[2]), dim=1)
        return x
    
class ResUnetSAR(GenericResunet):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.sar_input)

    def prepare_input(self, x):
        #x_img = torch.cat(x[1], dim=1)
        x_img = self.get_sar(x)
        x = torch.cat((x_img, x[2]), dim=1)
        return x
    
class ResUnetEF(GenericResunet):    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.opt_input + self.sar_input - 1)

    def prepare_input(self, x):
        x_img_0 = self.get_opt(x)
        x_img_1 = self.get_sar(x)
        x = torch.cat((x_img_0, x_img_1, x[2]), dim=1)
        return x
    
class ResUnetJF(GenericModel):
    def __init__(self, *args, **kargs):
        super(ResUnetJF, self).__init__(*args, **kargs)
        self.encoder_opt = ResUnetEncoder(self.opt_input, self.depths)
        self.encoder_sar = ResUnetEncoder(self.sar_input, self.depths)
        self.decoder = ResUnetDecoderJF(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)


    def forward(self, x):
        x_img_opt = self.get_opt(x)
        x_opt = torch.cat((x_img_opt, x[2]), dim=1)

        x_img_sar = self.get_sar(x)
        x_sar = torch.cat((x_img_sar, x[2]), dim=1)

        x_opt = self.encoder_opt(x_opt)
        x_sar = self.encoder_sar(x_sar)
        x = []
        for i in range(len(x_opt)):
            x_cat = torch.cat((x_opt[i], x_sar[i]), dim=1)
            x.append(x_cat)

        x = self.decoder(x)
        x = self.classifier(x)

        return x
    
class ResUnetJFNoSkip(GenericModel):
    def __init__(self, *args, **kargs):
        super(ResUnetJFNoSkip, self).__init__(*args, **kargs)

        self.encoder_opt = ResUnetEncoder(self.opt_input, self.depths)
        self.encoder_sar = ResUnetEncoder(self.sar_input, self.depths)
        self.decoder = ResUnetDecoderJF(self.depths)
        self.classifier = ResUnetClassifier(self.depths[0], self.n_classes)


    def forward(self, x):
        x_img_opt = self.get_opt(x)
        x_opt = torch.cat((x_img_opt, x[2]), dim=1)

        x_img_sar = self.get_sar(x)
        x_sar = torch.cat((x_img_sar, x[2]), dim=1)

        x_opt = self.encoder_opt(x_opt)
        x_sar = self.encoder_sar(x_sar)
        x = torch.cat((x_opt[-1], x_sar[-1]), dim=1)

        x = self.decoder(x)
        x = self.classifier(x)

        return x
    
class ResUnetLF(GenericModel):
    def __init__(self, *args, **kargs):
        super(ResUnetLF, self).__init__(*args, **kargs)

        self.encoder_opt = ResUnetEncoder(self.opt_input, self.depths)
        self.encoder_sar = ResUnetEncoder(self.sar_input, self.depths)
        self.decoder_opt = ResUnetDecoder(self.depths)
        self.decoder_sar = ResUnetDecoder(self.depths)
        self.classifier = ResUnetClassifier(2*self.depths[0], self.n_classes)

    def forward(self, x):
        x_img_opt = self.get_opt(x)
        x_opt = torch.cat((x_img_opt, x[2]), dim=1)

        x_img_sar = self.get_sar(x)
        x_sar = torch.cat((x_img_sar, x[2]), dim=1)

        x_opt = self.encoder_opt(x_opt)
        x_sar = self.encoder_sar(x_sar)

        x_opt = self.decoder_opt(x_opt)
        x_sar = self.decoder_sar(x_sar)

        x = torch.cat((x_opt, x_sar), dim=1)

        x = self.classifier(x)

        return x

class ResUnetOptMultiTask(ModelModuleMultiTask):
    def __init__(self, params, training_params) -> None:
        super().__init__(training_params)
        self.opt_input = len(params['train_opt_imgs'][0]) * params['opt_bands'] + 1
        self.sar_input = len(params['train_sar_imgs'][0]) * params['sar_bands'] + 1

        #self.opt_imgs = len(params['train_opt_imgs'][0])
        #self.sar_imgs = len(params['train_sar_imgs'][0])
        self.n_classes = params['n_classes']
        self.depths = params['resunet_depths']

        self.encoder = ResUnetEncoder(self.opt_input, self.depths)
        self.decoder_def = ResUnetDecoder(self.depths)
        self.decoder_cloud = ResUnetDecoder(self.depths)
        self.classifier_def = ResUnetClassifier(self.depths[0], self.n_classes)
        self.classifier_cloud = ResUnetRegressionClassifier(self.depths[0])

    def get_opt(self, x):
        return rearrange(x[0], 'b i c h w -> b (i c) h w')
    
    def get_sar(self, x):
        return rearrange(x[1], 'b i c h w -> b (i c) h w')


    def forward(self, x):
        x_img_opt = self.get_opt(x)
        x_opt = torch.cat((x_img_opt, x[2]), dim=1)

        x_opt = self.encoder(x_opt)

        x_def = self.decoder_def(x_opt)
        x_cloud = self.decoder_cloud(x_opt)

        x_def = self.classifier_def(x_def)
        x_cloud = self.classifier_cloud(x_cloud)
        #x_cloud = torch.squeeze(x_cloud, 1)

        return x_def, x_cloud