from torch import nn
import torch
from .layers import SwinEncoder, SwinDecoder, SwinClassifier, SwinDecoderJF, SwinRegressionClassifier
from abc import abstractmethod
from einops import rearrange
from ..utils import ModelModule, ModelModuleMultiTask
#from conf import general

class GenericModel(ModelModule):
    def __init__(self, params, training_params) -> None:
        super(GenericModel, self).__init__(training_params)
        self.opt_input = len(params['train_opt_imgs'][0]) * params['opt_bands'] + 1
        self.sar_input = len(params['train_sar_imgs'][0]) * params['sar_bands'] + 1

        #self.opt_imgs = len(params['train_opt_imgs'][0])
        #self.sar_imgs = len(params['train_sar_imgs'][0])
        self.n_classes = params['n_classes']

        self.img_size = params['swin_params']['img_size']
        self.base_dim = params['swin_params']['base_dim']
        self.window_size = params['swin_params']['window_size']
        self.shift_size = params['swin_params']['shift_size']
        self.patch_size = params['swin_params']['patch_size']
        self.n_heads = params['swin_params']['n_heads']
        self.n_blocks = params['swin_params']['n_blocks']

    def get_opt(self, x):
        return rearrange(x[0], 'b i c h w -> b (i c) h w')
    
    def get_sar(self, x):
        return rearrange(x[1], 'b i c h w -> b (i c) h w')

class GenericSwinUnet(GenericModel):
    def prepare_model(self, in_channels):
        self.encoder = SwinEncoder(
            input_depth = in_channels, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
    
        self.decoder = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        
        self.classifier = SwinClassifier(
            self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes = self.n_classes)

    @abstractmethod
    def prepare_input(self, x):
        pass
    
    def forward(self, x):
        x = self.prepare_input(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.classifier(x)
        return x

class SwinUnetOpt(GenericSwinUnet):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.opt_input)

    def prepare_input(self, x):
        #x_img = torch.cat(x[0], dim=1)
        x_img = self.get_opt(x)
        x = torch.cat((x_img, x[2]), dim=1)
        return x

class SwinUnetSAR(GenericSwinUnet):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.sar_input)

    def prepare_input(self, x):
        #x_img = torch.cat(x[1], dim=1)
        x_img = self.get_sar(x)
        x = torch.cat((x_img, x[2]), dim=1)
        return x
    
class SwinUnetEF(GenericSwinUnet):    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.prepare_model(self.opt_input + self.sar_input - 1)

    def prepare_input(self, x):
        x_img_0 = self.get_opt(x)
        x_img_1 = self.get_sar(x)
        x = torch.cat((x_img_0, x_img_1, x[2]), dim=1)
        return x
    
class SwinUnetJF(GenericModel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        self.encoder_opt = SwinEncoder(
            input_depth = self.opt_input, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )

        self.encoder_sar = SwinEncoder(
            input_depth = self.sar_input, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
    
        self.decoder = SwinDecoderJF(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        
        self.classifier = SwinClassifier(
            self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes = self.n_classes)


    def forward(self, x):
        x_img_opt = self.get_opt(x)
        x_opt = torch.cat((x_img_opt, x[2]), dim=1)

        x_img_sar = self.get_sar(x)
        x_sar = torch.cat((x_img_sar, x[2]), dim=1)

        x_opt = self.encoder_opt(x_opt)
        x_sar = self.encoder_sar(x_sar)

        x = self.decoder([x_opt, x_sar])

        x = self.classifier(x)

        return x

class SwinUnetLF(GenericModel):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        self.encoder_opt = SwinEncoder(
            input_depth = self.opt_input, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )

        self.encoder_sar = SwinEncoder(
            input_depth = self.sar_input, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
    
        self.decoder_opt = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )

        self.decoder_sar = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        
        self.classifier = SwinClassifier(
            2*self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes = self.n_classes)


    def forward(self, x):
        x_img_opt = self.get_opt(x)
        x_opt = torch.cat((x_img_opt, x[2]), dim=1)

        x_img_sar = self.get_sar(x)
        x_sar = torch.cat((x_img_sar, x[2]), dim=1)

        x_opt = self.encoder_opt(x_opt)
        x_sar = self.encoder_sar(x_sar)

        x_opt = self.decoder_opt(x_opt)
        x_sar = self.decoder_sar(x_sar)

        x = torch.cat((x_opt, x_sar), dim=-1)
        x = self.classifier(x)

        return x

class SwinUnetOptMultiTask(ModelModuleMultiTask):
    def __init__(self,  params, training_params):
        super().__init__(training_params)
        self.opt_input = len(params['train_opt_imgs'][0]) * params['opt_bands'] + 1
        self.sar_input = len(params['train_sar_imgs'][0]) * params['sar_bands'] + 1

        self.n_classes = params['n_classes']

        self.img_size = params['swin_params']['img_size']
        self.base_dim = params['swin_params']['base_dim']
        self.window_size = params['swin_params']['window_size']
        self.shift_size = params['swin_params']['shift_size']
        self.patch_size = params['swin_params']['patch_size']
        self.n_heads = params['swin_params']['n_heads']
        self.n_blocks = params['swin_params']['n_blocks']

        self.encoder = SwinEncoder(
            input_depth = self.opt_input, 
            base_dim = self.base_dim, 
            window_size = self.window_size,
            shift_size = self.shift_size,
            img_size = self.img_size,
            patch_size = self.patch_size,
            n_heads = self.n_heads,
            n_blocks = self.n_blocks
            )
    
        self.decoder_def = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        
        self.classifier_def = SwinClassifier(
            self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size,
            n_classes = self.n_classes)
        
        self.decoder_cloud = SwinDecoder(
            base_dim=self.base_dim,
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size
            )
        
        self.classifier_cloud = SwinRegressionClassifier(
            self.base_dim, 
            n_heads=self.n_heads,
            n_blocks = self.n_blocks,
            window_size = self.window_size,
            shift_size = self.shift_size)

    def get_opt(self, x):
        return rearrange(x[0], 'b i c h w -> b (i c) h w')
    
    def get_sar(self, x):
        return rearrange(x[1], 'b i c h w -> b (i c) h w')
    
    def forward(self, x):
        x_img_opt = self.get_opt(x)
        x_opt = torch.cat((x_img_opt, x[2]), dim=1)

        x = self.encoder(x_opt)

        x_def = self.decoder_def(x)
        x_cloud = self.decoder_cloud(x)

        x_def = self.classifier_def(x_def)
        x_cloud = self.classifier_cloud(x_cloud)

        return x_def, x_cloud