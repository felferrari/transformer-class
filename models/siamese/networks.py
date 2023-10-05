import segmentation_models_pytorch as smp
import timm
from torch import nn
import torch
from .layers import Decoder, Classifier, SkipConn
from abc import abstractclassmethod

model = timm.create_model('resnet26d', features_only = True, pretrained=False)


class GenericModel(nn.Module):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.opt_input = params['opt_bands']
        self.sar_input = params['sar_bands']

        #self.n_opt_imgs = len(params['train_opt_imgs'][0])
        #self.n_sar_imgs = len(params['train_sar_imgs'][0])

        self.n_classes = params['n_classes']
        self.depths = params['resunet_depths']

    @abstractclassmethod
    def create_model(self):
        pass

class SingleInputModel(GenericModel):
    def create_model(self, in_channels, base_encoder):
        self.encoder = timm.create_model(base_encoder, features_only = True, pretrained=False, in_chans = in_channels)
        self.skip = SkipConn(self.encoder.feature_info.channels())
        self.decoder = Decoder(self.skip.out_channels(), self.encoder.feature_info.channels(), self.encoder.feature_info.reduction()[0])
        self.classifier = Classifier(self.encoder.feature_info.channels()[0], self.n_classes)
        #self.average_pool = nn.AvgPool2d(2, 2)
        self.first_reduction = self.encoder.feature_info.reduction()[0]

    def forward(self, x_imgs, x_prev):
        x_0 = self.encoder(x_imgs[0])
        x_1 = self.encoder(x_imgs[1])
        
        x_diff = [x_1[i] - x_0[i] for i in range(len(x_1))]
        x_cat = [torch.concat([x_1[i], x_0[i]], dim=1) for i in range(len(x_1))]

                
        if self.first_reduction == 1:
            x_prev = [nn.functional.avg_pool2d(x_prev, 2**i, 2**i) for i in range(len(x_diff))]
        else:
            x_prev = [nn.functional.avg_pool2d(x_prev, 2**(i+1), 2**(i+1)) for i in range(len(x_diff))]
        #x = self.skip(x_diff, x_prev)
        x = self.skip(x_cat, x_prev)
        x = self.decoder(x)
        x = self.classifier(x)

        return x


class SiameseOpt(SingleInputModel):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(params, *args, **kwargs)
        self.create_model(self.opt_input, params['base_encoder'])

    def forward(self, x):
        return super().forward(x[0], x[2])
    
class SiameseSAR(SingleInputModel):
    def __init__(self, params, *args, **kwargs) -> None:
        super().__init__(params, *args, **kwargs)
        self.create_model(self.sar_input, params['base_encoder'])

    def forward(self, x):
        return super().forward(x[1], x[2])

