from torch import nn
import torch

from models.resunet.layers import ResidualBlock

class Decoder(nn.Module):
    def __init__(self, chans_in, chans_out, last_upsample, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2)
            for i in range(len(chans_out)-1)
        ])

        self.convs = nn.ModuleList([
            nn.Conv2d(chans_out[i+1] + chans_in[i], chans_out[i], 1)
            for i in range(len(chans_out)-2)
        ])
        self.convs.append(nn.Conv2d(chans_in[len(chans_in)-1] + chans_in[len(chans_in)-2], chans_out[len(chans_out)-2], 1))

        #self.convs = nn.ModuleList([
        #    ResidualBlock(chans_out[i+1] + chans_in[i], chans_out[i])
        #    for i in range(len(chans_out)-2)
        #])
        #self.convs.append(ResidualBlock(chans_in[len(chans_in)-1] + chans_in[len(chans_in)-2], chans_out[len(chans_out)-2]))

        self.last_up = None
        if last_upsample > 1:
            self.last_up = nn.Upsample(scale_factor=last_upsample)
            



    def forward(self, x):
        y = x[-1]
        for i in range(len(self.upsamples)-1, -1, -1):
            y = self.upsamples[i](y)
            y = torch.concat([y, x[i]], dim = 1)
            y = self.convs[i](y)

        if self.last_up is not None:
            y = self.last_up(y)

        return y




class Classifier(nn.Module):
    def __init__(self, depth, n_classes, last_activation = nn.Softmax):
        super().__init__()
        self.last_conv = nn.Conv2d(depth, n_classes, kernel_size=1)
        self.last_act = last_activation(dim=1)


    def forward(self, x):
        x = self.last_conv(x)
        x = self.last_act(x)
        return x

class SkipConn(nn.Module):
    def __init__(self, in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #self.in_channels = in_channels
        #self._out_channels = [n_chan + 1 for n_chan in in_channels]
        #self._out_channels = [n_chan for n_chan in in_channels]
        #self._out_channels = [2*n_chan for n_chan in in_channels] #dual encoders
        self._out_channels = [2*n_chan + 1  for n_chan in in_channels]

    def out_channels(self):
        return self._out_channels

    def forward(self, x_imgs, x_prev):
        return [torch.concat([x_imgs[i], x_prev[i]], dim=1) for i in  range(len(x_imgs))]
        #return x_imgs