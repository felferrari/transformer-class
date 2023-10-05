from torch import nn
import torch
from ..resunet.layers import ResidualBlock

class Decoder(nn.Module):
    def __init__(self, chan_in, chan_out, last_upsample, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2)
            for i in range(len(chan_in)-1)
        ])

        self.convs = nn.ModuleList([
            #nn.Sequential(
            #    nn.Conv2d(chan_in[i] + chan_out[i+1], chan_out[i], 3, padding = 1),
            #    #nn.BatchNorm2d(chan_out[i]),
            #    nn.ReLU()
            #)
            ResidualBlock(chan_in[i] + chan_out[i+1], chan_out[i])
            for i in range(len(chan_in)-2)
        ])
        self.convs.append(
            #nn.Sequential(
            #    nn.Conv2d(chan_in[-1] + chan_in[-2], chan_out[-2], 3, padding = 1),
            #    #nn.BatchNorm2d(chan_out[-2]),
            #    nn.ReLU()
            #)
            ResidualBlock(chan_in[-1] + chan_in[-2], chan_out[-2])
        )

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

class SkipConnConcat(nn.Module):
    def __init__(self, in_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels

    def out_channels(self):
        return self.in_channels

    def forward(self, x):
        return x
    
class JointFusionConcat(nn.Module):
    def __init__(self, input_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = input_channels

    def out_channels(self):
        return [self.in_channels[0][i] + self.in_channels[1][i] for i in range(len(self.in_channels[0]))]

    def forward(self, x):
        return [torch.cat([x[0][i], x[1][i]], dim = 1) for i in range(len(x[0]))]