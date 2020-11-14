import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
#from cc_attention.functions import CrissCrossAttention
from cc_attention.CC import CC_module as CrissCrossAttention
import math

class BottomUp(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(BottomUp, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))

        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x):
        global_fms, global_outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])
            else:
                feature = self.laterals[i](x[i]) + up
            global_fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = self.predict[i](feature)
            global_outs.append(feature)

        return global_fms, global_outs

class RCCA(nn.Module):
    def __init__(self, in_channel):
        super(RCCA, self).__init__()
        self.criss_cross = CrissCrossAttention(in_channel)
        self.bn = nn.BatchNorm2d(in_channel) # new

    def forward(self, x, recurrent=2):
        for i in range(recurrent):
            x = self.criss_cross(x)
            #x = self.bn(x) # new
        x = self.bn(x) # new
        return x

class Network(nn.Module):
    def __init__(self, dataset, flag):
        super(Network, self).__init__()
        self.flag = flag

        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.select = {
            'conv1': (64, 112, 112),   # [batch_size, 64, 112, 112]
            'bn1': (64, 112, 112),
            'relu': (64, 112, 112),
            'maxpool': (64, 56, 56),
            'layer1': (256, 56, 56),  # [batch_size, 256, 56, 56]
            'layer2': (512, 28, 28),  # [batch_size, 512, 28, 28]
            'layer3': (1024, 14, 14),  # [batch_size, 1024, 14, 14]
            'layer4': (2048, 7, 7),  # [batch_size, 2048, 7, 7]
        }
        self.position = [
            'layer1',
            'layer2',
            'layer3',
            'layer4',
            ]
        self.channel_settings = \
            [self.select[layer][0] for layer in self.position]

        self.bottom_up = BottomUp(self.channel_settings[::-1], (224, 224), 8)
        if self.flag:
            ccas = [ \
                RCCA(channel) \
                for channel in self.channel_settings \
                ]
            self.ccas = nn.ModuleList(ccas)

    def forward(self, sample):
        x = sample['image']
        xs = []
        for name, layer in self.resnet._modules.items():
            if name not in self.select:
                break
            x = layer(x)
            if name in self.position:
                if self.flag:
                    x1 = self.ccas[self.position.index(name)](x)
                    xs.append(x1)
                else:
                    xs.append(x)
        fms, outs = self.bottom_up(xs[::-1])
        lm_pos_map = sum(outs) / len(outs)
        
        return {'lm_pos_map' : lm_pos_map}
