import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models

from .blocks import ResBlocks, Interpolate, LayerNorm
from utils import weights_init


def calculate_parameters(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        count = 1
        for j in i.size():
            count *= j
        k = k + count
    print(f"Number of total parameters: {str(k)}\n")


class ResNetClassifier(nn.Module):
    def __init__(self, num_class, init=None):
        super().__init__()
        self.avgpool = nn.AvgPool2d(7, 1, 0)
        self.fc = nn.Linear(2048, num_class)

        if init:
            self.apply(weights_init(init))

    def forward(self, h):
        x = self.avgpool(h)
        h = x.reshape(x.shape[0], -1)
        return self.fc(h)


class FCN8sSegment(nn.Module):
    def __init__(self, num_class, init=None):
        super().__init__()
        self.full_conv = nn.Sequential(
            nn.Conv2d(2048, 512, 1), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5),
            nn.Conv2d(512, 4096, 7), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, 4096, 1), nn.ReLU(inplace=True), nn.Dropout2d(p=0.5),
            nn.Conv2d(4096, num_class, 1))
        self.score_pool3 = nn.Conv2d(512, num_class, 1)
        self.score_pool4 = nn.Conv2d(1024, num_class, 1)
        # self.score_pool3 = nn.Conv2d(256, num_class, 1)
        # self.score_pool4 = nn.Conv2d(512, num_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            num_class, num_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            num_class, num_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_class, num_class, 4, stride=2, bias=False)

        if init:
            self.apply(weights_init(init))

        for param in self.score_pool3.parameters():
            nn.init.constant_(param, 0)
        for param in self.score_pool4.parameters():
            nn.init.constant_(param, 0)

    def _crop(self, input, shape, offset=0):
        _, _, h, w = shape
        return input[:, :, offset:offset + h, offset:offset + w].contiguous()

    def forward(self, pool3, pool4, h, x_shape):
        h = self.full_conv(h)
        h = self.upscore2(h)
        upscore2 = h    # 1/32 -> 1/16

        h = self.score_pool4(pool4)
        h = self._crop(h, upscore2.shape, 5)
        score_pool4c = h  # 1/16
        fuse_pool4 = upscore2 + score_pool4c
        upscore_pool4 = self.upscore_pool4(fuse_pool4)

        score_pool3 = self.score_pool3(pool3)
        score_pool3c = self._crop(score_pool3, upscore_pool4.shape, offset=9)
        fuse_pool3 = upscore_pool4 + score_pool3c
        upscore8 = self.upscore8(fuse_pool3)

        score = self._crop(upscore8, x_shape, offset=31)
        return score


class ResDis(nn.Module):
    def __init__(self, input_dim, params, init=None):
        super().__init__()
        self.gan_type = params['gan_type']
        dim, num_downsample = params['dim'], params['num_downsample']

        model = [nn.Conv2d(input_dim, dim, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        for i in range(num_downsample - 1):
            model += [nn.Conv2d(dim, dim * 2, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            dim *= 2
        model += [nn.Conv2d(dim, 1, 1, 1, 0)]
        self.model = nn.Sequential(*model)

        if init:
            self.apply(weights_init(init))

    def forward(self, x):
        return self.model(x)

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = torch.zeros_like(out0.data).cuda().detach()
                all1 = torch.ones_like(out1.data).cuda().detach()
                loss += torch.mean(F.binary_cross_entropy(torch.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(torch.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2)
            elif self.gan_type == 'nsgan':
                all1 = torch.ones_like(out0.data).cuda().detach()
                loss += torch.mean(F.binary_cross_entropy(torch.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


##################################################################################
# Single task
##################################################################################
class Res101Gen(nn.Module):
    def __init__(self, init=None, pretrained=True):
        super().__init__()
        self.enc = ResNet101Encoder(pretrained)
        self.dec = ResNet101Decoder(init)

    def forward(self, x, t_shape):
        h, noise = self.encode(x, seg=False)
        x_recon = self.decode(h + noise, t_shape) if self.training else self.decode(h, t_shape)
        return x_recon, h

    def encode(self, x, seg):
        h = self.enc(x, seg)
        noise = torch.randn_like(h[-1]) if seg else torch.randn_like(h)
        return h, noise

    def decode(self, h, t_shape):
        recon_x = self.dec(h, t_shape)
        return recon_x


class ResNet101Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = models.resnet101(pretrained)
        self.conv_block1 = nn.Sequential(*[model.conv1, model.bn1, model.relu, model.maxpool])
        self.conv_block2 = model.layer1

    def forward(self, x, seg):
        x = F.pad(x, (99, 99, 99, 99), mode='constant', value=0) if seg else x
        h = self.conv_block1(x)
        h = self.conv_block2(h)
        return h


class ResNet101Decoder(nn.Module):
    def __init__(self, init=None):
        super().__init__()
        m_dim = 64
        self.model = nn.Sequential(ResBlocks(256, m_dim=m_dim, num_blocks=3),
                                   Interpolate(2, 'bilinear'), nn.Conv2d(256, 128, 3, 1, 1), LayerNorm(128), nn.ReLU(inplace=True),
                                   Interpolate(2, 'bilinear'), nn.Conv2d(128, 64, 3, 1, 1), LayerNorm(64), nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 3, 7, 1, 3), nn.Tanh())
        if init:
            self.apply(weights_init(init))

    def forward(self, h, t_shape):
        recon_x = self.model(h)
        recon_x = self._crop(recon_x, t_shape)
        return recon_x

    def _crop(self, x, shape):
        _, _, h, w = shape
        offset_h = (x.shape[2] - h) // 2
        offset_w = (x.shape[3] - w) // 2
        return x[:, :, offset_h:offset_h + h, offset_w:offset_w + w].contiguous()


class ResNet101EncoderBridge(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = models.resnet101(pretrained)
        self.conv_block3 = model.layer2
        self.conv_block4 = model.layer3
        self.conv_block5 = model.layer4

    def forward(self, h, seg):
        pool3 = self.conv_block3(h)
        pool4 = self.conv_block4(pool3)
        pool5 = self.conv_block5(pool4)
        if seg:
            return pool3, pool4, pool5
        else:
            return pool5


##################################################################################
# Multi-task
##################################################################################
class Res101GenShare(nn.Module):
    def __init__(self, init=None, pretrained=True):
        super().__init__()
        self.enc = ResNet101EncoderShare(pretrained)
        self.dec = ResNet101DecoderShare(init)

    def forward(self, x, t_shape):
        h, noise = self.encode(x, seg=False)
        x_recon = self.decode(h + noise, t_shape) if self.training else self.decode(h, t_shape)
        return x_recon, h

    def encode(self, x, seg, s):
        h = self.enc(x, seg, s)
        n = h[-1] if seg else h
        noise = torch.randn_like(n)
        return h, noise

    def decode(self, h, t_shape, s):
        recon_x = self.dec(h, t_shape, s)
        return recon_x


class ResNet101EncoderShare(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        model = models.resnet101(pretrained)
        self.conv_block1 = nn.Sequential(*[model.conv1, model.bn1, model.relu, model.maxpool])
        self.conv_block2 = model.layer1
        model2 = models.resnet101(pretrained)
        self.conv_block1_2 = nn.Sequential(*[model2.conv1, model2.bn1, model2.relu, model2.maxpool])

    def forward(self, x, seg, s):
        x = F.pad(x, (99, 99, 99, 99), mode='constant', value=0) if seg else x
        h = self.conv_block1_2(x) if s else self.conv_block1(x)
        h = self.conv_block2(h)
        return h


class ResNet101DecoderShare(nn.Module):
    def __init__(self, init=None):
        super().__init__()
        m_dim = 64
        self.model1 = nn.Sequential(ResBlocks(256, m_dim=m_dim, num_blocks=3))
        self.model2_1 = nn.Sequential(Interpolate(2, 'bilinear'), nn.Conv2d(256, 128, 3, 1, 1), LayerNorm(128), nn.ReLU(inplace=True),
                                      Interpolate(2, 'bilinear'), nn.Conv2d(128, 64, 3, 1, 1), LayerNorm(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 3, 7, 1, 3), nn.Tanh())
        self.model2_2 = nn.Sequential(Interpolate(2, 'bilinear'), nn.Conv2d(256, 128, 3, 1, 1), LayerNorm(128), nn.ReLU(inplace=True),
                                      Interpolate(2, 'bilinear'), nn.Conv2d(128, 64, 3, 1, 1), LayerNorm(64), nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 3, 7, 1, 3), nn.Tanh())
        if init:
            self.apply(weights_init(init))

    def forward(self, h, t_shape, s):
        recon_x = self.model1(h)
        recon_x = self.model2_1(recon_x) if s else self.model2_2(recon_x)
        return recon_x

    def _crop(self, x, shape):
        _, _, h, w = shape
        offset_h = (x.shape[2] - h) // 2
        offset_w = (x.shape[3] - w) // 2
        return x[:, :, offset_h:offset_h + h, offset_w:offset_w + w].contiguous()


if __name__ == '__main__':
    from blocks import ResBlocks, Interpolate, LayerNorm
    # print("ResNet101Encoder")
    # m = ResNet101Encoder(False)
    # calculate_parameters(m)

    print("ResNet101Decoder")
    m = ResNet101Decoder()
    calculate_parameters(m)

    print("ResNet101EncoderBridge")
    m = ResNet101EncoderBridge(False)
    calculate_parameters(m)

    print("ResDis (a)")
    params = {}
    params['gan_type'] = 'lsgan'
    params['dim'] = 16
    params['num_downsample'] = 4
    m = ResDis(3, params)
    calculate_parameters(m)

    print("ResDis (h)")
    params = {}
    params['gan_type'] = 'lsgan'
    params['dim'] = 64
    params['num_downsample'] = 2
    m = ResDis(256, params)
    calculate_parameters(m)

    print("FCN8sSegment")
    m = FCN8sSegment(20)
    calculate_parameters(m)
