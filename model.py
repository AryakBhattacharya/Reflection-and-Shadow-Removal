import torch
import torch.nn as nn
import torch.nn.functional as F

# UNet building block

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_bn=True, act='relu'):
        super().__init__()
        if down:
            layers = [nn.Conv2d(
                in_ch, out_ch, 4, 2, 1,
                padding_mode="reflect",  # important
                bias=not use_bn
            )]
        else:
            layers = [nn.ConvTranspose2d(
                in_ch, out_ch, 4, 2, 1,
                bias=not use_bn
            )]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        if act == 'relu':
            layers.append(nn.ReLU(inplace=False))
        else:
            layers.append(nn.LeakyReLU(0.2, inplace=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# Pix2Pix Generator

class Pix2PixGenerator(nn.Module):
    def __init__(self, in_ch=5, out_ch=3, base=64):
        super().__init__()
        # encoder
        self.d1 = UNetBlock(in_ch,    base,   down=True,  use_bn=False, act='lrelu')
        self.d2 = UNetBlock(base,     base*2, down=True,  act='lrelu')
        self.d3 = UNetBlock(base*2,   base*4, down=True,  act='lrelu')
        self.d4 = UNetBlock(base*4,   base*8, down=True,  act='lrelu')
        self.d5 = UNetBlock(base*8,   base*8, down=True,  act='lrelu')
        self.d6 = UNetBlock(base*8,   base*8, down=True,  act='lrelu')
        self.d7 = UNetBlock(base*8,   base*8, down=True,  act='lrelu')
        self.b  = UNetBlock(base*8,   base*8, down=True,  use_bn=False, act='lrelu')

        # decoder
        self.u1 = UNetBlock(base*8,   base*8, down=False)
        self.u2 = UNetBlock(base*16,  base*8, down=False)
        self.u3 = UNetBlock(base*16,  base*8, down=False)
        self.u4 = UNetBlock(base*16,  base*8, down=False)
        self.u5 = UNetBlock(base*16,  base*4, down=False)
        self.u6 = UNetBlock(base*8,   base*2, down=False)
        self.u7 = UNetBlock(base*4,   base,   down=False)

        self.out = nn.Sequential(
            nn.ConvTranspose2d(base*2, out_ch, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]

        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        b  = self.b(d7)

        u1 = self.u1(b)
        u2 = self.u2(torch.cat([u1, d7], dim=1))
        u3 = self.u3(torch.cat([u2, d6], dim=1))
        u4 = self.u4(torch.cat([u3, d5], dim=1))
        u5 = self.u5(torch.cat([u4, d4], dim=1))
        u6 = self.u6(torch.cat([u5, d3], dim=1))
        u7 = self.u7(torch.cat([u6, d2], dim=1))
        out = self.out(torch.cat([u7, d1], dim=1))

        # enforce exact size match
        if out.shape[-2] != H or out.shape[-1] != W:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out