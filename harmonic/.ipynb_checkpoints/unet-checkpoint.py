import torch
import torch.nn as nn
import torch.nn.functional as F

from harmonic import SinConv
from torchvision import models

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class ups(nn.Module):
    def __init__(self, in_ch, out_ch, sins, kn=7):
        super(ups, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        
        if (kn == 0):
            self.sinconv = Identity()
            self.conv = double_conv(in_ch, out_ch)
        else:
            self.sinconv = SinConv(in_ch, out_ch, sins=sins, kernel_size=kn, padding=int(kn/2))
            self.conv = double_conv(out_ch, out_ch)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        #x1 = F.upsample(x1, scale_factor=2, mode='bilinear', align_corners=True)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.sinconv(x)
        x = self.conv(x)
        return x
    
class eups(nn.Module):
    def __init__(self, in_ch, out_ch, sins, kn=7):
        super(eups, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        
        if (kn == 0):
            self.sinconv = Identity()
            self.conv = double_conv(int(in_ch//8*5), out_ch)
        else:
            #self.sinconv = SinConv(in_ch//2, out_ch, sins=sins, kernel_size=kn, padding=int(kn/2))
            #self.conv = double_conv(out_ch//4+in_ch//2, out_ch)
            # change: base2
            self.sinconv = SinConv(in_ch//2,in_ch, sins=sins, kernel_size=kn, padding=int(kn/2))
            self.conv = double_conv(in_ch//4+in_ch//2, out_ch)

    def forward(self, x1, x2):
        #x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        #x1 = F.upsample(x1, scale_factor=2, mode='bilinear', align_corners=True)
        
        x = self.sinconv(x1)
        x1 = F.pixel_shuffle(x, 2)
        
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, sins):
        super(outconv, self).__init__()
        self.sinconv = SinConv(in_ch, in_ch, sins=sins, kernel_size=7, padding=3)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.sinconv(x)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, sins,up_k=[7,7,7,7]):
        super(UNet, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = ups(1024, 256, sins, up_k[0])
        self.up2 = ups(512, 128, sins, up_k[1])
        self.up3 = ups(256, 64, sins, up_k[2])
        self.up4 = ups(128, 64, sins, up_k[3])
        self.outc = outconv(64, len(sins), sins)

    def freeze_encoder(self, freeze):
        return None

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    
class eUNet(nn.Module):
    def __init__(self, n_channels, sins,up_k=[7,7,7,7]):
        super(eUNet, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = eups(1024, 256, sins, up_k[0])
        self.up2 = eups(512, 128, sins, up_k[1])
        self.up3 = eups(256, 64, sins, up_k[2])
        self.up4 = eups(128, 64, sins, up_k[3])
        self.outc = outconv(64, len(sins), sins)

    def freeze_encoder(self, freeze):
        return None

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    
class nUNet(nn.Module):
    def __init__(self, n_channels, sins,up_k=[7,7,7,7]):
        super(nUNet, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = eups(1024, 256, sins, up_k[0])
        self.up2 = eups(512, 128, sins, up_k[1])
        self.up3 = eups(256, 64, sins, up_k[2])
        self.up4 = eups(128, 64, sins, up_k[3])
        #self.outc = outconv(64, len(sins), sins)
        self.outc4 = out_mod(64, len(sins))
        self.outc3 = out_mod(64, len(sins))
        self.outc2 = out_mod(128, len(sins))
        self.outc1 = out_mod(256, len(sins))

    def freeze_encoder(self, freeze):
        return None

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        y = self.outc1(x)
        x = self.up2(x, x3)
        y = self.outc2(x,y)
        x = self.up3(x, x2)
        y = self.outc3(x,y)
        x = self.up4(x, x1)
        
        x = self.outc4(x,y)
        return x
    
class aUNet(nn.Module):
    def __init__(self, n_channels, sins):
        super(aUNet, self).__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = ups(1024, 256, sins)
        self.up2 = ups(512, 128, sins)
        self.up3 = ups(256, 64, sins)
        self.up4 = ups(128, 64, sins)
        #self.outc = outconv(64, len(sins), sins)
        self.outc4 = out_mod(64, len(sins))
        self.outc3 = out_mod(64, len(sins))
        self.outc2 = out_mod(128, len(sins))
        self.outc1 = out_mod(256, len(sins))

    def freeze_encoder(self, freeze):
        return None

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        y = self.outc1(x)
        x = self.up2(x, x3)
        y = self.outc2(x,y)
        x = self.up3(x, x2)
        y = self.outc3(x,y)
        x = self.up4(x, x1)
        
        x = self.outc4(x,y)
        return x
    
    
class bUNet(nn.Module):
    def __init__(self, n_channels, sins):
        super(bUNet, self).__init__()

        self.backbone = models.resnet34(pretrained = True)
        
        self.up1 = ups(768, 384, sins) ## 256 from layer3 
        self.up2 = ups(512, 192, sins) ## 128 from layer2
        self.up3 = ups(256, 64, sins)  ## 64 from layer1
        self.up4 = ups(128, 64, sins)  ## 64 from input 
        self.outc = outconv(64, len(sins), sins)
        
        self.backbone.avgpool = Identity()
        self.backbone.fc = Identity()

        
    def forward(self, x):
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x1 = self.backbone.relu(x)
        
        x2 = self.backbone.layer1(x1) ## OUT 64
        x3 = self.backbone.layer2(x2) ## OUT 128
        x4 = self.backbone.layer3(x3) ## OUT 256
        x5 = self.backbone.layer4(x4) ## OUT 512

        x = self.up1(x5, x4) ## 512 + 256 
        x = self.up2(x, x3) ## 384 + 128
        x = self.up3(x, x2) ## 256 + 64 
        x = self.up4(x, x1) ## 64
        x = self.outc(x) ## 12
        return x
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class out_mod(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch,conv_op=nn.Conv2d):
        super(out_mod, self).__init__()
        self.outc = double_conv(in_ch, in_ch)
        self.attention = nn.Sequential(conv_op(in_ch,out_ch,3,1,1),
                                     nn.Sigmoid())
        self.outc2 =conv_op(in_ch,out_ch,1) # 12D output
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, xr, y=None):
        x = self.outc2(self.outc(xr))
        if y is not None:
            att = self.attention(xr)
            x = att*x + (1.-att)*self.up(y)
        return x