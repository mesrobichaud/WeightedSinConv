import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter

from coordconv import CoordConv, AddCoords

'''
An alternative implementation for PyTorch with auto-infering the x-y dimensions.
'''


class AddSine(AddCoords):
    def __init__(self, alpha=0.5, beta=None, phase_shift=0.):
        super(AddSine, self).__init__(False)
        if beta is None:
            beta = alpha
        self.alpha = Parameter(torch.FloatTensor([alpha]))
        self.beta = Parameter(torch.FloatTensor([beta]))
        self.phase = Parameter(torch.FloatTensor([phase_shift]))

    def generate_xy(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()

        sx = self.phase

        xx_channel = torch.linspace(0., 1., x_dim).repeat(1, y_dim, 1).to(self.phase.device)
        yy_channel = torch.linspace(0., 1., y_dim).repeat(1, x_dim, 1).transpose(1, 2).to(self.phase.device)

        xx_channel = xx_channel.float() * self.alpha
        yy_channel = yy_channel.float() * self.beta

        channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3) + yy_channel.repeat(batch_size, 1, 1,
                                                                                             1).transpose(2, 3)
        channel = torch.sin(channel + sx)
        return channel

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                                         "alpha=" + str(self.alpha.item()) + \
               " beta=" + str(self.beta.item()) + \
               " phase=" + str(self.phase.item()) + ")"

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """

        xx_channel = self.generate_xy(input_tensor)
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor)], dim=1)

        return ret
    
class AddHSine(AddCoords):
    def __init__(self, alpha=0.5, beta=None, phase_shift=0., square = 0.):
        super(AddHSine, self).__init__(False)
        if beta is None:
            beta = alpha
        self.alpha = Parameter(torch.FloatTensor([alpha]))
        self.beta = Parameter(torch.FloatTensor([beta]))
        self.phase = Parameter(torch.FloatTensor([phase_shift]))
        self.square = Parameter(torch.FloatTensor([square]))

    def generate_xy(self, input_tensor):
        batch_size, _, x_dim, y_dim = input_tensor.size()

        sx = self.phase
        sq = self.square
        
        f = torch.sqrt(self.alpha**2+self.beta**2)

        xx_channel = torch.linspace(0., 1., x_dim).repeat(1, y_dim, 1).to(self.phase.device)
        yy_channel = torch.linspace(0., 1., y_dim).repeat(1, x_dim, 1).transpose(1, 2).to(self.phase.device)

        xx_1 = xx_channel.float() * self.alpha
        yy_1 = yy_channel.float() * self.beta
        
        xx_2 = xx_channel.float() * 3*self.alpha
        yy_2 = yy_channel.float() * 3*self.beta
        
        xx_3 = xx_channel.float() * 5*self.alpha
        yy_3 = yy_channel.float() * 5*self.beta

        channel1 = xx_1.repeat(batch_size, 1, 1, 1).transpose(2, 3) + yy_1.repeat(batch_size, 1, 1,
                                                                                             1).transpose(2, 3)
        channel2 = xx_2.repeat(batch_size, 1, 1, 1).transpose(2, 3) + yy_2.repeat(batch_size, 1, 1,
                                                                                             1).transpose(2, 3)
        channel3 = xx_3.repeat(batch_size, 1, 1, 1).transpose(2, 3) + yy_3.repeat(batch_size, 1, 1,
                                                                                             1).transpose(2, 3)
        channel = torch.sin(channel1 + sx/f)+ 1/3*sq*torch.sin(channel2 + sx*3/f)+1/5*sq*torch.sin(channel3 + sx*5/f)
        
        
        return channel

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
                                         "alpha=" + str(self.alpha.item()) + \
               " beta=" + str(self.beta.item()) + \
               " phase=" + str(self.phase.item()) + \
               " square=" + str(self.square.item()) + ")"

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """

        xx_channel = self.generate_xy(input_tensor)
        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor)], dim=1)

        return ret


class SinConv(CoordConv):
    def __init__(self, in_channels, out_channels, sins=[], **kwargs):
        super(SinConv, self).__init__(in_channels, out_channels, **kwargs)
        self.addcoords = nn.Sequential(*[AddSine(a, b, p) for a, b, p in sins])
        in_size = in_channels + len(sins)
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

class SquareConv(CoordConv):
    def __init__(self, in_channels, out_channels, sins=[], **kwargs):
        super(SquareConv, self).__init__(in_channels, out_channels, **kwargs)
        self.addcoords = nn.Sequential(*[AddHSine(a, b, p, s) for a, b, p, s in sins])
        in_size = in_channels + len(sins)
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)