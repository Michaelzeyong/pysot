# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    # print(f"batch: {batch}, channel: {channel}") # batch: 1, channel: 256
    # print(f"batch   : {kernel[:,0,0,0].numel()}")
    # print(f"channel : {kernel[0,:,0,0].numel()}")
    out = F.conv2d(x, kernel, groups=batch*channel)
    # print(f"x: {x}")
    # print(f"kernel: {kernel}")
    out = out.view(batch, channel, out.size(2), out.size(3))

    # this can avoid TracerWarning: Converting a tensor to a Python integer might cause the trace to be incorrect.
    # but useless
    # out = F.conv2d( x.view(1, kernel[:,0,0,0].numel()*kernel[0,:,0,0].numel(), x[0,0,:,0].numel(), x[0,0,0,:].numel()), 
    #                 kernel.view(kernel[:,0,0,0].numel()*kernel[0,:,0,0].numel(), 1, kernel[0,0,:,0].numel(), kernel[0,0,0,:].numel()), 
    #                 groups=256)
    # # print(f"x: {x}")
    # # print(f"kernel: {kernel}")
    # out = out.view(kernel[:,0,0,0].numel(), kernel[0,:,0,0].numel(), out[0,0,:,0].numel(), out[0,0,0,:].numel())


    return out
