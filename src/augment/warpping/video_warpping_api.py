# encoding: utf-8

import time
import torch
import itertools
import numpy as np
import torch.nn as nn
from PIL import Image
from augment.warpping.grid_sample import grid_sample
from torch.autograd import Variable
from augment.warpping.tps_grid_gen import TPSGridGen

# source_image = Image.open('demo/source_avatar.jpg').convert(mode = 'RGB')
# source_image = np.array(source_image).astype('float32')
# source_image = np.expand_dims(source_image.swapaxes(2, 1).swapaxes(1, 0), 0)
# source_image = Variable(torch.from_numpy(source_image))
# _, _, source_height, source_width = source_image.size()
# target_height = 400
# target_width = 400
#
# # creat control points
# target_control_points = torch.Tensor(list(itertools.product(
#     torch.arange(-1.0, 1.00001, 2.0 / 4),
#     torch.arange(-1.0, 1.00001, 2.0 / 4),
# )))
# source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-0.1, 0.1)
#
# print('initialize module')
# beg_time = time.time()
# tps = TPSGridGen(target_height, target_width, target_control_points)
# past_time = time.time() - beg_time
# print('initialization takes %.02fs' % past_time)
#
# source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))
# grid = source_coordinate.view(1, target_height, target_width, 2)
# canvas = Variable(torch.Tensor(1, 3, target_height, target_width).fill_(255))
# target_image = grid_sample(source_image, grid, canvas)
# target_image = target_image.data.numpy().squeeze().swapaxes(0, 1).swapaxes(1, 2)
# target_image = Image.fromarray(target_image.astype('uint8'))
# target_image.save('demo/target_avatar.jpg')


class Video_Warp(nn.Module):
    def __init__(self, scale=1):
        super(Video_Warp, self).__init__()
        self.scale = scale
        # creat control points

    def forward(self, video):
        bsz = video.size(0)
        target_control_points = torch.Tensor(list(itertools.product(torch.arange(-1.0/self.scale, 1.00001/self.scale, 2.0 / 4 / self.scale),
                                                                    torch.arange(-1.0/self.scale, 1.00001/self.scale, 2.0 / 4 / self.scale),))).cuda()
        source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-0.1, 0.1).cuda()
        b, c, t, h, w = video.size()
        tps = TPSGridGen(h, w, target_control_points)
        source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)).cuda())
        grid = source_coordinate.view(1, h, w, 2).repeat(bsz, 1, 1, 1)
        canvas = Variable(torch.Tensor(1, 3, h, w).fill_(255)).cuda()
        target_video = video.clone()
        for i in range(t):
            target_video[:, :, i, :, :] = grid_sample(video[:, :, i, :, :], grid)
        return target_video