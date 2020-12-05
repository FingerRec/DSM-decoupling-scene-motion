from torch import nn
from augment.basic_augmentation.spatial_wrap import spatial_warp_torch
from augment.warpping.video_warpping_api import Video_Warp
from augment.stn.stn import STN_API
import numpy as np


class GenPositive(nn.Module):
    def __init__(self, prob=0.3):
        super(GenPositive, self).__init__()
        self.video_warp_api = Video_Warp()
        self.stn = STN_API()
        self.prob = prob

    def video_warp(self, x):
        return self.video_warp_api(x)

    def video_st(self, x):
        return self.stn(x)

    def spatial_wrap(self, x):
        low_x = 60
        up_x = 140
        low_y = 60
        up_y = 140
        low_bound = 10
        up_bound = 20
        for i in range(np.random.randint(1, 4)):
            rand1 = (
            np.random.randint(low_x, up_x), np.random.randint(low_y, up_y), np.random.randint(low_bound, up_bound),
            np.random.randint(low_bound, up_bound))
            x = spatial_warp_torch(x, rand1)
        return x

    def forward(self, x):
        x = self.spatial_wrap(x)
        return x
