"""
the code for smmoth temporal augmentation
"""

import numpy as np
import torch.nn as nn


class STA(nn.Module):
    def __init__(self, length=16, max_degree=30):
        super(STA, self).__init__()
        self.T = length

    def smooth_rotation(self, x):
        bsz = x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        return lam

    def smooth_crop(self):
        c = None
        return c

    def forward(self, x, return_conv=False):
        return x
