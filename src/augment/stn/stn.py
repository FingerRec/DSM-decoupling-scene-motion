import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import random
import numpy as np


class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class STN_API(nn.Module):
    def __init__(self):
        super(STN_API, self).__init__()

    def maxtrix_padding(self, mat, size):
        new_mat = torch.zeros(size).cuda()
        new_mat[:, -1, -1] = 1
        new_mat[:, :mat.size(1), :] = mat
        return new_mat

    def matrix_reduce(self, mat, size):
        return mat[:size[0], :size[1], :size[2]]

    def rotate(self, max_degree=45):
        degree = np.random.randint(0, max_degree)
        angle = -degree * math.pi / 180
        mat = torch.zeros((2, 3)).cuda()
        mat[0, 0] = math.cos(angle)
        mat[1, 1] = math.cos(angle)
        mat[0, 1] = -math.sin(angle)
        mat[1, 0] = math.sin(angle)
        return mat

    def scale(self, max_scale=1.15, min_scale=0.85):
        scale_x = random.random() * (max_scale - min_scale) + min_scale
        scale_y = random.random() * (max_scale - min_scale) + min_scale
        mat = torch.zeros((2, 3)).cuda()
        mat[0, 0] = math.cos(scale_x)
        mat[1, 1] = math.cos(scale_y)
        return mat

    def translation(self, up_bound_x=0.3, up_bound_y=0.3):
        tran_x = random.random() * up_bound_x
        tran_y = random.random() * up_bound_y
        mat = torch.zeros((2, 3)).cuda()
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[0, 2] = tran_x
        mat[1, 2] = tran_y
        return mat

    def shear(self, up_bound_x=0.3, up_bound_y=0.3):
        mat = torch.zeros((2, 3)).cuda()
        shear_x = random.random() * up_bound_x
        shear_y = random.random() * up_bound_y
        mat[0, 0] = 1
        mat[1, 1] = 1
        mat[0, 1] = shear_x
        mat[1, 0] = shear_y
        return mat

    def reflection(self):
        mat = torch.zeros((2, 3)).cuda()
        prob = random.random()
        if prob < 0.5:
            mat[0, 0] = -1
            mat[1, 1] = 1
        else:
            mat[0, 0] = 1
            mat[1, 1] = -1
        return mat

    # random generate affnix matrix
    def forward(self, x):
        bsz, c, t, h, w = x.size()
        output = torch.zeros_like(x).cuda()
        theta = torch.zeros((bsz, 2, 3)).cuda()
        for j in range(bsz):
            # theta[j] = self.rotate()
            theta[j] = self.scale()
            # theta[j] = self.translation()
            # theta[j] = self.shear()
            # theta[j] = self.reflection()
        grid = F.affine_grid(theta, (bsz, c, h, w))
        for i in range(t):
            output[:, :, i, :, :] = F.grid_sample(x[:, :, i, :, :], grid)
        # new_theta = self.maxtrix_padding(theta, (bsz, 3, 3))
        # # for each sample in batch size, should get inverse
        # inverse_theta = new_theta.clone()
        # for j in range(bsz):
        #     inverse_theta[j] = torch.inverse(new_theta[j])
        # inverse_theta = self.matrix_reduce(inverse_theta, (bsz, 2, 3))
        # inverse_grid = F.affine_grid(inverse_theta, (bsz, c, h, w))
        # for i in range(t):
        #     output[:, :, i, :, :] = F.grid_sample(x[:, :, i, :, :], inverse_grid)
        return output
