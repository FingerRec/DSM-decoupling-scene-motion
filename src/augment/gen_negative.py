from torch import nn
from .basic_augmentation.eda import VideoEda


class GenNegative(nn.Module):
    def __init__(self, prob=0.3):
        super(GenNegative, self).__init__()
        self.prob = prob
        self.t_eda = VideoEda()

    def temporal_dropout(self, x):
        return self.t_drop(x)

    def temporal_shuffle(self, x):
        return self.t_shuffle(x)

    def temporal_eda(self, x):
        return self.t_eda(x)

    def forward(self, x):
        # x = self.temporal_shuffle(x)
        # x = self.temporal_dropout(x)
        x = self.temporal_eda(x)
        return x