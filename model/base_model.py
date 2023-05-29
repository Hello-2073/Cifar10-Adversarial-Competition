import abc
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(abc.ABC):
    def __init__(self, black_box: nn.Module):
        self.net = black_box

    def forward(self, img):
        pass
