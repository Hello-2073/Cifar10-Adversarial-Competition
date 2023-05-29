import time
from tqdm import tqdm, trange
import torch.nn as nn

from .base_model import BaseModel

class JacobianModel(BaseModel):
    def __init__(self, black_box: nn.Module, attack_rate: float):
        self.net = black_box
        self.attack_rate = attack_rate

    def forward(self, imgs, labels):
        for i in trange(imgs.size[0]):
            img, label = imgs[i], labels[i]
            jacobian = torch.autograd.jacobian(func=self.net, inputs=img)
            imgs[i] += self.attack_rate * torch.sign(jacobian[label])

        return imgs