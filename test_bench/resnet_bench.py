import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .base_bench import BaseBench


os.environ['TORCH_HOME']='D:/dataset/torchvision' 


resnet_dict = {
    'resnet18': 512,
    'resnet50': 1024,
    'resnet101': 1024,
}


def make_resnet(n_cls, name='resnet18'):
    resnet = eval('torchvision.models.' + name)(pretrained=True)
    resnet.fc = torch.nn.Linear(resnet_dict[name], n_cls)
    return resnet


class ResnetBench(BaseBench):
    def __init__(self, n_cls, name, cuda=True):
        self.name = name
        self.resnet = make_resnet(n_cls, name)
        if cuda:
            self.resnet = self.resnet.cuda()
        self.cuda = cuda

    def model(self):
        return self.resnet

    def train(self, data, iteration=200, lr=1e-3, betas=(5e-3, 5e-3), save_root='./checkpoints/'):
        optim = torch.optim.Adam(self.resnet.fc.parameters(), lr=lr, betas=betas)
        self.resnet.eval()
        for epoch in range(iteration):
            self.resnet.fc.train()
            total_loss = 0
            for step, (imgs, labels) in enumerate(data):
                if self.cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                preds = self.resnet(imgs)
                loss = F.cross_entropy(preds, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()
            print(total_loss)
            
            if epoch % 5 == 0:
                self.save(os.path.join(save_root, self.name, 'epoch{}.pth'.format(epoch)))

    def test(self, data):
        self.resnet.eval()
        preds = []
        labels = []
        for img, label in data:
            if self.cuda:
                img = img.cuda()
                label = label.cuda()
            pred = self.resnet(img)
            preds.append(F.argmax(pred))
            labels.append(F.argmax(label))
        confusion_matrix = F.confusion_matrix(preds, labels)
        return preds, confusion_matrix

    def save(self, path):
        torch.save(self.resnet.fc.state_dict(), path)

    def load(self, path):
        self.resnet.fc.load_state_dict(torch.load(path))
            