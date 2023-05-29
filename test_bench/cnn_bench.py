import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .base_bench import BaseBench


class CNN(nn.Module):
    def __init__(self, n_cls):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)  
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_cls)

    def forward(self, x): 
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) 
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) 
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)        
        return x


class CnnBench(BaseBench):
    def __init__(self, n_cls, name, cuda=True):
        self.name = name
        self.net = CNN(n_cls)
        if cuda:
            self.net = self.net.cuda()
        self.cuda = cuda

    def model(self):
        return self.net

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
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
            