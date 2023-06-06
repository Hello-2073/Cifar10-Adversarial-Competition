import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from .base_bench import BaseBench


class CNN(nn.Module):
    def __init__(self, n_cls):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0)
        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  
        self.fc1 = nn.Linear(400, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_cls)

    def forward(self, x): 
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=x.view(-1,16*5*5) #卷积结束后将多层图片平铺batchsize行16*5*5列，每行为一个sample，16*5*5个特征
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

    def train(self, data, iteration=200, lr=1e-2, save_root='./checkpoints/'):
        optim = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        self.net.eval()
        train_loss = []# 记录batch训练过程中的loss变化
        for epoch in range(iteration):
            self.net.train()
            total_loss = 0
            for step, (imgs, labels) in enumerate(data):
                if self.cuda:
                    imgs = imgs.cuda()
                    labels = labels.cuda()
                # import pdb; pdb.set_trace();
                preds = self.net(imgs)
                preds = F.softmax(preds, dim=1)
                # print("preds:{}, labels:{}".format(preds, labels))
                # print(preds)
                loss = F.cross_entropy(preds, labels)
                # print(preds, labels)
                # print("step: {}, loss: {}".format(step, loss))
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.item()
                # train_loss.append(loss.item())  # 记录最终训练误差      
                # print("step: {}, loss: {}".format(step, total_loss))
                if step % 1000 == 999:    # 每2000批次打印一次
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, step + 1, total_loss / 2000))
                    total_loss = 0.0
            print("epoch: {}, loss: {}".format(epoch, total_loss/len(data)))
            train_loss.append(total_loss/len(data))  # 记录最终训练误差      
        plt.plot(range(len(train_loss)),train_loss)
        plt.xlabel("Epoch")
        plt.ylabel("loss")
        plt.title("Train loss")
        plt.show()
            # filepath = os.path.join(save_root, self.name)
            # if not os.path.isdir(filepath):
            #     # 创建文件夹
            #     os.mkdir(filepath)
            # if epoch % 1 == 0:
            #     self.save(os.path.join(filepath, 'epoch{}.pth'.format(epoch)))

    def test(self, data):
        self.net.eval()
        preds = []
        labels = []
        for img, label in data:
            if self.cuda:
                img = img.cuda()
                label = label.cuda()
            pred = self.net(img)
            preds.append(torch.argmax(pred))
            labels.append(torch.argmax(label))
        # confusion_matrix = F.confusion_matrix(preds, labels)
        acc = 0
        for i in range(len(preds)):
            if preds[i] == labels[i]:
                acc += 1
        acc /= len(preds)
        print('Accuracy on test set:%d %%' % (acc))
        return preds, acc

    def save(self, path):
        print(path)
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
            