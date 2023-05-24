import os
import pickle
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 类别名
label_name = ["airplane", "automobile", "bird","cat", "deer", "dog","frog", "horse", "ship", "truck"]


DATA_BATCHES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
TEST_BATCHES = ['test_batch']


class Cifar10(Dataset):
    def __init__(self, root: str, train: bool, transform=None):
        super(Cifar10, self).__init__()
        self.imgs = []
        self.labels = []
        batches = DATA_BATCHES if train else TEST_BATCHES
        for batch in batches:
            with open(os.path.join(root, batch),'rb') as f:
                img_dict=pickle.load(f, encoding='bytes')
                # b'batch_label', b'labels', b'data', b'filenames'
                self.imgs.append(np.reshape(img_dict[b'data'], (-1, 3, 32, 32)))
                self.labels.append(img_dict[b'labels'])
        self.imgs = np.concatenate(self.imgs, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        print(self.imgs.shape, self.labels.shape)
 
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.N_CLS = 10

    def __getitem__(self, index):
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.N_CLS * [0]
        label[self.labels[index]] = 1
        label = torch.LongTensor(label)
 
        return img, label
 
    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Cifar10('D:\\Datasets\\cifar-10-python\\cifar-10-batches-py', train=False)
    print(dataset.imgs[:3], dataset.labels[:3])
    data = DataLoader(dataset, batch_size=5)

    for epoch in range(1):
        print("Epoch {}".format(epoch))
        for step, (imgs, labels) in enumerate(data):
            print("step {}: {}".format(step, labels))
