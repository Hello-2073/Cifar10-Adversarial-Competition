import os
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class Cifar10Clean500(Dataset):
    r"""
    Official cifar10_clean_500 dataset.
    """
    def __init__(self, root, attack_id=0):
        super(Cifar10Clean500, self).__init__()
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.paths = []
        self.labels = []
        with open(os.path.join(root, 'label.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                content = line.split(' ')
                filename = content[0]
                label = content[1]
                if attack_id > 0:
                    self.paths.append(os.path.join(root, 'attack_{}'.format(attack_id), filename))
                else:
                    self.paths.append(os.path.join(root, 'images', filename))
                self.labels.append(int(label))
        self.num_cls = max(self.labels) + 1

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.num_cls * [0]
        label[self.labels[index]] = 1
        label = torch.LongTensor(label)
        return img, label

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':
    dataset = Cifar10Clean500('D:\\Datasets\\cifar10_clean_500')
    print(dataset.paths[:3], dataset.labels[:3])
    data = DataLoader(dataset, batch_size=5)

    for epoch in range(1):
        print("Epoch {}".format(epoch))
        for step, (imgs, labels) in enumerate(data):
            print("step {}: {}".format(step, labels))
