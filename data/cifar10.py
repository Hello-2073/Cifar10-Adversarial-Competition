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
label_names = ["airplane", "automobile", "bird","cat", "deer", "dog","frog", "horse", "ship", "truck"]


# DATA_BATCHES = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
DATA_BATCHES = ['data_batch_1']
TEST_BATCHES = ['test_batch']


class Cifar10(Dataset):
    def __init__(self, root: str, train: bool, transform=None):
        super(Cifar10, self).__init__()
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.paths = []
        self.labels = []

        for index, label_name in enumerate(label_names):
            print("=================",label_name,"================")
            if train:
                base_path = os.path.join(root, 'cifar-10-output', 'train', label_name)
            else:
                base_path = os.path.join(root, 'cifar-10-output', 'train', label_name)
            for pic_name in os.listdir(base_path):
                # print("pic_name：", pic_name, "category", index)
                self.paths.append(os.path.join(base_path, pic_name))
                self.labels.append(index)
        self.N_CLS = 10

    def __getitem__(self, index):
        path = self.paths[index]
        print(path)
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.N_CLS * [0]
        label[self.labels[index]] = 1
        label = torch.LongTensor(label)
        return img, label
 
    def __len__(self):
        return self.N_CLS

      
def binary2img(root, batch, save_cls):
    with open(os.path.join(root, 'cifar-10-batches-py', batch), 'rb') as f:
        img_dict = pickle.load(f, encoding='bytes')
    #  img_dict字典有4个keys
    #    1 b'batch_label'：batch名称（文件名）
    #    2 b'labels' ：标签
    #    3 b'data'：图像数据
    #    4 b'filenames'：图像名称（用这个作为文件名）
    classification = {
        "0": "airplane",
        "1": "automobile",
        "2": "bird",
        "3": "cat",
        "4": "deer",
        "5": "dog",
        "6": "frog",
        "7": "horse",
        "8": "ship",
        "9": "truck",
    }
    for i in range(0, 10000):
        img = np.reshape(img_dict[b'data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        label = str(img_dict[b'labels'][i])
        save_path = os.path.join(root, 'cifar-10-output', save_cls, str(classification.get(label)))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pic_name = save_path + f'\\%d.png' % i
        Image.imread(pic_name, img)
  

if __name__ == '__main__':
    # root = '..\\dataset\\cifar-10-python'
    root = 'D:\\大三下课程\\人工智能安全导论\\Cifar10-Adversarial-Competition\\dataset\\cifar-10-python'
    dataset = Cifar10(root, train=True)
    print(dataset.paths[0], dataset.labels[0])
    data = DataLoader(dataset, batch_size=1)

    for epoch in range(1):
        print("Epoch {}".format(epoch))
        for step, (img, label) in enumerate(data):
            print("step {}: {}".format(step, img.size()))
