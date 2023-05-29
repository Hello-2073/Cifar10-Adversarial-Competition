import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from data import Cifar10Clean500
from model import JacobianModel
from test_bench import CnnBench


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--bench', default='cnn')
    parser.add_argument('--dataset', default='D:/Datasets/cifar10_clean_500')
    parser.add_argument('--iteration', default=5)
    parser.add_argument('--cuda', default=True)
    args = parser.parse_args()

    bench = CnnBench(10, args.bench)
    bench.load(os.path.join('./test_bench/checkpoints', args.bench, "epoch{}.pth".format(args.checkpoint)))
    classifier = bench.model()
    attacker = JacobianModel(classifier, 0.01)

    dataset = Cifar10Clean500(root=args.train_set, train=False)
    data = DataLoader(dataset, batch_size=1)

    for epoch in range(iteration):
        for step, (imgs, labels) in enumerate(data):
            if args.cuda:
                imgs = imgs.cuda()
                labels = labels.cuda()
            imgs = attacker(imgs)
            
    idx = 0
    while os.path.exists('D:/Datasets/cifar10_clean_500_{}'.format(idx)):
        idx += 1

    os.mkdir('D:/Datasets/cifar10_clean_500_{}'.format(idx))
    for i in range(imgs.shape[0]):
        img = imgs[i].cpu()
        img = img.squeeze(0)
        img = transforms.ToPILImage(img)
        img.save('D:/Datasets/cifar10_clean_500_{}/{}.png'.format(idx, i))
