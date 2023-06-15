import os
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import Cifar10Clean500
from model import JacobianModel
from test_bench import CnnBench


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--bench', default='cnn')
    parser.add_argument('--dataset', default='D:/Datasets/cifar10_clean_500')
    parser.add_argument('--checkpoint', default=115)
    parser.add_argument('--iteration', default=10)
    parser.add_argument('--cuda', default=True)
    args = parser.parse_args()

    # Load a white-box model
    bench = CnnBench(10, args.bench)
    bench.load(os.path.join('./test_bench/checkpoints', args.bench, "epoch{}.pth".format(args.checkpoint)))
    classifier = bench.model()

    # Init an attack model
    attacker = JacobianModel(classifier, 0.01)

    idx = 0
    while os.path.exists('D:/Datasets/cifar10_clean_500/attack_{}'.format(idx)):
        idx += 1
    os.mkdir('D:/Datasets/cifar10_clean_500/attack_{}'.format(idx))
    
    # Load
    dataset = Cifar10Clean500(root=args.dataset)
    data = DataLoader(dataset, batch_size=1)

    toPIL = transforms.ToPILImage()

    for step, (img, label) in enumerate(data):
        if args.cuda:
            img = img.cuda()
            label = label.cuda()
        for epoch in range(args.iteration):
            img = attacker(img, label)
        img = img.cpu()
        img = img.squeeze(0)
        img = toPIL(img)
        img.save('D:/Datasets/cifar10_clean_500/attack_{}/{}.png'.format(idx, step))            
