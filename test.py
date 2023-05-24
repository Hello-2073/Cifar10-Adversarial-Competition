import os
import argparse
from torch.utils.data import DataLoader

from data import Cifar10, Cifar10Clean500
from test_bench import ResnetBench


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', default='resnet18')
    parser.add_argument('--pretrained', default=True)
    parser.add_argument('--train_set', default='D:/Datasets/cifar-10-python/cifar-10-batches-py')
    parser.add_argument('--test_set', default='D:/Datasets/cifar10_clean_500')
    parser.add_argument('--checkpoint', default=50)
    parser.add_argument('--batch_size', default=24)
    parser.add_argument('--iteration', default=5)
    parser.add_argument('--cuda', default=True)
    args = parser.parse_args()

    bench = ResnetBench(n_cls=10, name=args.bench, cuda=args.cuda)

    if not args.pretrained:
        dataset = Cifar10(root=args.train_set, train=True)
        data = DataLoader(dataset, batch_size=args.batch_size)
        bench.train(data, save_root='./test_bench/checkpoints')

    bench.load(os.path.join('./test_bench/checkpoints', args.bench, "epoch{}.pth".format(args.checkpoint)))

    dataset = Cifar10(root=args.train_set, train=False)
    data = DataLoader(dataset, batch_size=1)
    preds, confusion_matrix = bench.test(data)
    print(confusion_matrix)
    