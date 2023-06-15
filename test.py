import os
import argparse
from torch.utils.data import DataLoader

from data import Cifar10, Cifar10Clean500
from test_bench import ResnetBench, CnnBench


if __name__ == '__main__':
    root = 'D:\\大三下课程\\人工智能安全导论\\Cifar10-Adversarial-Competition\\'
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', default='cnn')
    parser.add_argument('--train_set', default='D:/Datasets/cifar-10-python')
    parser.add_argument('--test_set', default='D:/Datasets/cifar10_clean_500')
    # parser.add_argument('--attack_id', default=101)
    parser.add_argument('--pretrain', default=0)
    parser.add_argument('--checkpoint', default=115)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--iteration', default=50)
    parser.add_argument('--cuda', default=True)
    args = parser.parse_args()

    bench = CnnBench(n_cls=10, name=args.bench, cuda=args.cuda)

    if not os.path.exists(os.path.join('./test_bench/checkpoints', args.bench, "epoch{}.pth".format(args.checkpoint))):
        if args.pretrain > 0:
            bench.load(os.path.join('./test_bench/checkpoints', args.bench, "epoch{}.pth".format(args.pretrain))) 
        dataset = Cifar10(root=args.train_set, train=True)
        print("Trainset size is {}.".format(len(dataset)))
        data = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        bench.train(data, save_root=root+'test_bench\\checkpoints', iteration=args.iteration)
    
    bench.load(os.path.join('./test_bench/checkpoints', args.bench, "epoch{}.pth".format(args.checkpoint)))

    dataset = Cifar10Clean500(root=args.test_set, attack_id=0)
    data = DataLoader(dataset, batch_size=1)
    preds, confusion_matrix = bench.test(data)
    print("accuracy=", confusion_matrix)