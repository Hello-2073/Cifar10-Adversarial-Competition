import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--dataset', default='D:/Datasets/coco_tampered/')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--iteration', default=5)
    args = parser.parse_args()