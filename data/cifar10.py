import os
import pickle
import numpy as np
import imageio


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
        imageio.imwrite(pic_name, img)


if __name__ == '__main__':
    root = '..\\dataset\\cifar-10-python'
    for i in range(5):
        binary2img(root, f'data_batch_%d' % (i + 1), 'train')
    binary2img(root, f'test_batch', 'test')
