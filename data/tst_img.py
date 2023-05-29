import os
import pickle
import numpy as np
import imageio
import cv2
from PIL import Image


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
        label = str(img_dict[b'labels'][i])

        img_0 = img.transpose(1, 2, 0)
        save_path = os.path.join(root, 'cifar-10-output-test', save_cls, str(classification.get(label)), '1_2_0')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pic_name = save_path + f'\\%d.png' % i
        # imageio.imwrite(pic_name, img)
        # cv2.imwrite(pic_name, img)
        img_0 = Image.fromarray(img_0, 'RGB')
        img_0.save(pic_name)

        img_1 = img.transpose(0, 2, 1)
        save_path = os.path.join(root, 'cifar-10-output-test', save_cls, str(classification.get(label)), '0_2_1')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pic_name = save_path + f'\\%d.png' % i
        # imageio.imwrite(pic_name, img_1)
        # cv2.imwrite(pic_name, img_1)
        img_1 = Image.fromarray(img_1, 'RGB')
        img_1.save(pic_name)

        img_2 = img.transpose(1, 0, 2)
        save_path = os.path.join(root, 'cifar-10-output-test', save_cls, str(classification.get(label)), '1_0_2')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pic_name = save_path + f'\\%d.png' % i
        # imageio.imwrite(pic_name, img_2)
        # cv2.imwrite(pic_name, img_2)
        img_2 = Image.fromarray(img_2, 'RGB')
        img_2.save(pic_name)

        img_3 = img.transpose(0, 1, 2)
        save_path = os.path.join(root, 'cifar-10-output-test', save_cls, str(classification.get(label)), '0_1_2')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pic_name = save_path + f'\\%d.png' % i
        # imageio.imwrite(pic_name, img_3)
        # cv2.imwrite(pic_name, img_3)
        img_3 = Image.fromarray(img_3, 'RGB')
        img_3.save(pic_name)

        img_4 = img.transpose(2, 0, 1)
        save_path = os.path.join(root, 'cifar-10-output-test', save_cls, str(classification.get(label)), '2_0_1')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pic_name = save_path + f'\\%d.png' % i
        # imageio.imwrite(pic_name, img_4)
        # cv2.imwrite(pic_name, img_4)
        img_4 = Image.fromarray(img_4, 'RGB')
        img_4.save(pic_name)

        img_5 = img.transpose(2, 1, 0)
        save_path = os.path.join(root, 'cifar-10-output-test', save_cls, str(classification.get(label)), '2_1_0')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pic_name = save_path + f'\\%d.png' % i
        # imageio.imwrite(pic_name, img_5)
        # cv2.imwrite(pic_name, img_5)
        img_5 = Image.fromarray(img_5, 'RGB')
        img_5.save(pic_name)


if __name__ == '__main__':
    root = '..\\dataset\\cifar-10-python'
    # for i in range(5):
    #     binary2img(root, f'data_batch_%d' % (i + 1), 'train')
    binary2img(root, f'test_batch', 'test')
