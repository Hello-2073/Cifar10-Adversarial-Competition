# 数据集

Overview: 我们使用Cifar10数据集训练图像识别模型，然后使用Cifar10-Clean-500进行对抗样本的生成。

## 1 下载链接

Cifar10: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Cifar10-Clean-500(BUAA校园网): http://221.122.70.196/media/competition/data/10/cifar10_clean_500.zip

Cifar10官方完整二进制数据转化成图片后数据集: https://bhpan.buaa.edu.cn:443/link/86739175DA92D25A956B6D04F38B35DE


## 2 标签字典测试

经过图像对比与测试，课程组提供的`Cifar10-clean-500`与官方完整数据集`Cifar10`的类别与编码对应情况一致。

## 3 图像色彩空间测试

经图像色彩空间测试，课程组提供的`Cifar10-clean-500`与官方完整数据集`Cifar10`加载的图像在相同色彩空间中。