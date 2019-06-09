import cv2 as cv
import _pickle as pickle
import os
import numpy as np



# 将图片转换为cifar-10格式
def img2cifar(dirpath):
    data_train = []
    labels_train = []

    # 因为cifar-10的格式本质上是一个字典的形式，所以这里我们使用字典
    dict_train = {'data': [], 'labels': []}

    imgs = os.listdir(dirpath)
    num = len(imgs)
    count = 0

    # 对每张图片进行处理
    for img_name in imgs:
        img_name = dirpath + "/" + img_name
        label = img_name.split('_')[0].split('/')[-1]

        # 读入图片
        img = cv.imread(img_name)
        img = img.flatten()

        data_train.append(img)
        labels_train.append(int(label))

        # 打印处理进度
        print('当前标签：', label)
        print('当前进度：', count, '/', num)
        count += 1

    data_train = np.array(data_train)

    dict_train['data'] = data_train
    dict_train['labels'] = labels_train

  #  print('data: ', data_train)
    print('labels:', len(labels_train))
    print(dict_train)
    # 将dictionary数据写入到二进制文件中去
    with open('data_test_1', 'wb') as train:
        pickle.dump(dict_train, train)
    print('dict:', dict_train['data'].shape)


def main(filename):
        print("filename: ", filename)
        img2cifar(filename)

main('./test')

def load(filename):
    with open(filename, 'rb') as f:
        cifar = pickle.load(f, encoding='iso-8859-1')
        data = cifar['data']
        label = cifar['labels']
        print('cifar:  ', cifar)
        print('data: ', data)
        print('label: ', label)
        print('data_shape: ', data.shape)
        print('label: ', len(label))

#load('./data_train_1')