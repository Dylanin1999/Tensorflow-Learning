import cv2 as cv
import os


def img_resize():
    for label in range(10):
        path = './Img/'
        path = path + str(label) + '/'
        filename = os.listdir(path)
        print('当前文件夹为：', path)
        num = len(filename)
        count = 0
        for img_name in filename:
            img_name = path + img_name
            print('当前文件为：', img_name)
            Image = cv.imread(img_name)
            Image = cv.resize(Image, (224, 224))
            cv.imwrite(img_name, Image)
            print('当前进度：', count, '/', num+1)
            count += 1



