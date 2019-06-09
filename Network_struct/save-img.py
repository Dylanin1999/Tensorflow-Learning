import cv2 as cv
import os

def save_img():
    for i in range(10):
        img_dir = './Img/' + str(i) + '/'
        img_names = os.listdir(img_dir)
        count = 0
        for img in img_names:
            save_name = './test/' + img
            img = img_dir + img
            if count>=500 and count<600:
                #break
                pic = cv.imread(img)
                cv.imwrite(save_name, pic)
            count += 1

    print('finished')

save_img()