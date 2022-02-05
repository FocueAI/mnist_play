# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 10:48
# @Authon  : Alex
# @File    : 拼凑手写数字集.py
# @Software: PyCharm
# @Describe: 
import os
import shutil
from tqdm import tqdm
import numpy as np
import cv2,glob
import random
import numpy as np

train_dataset_path = r'./train_images/0/*.jpg'
char_0_list = glob.glob(train_dataset_path)

train_dataset_path = r'./train_images/1/*.jpg'
char_1_list = glob.glob(train_dataset_path)

train_dataset_path = r'./train_images/2/*.jpg'
char_2_list = glob.glob(train_dataset_path)

train_dataset_path = r'./train_images/3/*.jpg'
char_3_list = glob.glob(train_dataset_path)

train_dataset_path = r'./train_images/4/*.jpg'
char_4_list = glob.glob(train_dataset_path)

train_dataset_path = r'./train_images/5/*.jpg'
char_5_list = glob.glob(train_dataset_path)

train_dataset_path = r'./train_images/6/*.jpg'
char_6_list = glob.glob(train_dataset_path)

train_dataset_path = r'./train_images/7/*.jpg'
char_7_list = glob.glob(train_dataset_path)

train_dataset_path = r'./train_images/8/*.jpg'
char_8_list = glob.glob(train_dataset_path)

train_dataset_path = r'./train_images/9/*.jpg'
char_9_list = glob.glob(train_dataset_path)

save_path = r'./save'
if os.path.exists(save_path): shutil.rmtree(save_path)
os.mkdir(save_path)


# 切除空白的边框
def corp_margin(img):
    img2 = img.sum(axis=2)
    (row, col) = img2.shape
    row_top = 0
    raw_down = 0
    col_top = 0
    col_down = 0
    baise_sum = 254*3
    # for r in range(0, row):
    #     if img2.sum(axis=1)[r] < baise_sum * col:
    #         row_top = r
    #         break
    #
    # for r in range(row - 1, 0, -1):
    #     if img2.sum(axis=1)[r] < baise_sum * col:
    #         raw_down = r
    #         break

    for c in range(0, col):
        if img2.sum(axis=0)[c] < baise_sum * row:
            col_top = c
            break

    for c in range(col - 1, 0, -1):
        if img2.sum(axis=0)[c] < baise_sum * row:
            col_down = c
            break

    # new_img = img[0+5:col-5, col_top:col_down + 1, 0:3]
    # new_img = img[0 + 5:col - 5, col_top:col_down + 1, 0:3]
    # new_img = img[row_top:raw_down + 1, col_top:col_down + 1, 0:3]
    new_img = img[:, col_top:col_down + 1, :]
    return new_img


# 定义造数据的数量
txt_path = './corpus/num_samples.txt'
with open(txt_path,'r',encoding='utf-8') as reader:
    contents = reader.readlines()
    for pic_no, content in tqdm(enumerate(contents)):
        content =  content.strip()
        content_leng = len(content)
        img_zero = np.ones((28, 28*int(content_leng), 3), dtype='uint8') * 255
        # print('img_zero.shape:',img_zero.shape)
        acc_width = 0
        for index, per_str in enumerate(content):
            if '0' in per_str:
                img_path = char_0_list[random.randint(0,len(char_0_list)-1)]
                img = cv2.imread(img_path)
            elif '1' in per_str:
                img_path = char_1_list[random.randint(0,len(char_1_list)-1)]
                img = cv2.imread(img_path)
            elif '2' in per_str:
                img_path = char_2_list[random.randint(0,len(char_2_list)-1)]
                img = cv2.imread(img_path)
            elif '3' in per_str:
                img_path = char_3_list[random.randint(0,len(char_3_list)-1)]
                img = cv2.imread(img_path)
            elif '4' in per_str:
                img_path = char_4_list[random.randint(0,len(char_4_list)-1)]
                img = cv2.imread(img_path)
            elif '5' in per_str:
                img_path = char_5_list[random.randint(0,len(char_5_list)-1)]
                img = cv2.imread(img_path)
            elif '6' in per_str:
                img_path = char_6_list[random.randint(0,len(char_6_list)-1)]
                img = cv2.imread(img_path)
            elif '7' in per_str:
                img_path = char_7_list[random.randint(0,len(char_7_list)-1)]
                img = cv2.imread(img_path)
            elif '8' in per_str:
                img_path = char_8_list[random.randint(0,len(char_8_list)-1)]
                img = cv2.imread(img_path)
            elif '9' in per_str:
                img_path = char_9_list[random.randint(0,len(char_9_list)-1)]
                img = cv2.imread(img_path)
            else:
                continue
            # print('img_shape:',img.shape)
            new_img = corp_margin(img)
            new_img_h,new_img_w,_ = new_img.shape

            img_zero[:, acc_width:acc_width+new_img_w ,:] = new_img
            acc_width += new_img_w
        img_path = os.path.join(save_path,str(pic_no)+'.jpg')
        label_path = os.path.join(save_path,str(pic_no)+'.txt')
        img_zero = img_zero[:,0:acc_width+5,:]
        cv2.imwrite(img_path,img_zero)
        with open(label_path,'w',encoding='utf-8') as writer:
            writer.write(content)




