import os
import cv2
import numpy as np
import sklearn.cluster as skc
from keras.preprocessing import image
from sklearn.cluster import KMeans
import shutil
import glob
import random
import time

# 待聚类图像路径
imgpath = './tot_r_i_g_h_t/*.jpg'             # "./tf_v3_data/pse_train/JPEGImages/"
#聚类结果保存路径
outpath =  './tot_r_i_g_h_t_dealed_res'      # "./tf_v3_data/pse_train/results/"
if os.path.exists(outpath): shutil.rmtree(outpath)
os.mkdir(outpath)

img_size = (32,400)
#img_size = (400,32)
# 初始化样本数据
def load_batch_data(img_full_list):
    images = []
    image_names =[]
    for f in img_full_list:
            img_path = f
            img = image.load_img(img_path, target_size = img_size)
            img_array = image.img_to_array(img)
            img_array = img_array.reshape(img_size[0]*img_size[1]*3)
            images.append(img_array)
            image_names.append(f)
    data = np.array(images)
    return data,image_names

def load_sig_data(img_path):
    images = []
    image_names =[]
    image_name = os.path.basename(img_path)
    if '.jpg' in img_path:
        img = image.load_img(img_path, target_size = img_size)
        img_array = image.img_to_array(img)
        img_array = img_array.reshape(img_size[0]*img_size[1]*3)
        images.append(img_array)        # 图片数据
        image_names.append(image_name)  # 图片对应的标签数据
    data = np.array(images)
    return data,image_names

##################一边读数据一边聚类#######################
# 加载训练数据
print('step1: begin load data......')
tot_img_list = glob.glob(imgpath)
tot_img_num = len(tot_img_list)
random.shuffle(tot_img_list)
batch_size = 3000
ITER_COUNT = tot_img_num//batch_size
print('ITER_COUNT:',ITER_COUNT)
kmeans_model = KMeans(n_clusters=5, random_state=1, n_jobs=-1)
for i in range(ITER_COUNT):
    print("iter:",i)
    batch_train_datas = tot_img_list[i*batch_size:min((i+1)*batch_size,tot_img_num-1)]
    t0 = time.time()
    print('step2: begin train data......')
    X,Y = load_batch_data(batch_train_datas) # 加载训练集
    kmeans_model.fit(X)
    t1 = time.time()
    cast_time = t1 - t0
    print('cast-time:',(t1-t0))


print('step3: begin predict data......')
img_lists = glob.glob(imgpath)
for detail_image_path in img_lists:
    file_name = os.path.basename(detail_image_path)
    X,Y = load_sig_data(detail_image_path)
    res = kmeans_model.predict(X)
    print('res:',res)

    if not os.path.exists(os.path.join(outpath,str(res[0]))):
        os.mkdir(os.path.join(outpath,str(res[0])))
    dst_image_path = os.path.join(outpath,str(res[0]),file_name)
    shutil.copy(detail_image_path,dst_image_path)
    print('copy sucess..%s'%(dst_image_path))


###############################################################
# #开始聚类
# X,Y = load_data(imgpath)
# db = KMeans(n_clusters=2, random_state=1).fit(X)
# KMeans(n_clusters=2, random_state=1).fit_predict()
# KMeans().predict()
# labels = db.labels_
# #将图像按照聚类结果存入对应文件夹中
# for img_name,label in zip(Y,labels):
#     if not os.path.exists(os.path.join(outpath,str(label))):
#         os.mkdir(os.path.join(outpath,str(label)))
#     image = cv2.imread(imgpath+img_name)
#     cv2.imwrite(outpath+str(label)+"/"+img_name,image)
#################################################################