import scipy.io
import numpy as np
import os
from PIL import Image
import shutil

########取出 imagelabels 文件的值############

imagelabels_path = './imagelabels.mat'
labels = scipy.io.loadmat(imagelabels_path)
labels = np.array(labels['labels'][0]) - 1

######## 取出 flower dataset: train test valid 数据id标识 ########
setid_path = './setid.mat'
setid = scipy.io.loadmat(setid_path)

validation = np.array(setid['valid'][0]) - 1
np.random.shuffle(validation)

train = np.array(setid['trnid'][0]) - 1
np.random.shuffle(train)

test = np.array(setid['tstid'][0]) - 1
np.random.shuffle(test)

######## flower data path 数据保存路径 ########
flower_dir = []

######## flower data dirs 生成保存数据的绝对路径和名称 ########
# 修正：使用正确的路径
jpg_dir = "./jpg"  # 图像文件所在的目录
for img in os.listdir(jpg_dir):
    flower_dir.append(os.path.join(jpg_dir, img))

######## flower data dirs sort 数据的绝对路径和名称排序 从小到大 ########
flower_dir.sort()

# 确保输出目录存在
os.makedirs("./train", exist_ok=True)
os.makedirs("./val", exist_ok=True)
os.makedirs("./test", exist_ok=True)

#####生成flower data train的分类数据 #######
des_folder_train = "./train"
for tid in train:
    try:
        ######## open image and get label ########
        img_path = flower_dir[tid]
        img = Image.open(img_path)

        ######## resize img #######
        img = img.resize((256, 256), Image.LANCZOS)  # 更新了抗锯齿方法
        lable = labels[tid]

        ######类别目录路径
        classes = "c" + str(lable)
        class_path = os.path.join(des_folder_train, classes)

        os.makedirs(class_path, exist_ok=True)

        base_path = os.path.basename(img_path)
        despath = os.path.join(class_path, base_path)
        img.save(despath)
    except Exception as e:
        print(f"处理训练集图像 {tid} 时出错: {e}")

#####生成flower data validation的分类数据 #######
des_folder_validation = "./val"
for tid in validation:
    try:
        ######## open image and get label ########
        img_path = flower_dir[tid]
        img = Image.open(img_path)

        img = img.resize((256, 256), Image.LANCZOS)
        lable = labels[tid]

        classes = "c" + str(lable)
        class_path = os.path.join(des_folder_validation, classes)

        os.makedirs(class_path, exist_ok=True)

        base_path = os.path.basename(img_path)
        despath = os.path.join(class_path, base_path)
        img.save(despath)
    except Exception as e:
        print(f"处理验证集图像 {tid} 时出错: {e}")

#####生成flower data test的分类数据 #######
des_folder_test = "./test"
for tid in test:
    try:
        ######## open image and get label ########
        img_path = flower_dir[tid]
        img = Image.open(img_path)

        img = img.resize((224, 224), Image.LANCZOS)
        lable = labels[tid]

        classes = "c" + str(lable)
        class_path = os.path.join(des_folder_test, classes)

        os.makedirs(class_path, exist_ok=True)

        base_path = os.path.basename(img_path)
        despath = os.path.join(class_path, base_path)
        img.save(despath)
    except Exception as e:
        print(f"处理测试集图像 {tid} 时出错: {e}")

print("数据处理完成！")