import os
import numpy as np
from PIL import Image

# 设置图片路径和npy保存路径
image_dir = "./dataset/BCD/label/"
npy_file = "./dataset/BCD/npy.npy"


image_files = os.listdir(image_dir)

index_dict = {}

for image_file in image_files:
    if image_file.endswith(".tif"):
        image_name = image_file.split(".")[0]
        image_name = image_name[4:]
        index = int(image_name)
        index_dict[index] = np.zeros(10)  #n_class set to 10 to prevent gradient explosion and overfitting,there are actually only two classes.
        tmp = Image.open(image_dir + image_file)
        # print(tmp)
        # exit(-1)
        if np.sum(tmp) == 0:
            index_dict[index][0] = 1
        else:
            index_dict[index][1] = 1
# print(index_dict)

np.save(npy_file, index_dict)


# 创建并打开一个新的文本文件
with open('./Dataset/BCD/imagename.txt', 'w') as file:
    # 遍历所有图片文件名，并将文件名写入文本文件
    for image_file in image_files:
        image_name = image_file.split(".")[0]
        image_name = image_name[4:]
        index = int(image_name)
        file.write(str(index) + '\n')
