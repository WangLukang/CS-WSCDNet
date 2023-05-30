import cv2
import numpy as np
import os

# 指定待处理的文件夹路径
folder_path = './result_10class/DenseCRF_label/'
save_path = './result_10class/mask/'
os.makedirs(save_path, exist_ok=True)

# 循环读取文件夹下所有图片并处理
for file_name in os.listdir(folder_path):
    if file_name.endswith('.png'):
        # 读取灰度影像
        img_gray = cv2.imread(os.path.join(folder_path, file_name), cv2.IMREAD_GRAYSCALE)



        img_gray[img_gray == 0] = 254
        img_gray[img_gray == 1] = 254
        img_gray[img_gray == 255] = 254
        img_gray[img_gray == 2] = 255
        img_gray[img_gray == 254] = 0
        # 保存影像
        save_file_name = f'{file_name}'
        save_path1 = os.path.join(save_path, save_file_name)
        cv2.imwrite(save_path1, img_gray)