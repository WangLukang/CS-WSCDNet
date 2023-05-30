import os
import shutil

path = './dataset/label'
path2 = './Dataset/label1'

os.makedirs(path2, exist_ok=True)

# 获取文件夹中的所有文件
files = os.listdir(path)

# 遍历文件夹中的所有文件
for file in files:
    # 如果文件名长度大于4
  if file.endswith(".png"):
        image_name = file.split(".")[0]
        image_name = image_name[4:]
        index = int(image_name)
        shutil.copy(os.path.join(path, file), os.path.join(path2, str(index) + '.png'))
