import os
import shutil

path = './Dataset/landslide/A'
path2 = './Dataset/landslide/A1'

path3 = './Dataset/landslide'

# 获取文件夹中的所有文件
files = os.listdir(path)

# 遍历文件夹中的所有文件
for file in files:
    # 如果文件名长度大于4
    if len(file) > 4:
        # 重命名文件，去掉前四个字符
        image_name = file.split(".")[0]
        image_name = image_name[4:]
        index = int(image_name)
        shutil.copy(os.path.join(path, file), os.path.join(path2, str(index) + '.jpg'))
        shutil.copy(os.path.join(path3, 'B', str(index) + '.jpg'), os.path.join(path3, 'A2', str(index) + '.jpg'))