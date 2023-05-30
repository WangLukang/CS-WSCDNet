import os
import shutil

# 源文件夹和目标文件夹路径
# source_folder_path = "./Dataset/landslide/A1"
# target_folder_path = "./Dataset/landslide/B"
# save_path = "./Dataset/landslide/A2"
source_folder_path = "./result_BCD_CAMonly/mask"
target_folder_path = "./result_BCD_CAMonly/mask_train_seg"
save_path = "./result_BCD_CAMonly/mask_train_onlyCAM"
os.makedirs(save_path, exist_ok=True)
# 获取源文件夹下的所有文件名
source_files = os.listdir(source_folder_path)

# 遍历源文件夹下的所有文件
for source_file in source_files:
    # 获取源文件的路径
    source_file_path = os.path.join(source_folder_path, source_file)

    # 如果源文件是文件夹，则跳过
    if os.path.isdir(source_file_path):
        continue

    # 获取源文件的文件名（不带后缀）
    source_file_name = os.path.splitext(source_file)[0]

    # 遍历目标文件夹下的所有文件
    for target_file in os.listdir(target_folder_path):
        # 获取目标文件的路径
        target_file_path = os.path.join(target_folder_path, target_file)

        # 如果目标文件是文件夹，则跳过
        if os.path.isdir(target_file_path):
            continue

        # 获取目标文件的文件名（不带后缀）
        target_file_name = os.path.splitext(target_file)[0]

        # 如果源文件和目标文件的文件名相同，则将源文件复制到目标文件夹中
        if source_file_name == target_file_name:
            shutil.copy2(source_file_path, save_path)
            # print(f"File {source_file} has been copied to {save_path}.")
            break