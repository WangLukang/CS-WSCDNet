
import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
import imageio
from misc import imutils
from PIL import Image
import torch.nn.functional as F

IMG_FOLDER_NAME_B = "B1"
IMG_FOLDER_NAME_A = "A1"

cls_labels_dict = np.load('dataset/BCD/npy.npy', allow_pickle=True).item()

def decode_int_filename(int_filename):
    # s = str(int(int_filename))
    # return s[:4] + '_' + s[4:]
    filename_without_underscore = str(int_filename).replace("_", "")
    int_filename = int(filename_without_underscore)
    s = str(int_filename)
    return s


def load_image_label_list_from_npy(img_name_list):

    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])

def get_img_pathB(img_name, CAM_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(CAM_root, IMG_FOLDER_NAME_B, img_name + '.png')

def get_img_pathA(img_name, CAM_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(CAM_root, IMG_FOLDER_NAME_A, img_name + '.png')

def load_img_name_list(dataset_path):

    # img_name_list = np.loadtxt(dataset_path, dtype=np.int32)
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
    img_name_list = [int(name.replace('_', '')) for name in img_name_list]
    img_name_list = np.asarray(img_name_list, dtype=np.int32)

    return img_name_list


class TorchvisionNormalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]

        return proc_img


class VOC12ImageDataset(Dataset):

    def __init__(self, img_name_list_path, CAM_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None, to_torch=True):

        self.img_name_list = load_img_name_list(img_name_list_path)
        self.CAM_root = CAM_root

        self.resize_long = resize_long
        self.rescale = rescale
        self.crop_size = crop_size
        self.img_normal = img_normal
        self.hor_flip = hor_flip
        self.crop_method = crop_method
        self.to_torch = to_torch

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        imgA = np.asarray(imageio.imread(get_img_pathA(name_str, self.CAM_root)))
        imgB = np.asarray(imageio.imread(get_img_pathB(name_str, self.CAM_root)))

        if self.resize_long:
            imgA = imutils.random_resize_long(imgA, self.resize_long[0], self.resize_long[1])
            imgB = imutils.random_resize_long(imgB, self.resize_long[0], self.resize_long[1])

        if self.rescale:
            imgA = imutils.random_scale(imgA, scale_range=self.rescale, order=3)
            imgB = imutils.random_scale(imgB, scale_range=self.rescale, order=3)

        if self.img_normal:
            imgA = self.img_normal(imgA)
            imgB = self.img_normal(imgB)

        if self.hor_flip:
            imgA = imutils.random_lr_flip(imgA)
            imgB = imutils.random_lr_flip(imgB)

        if self.crop_size:
            if self.crop_method == "random":
                imgA = imutils.random_crop(imgA, self.crop_size, 0)
                imgB = imutils.random_crop(imgB, self.crop_size, 0)
            else:
                imgA = imutils.top_left_crop(imgA, self.crop_size, 0)
                imgB = imutils.top_left_crop(imgB, self.crop_size, 0)

        if self.to_torch:
            imgA = imutils.HWC_to_CHW(imgA)
            imgB = imutils.HWC_to_CHW(imgB)
            # arr_flipped = img[::-1].copy()  # 创建一个不具有负步幅的新数组
            # img = torch.from_numpy(arr_flipped)

        return {'name': name_str, 'imgA': imgA, 'imgB': imgB}

class VOC12ClassificationDataset(VOC12ImageDataset):

    def __init__(self, img_name_list_path, CAM_root,
                 resize_long=None, rescale=None, img_normal=TorchvisionNormalize(), hor_flip=False,
                 crop_size=None, crop_method=None):
        super().__init__(img_name_list_path, CAM_root,
                 resize_long, rescale, img_normal, hor_flip,
                 crop_size, crop_method)
        self.label_list = load_image_label_list_from_npy(self.img_name_list)

    def __getitem__(self, idx):
        out = super().__getitem__(idx)

        # label = torch.from_numpy(self.label_list[idx])
        # label = torch.nonzero(label)[:,0]
        # label = label[torch.randint(len(label),(1,))]
        # out['label'] = label

        out['label'] = torch.from_numpy(self.label_list[idx])

        return out


class VOC12ClassificationDatasetMSF(VOC12ClassificationDataset):

    def __init__(self, img_name_list_path, CAM_root, img_normal=TorchvisionNormalize(), scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, CAM_root, img_normal=img_normal)
        self.scales = scales

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        imgA = imageio.imread(get_img_pathA(name_str, self.CAM_root))
        imgB = imageio.imread(get_img_pathB(name_str, self.CAM_root))

        ms_img_listA = []
        ms_img_listB = []
        for s in self.scales:
            if s == 1:
                s_imgA = imgA
                s_imgB = imgB
            else:
                s_imgA = imutils.pil_rescale(imgA, s, order=3)
                s_imgB = imutils.pil_rescale(imgB, s, order=3)

            s_imgA = self.img_normal(s_imgA)
            s_imgB = self.img_normal(s_imgB)

            s_imgA = imutils.HWC_to_CHW(s_imgA)
            s_imgB = imutils.HWC_to_CHW(s_imgB)

            ms_img_listA.append(np.stack([s_imgA, np.flip(s_imgA, -1)], axis=0))
            ms_img_listB.append(np.stack([s_imgB, np.flip(s_imgB, -1)], axis=0))
        if len(self.scales) == 1:
            ms_img_listA = ms_img_listA[0]
            ms_img_listB = ms_img_listB[0]

        out = {"name": name_str, "imgA": ms_img_listA, "imgB": ms_img_listB, "size": (imgA.shape[0], imgA.shape[1]),
               "label": torch.from_numpy(self.label_list[idx])}
        return out


class VOC12_ours(Dataset):

    def __init__(self, img_name_list_path, CAM_root):

        self.ids = np.loadtxt(img_name_list_path, dtype=np.str)
        self.CAM_root = CAM_root
    def read_label(self, file, dtype=np.int32):
        f = Image.open(file)
        try:
            img = f.convert('P')
            img = np.array(img, dtype=dtype)
        finally:
            if hasattr(f, 'close'):
                f.close()

        if img.ndim == 2:
            return img
        elif img.shape[2] == 1:
            return img[:, :, 0]

    def get_label(self,i):
        label_path = os.path.join(self.CAM_root, 'SegmentationClassAug', self.ids[i] + '.png')
        label = self.read_label(label_path, dtype=np.int32)
        label[label == 255] = -1
        return label
    def get_label_by_name(self,i):
        label_path = os.path.join(self.CAM_root, 'SegmentationClassAug', i + '.png')
        label = self.read_label(label_path, dtype=np.int32)
        label[label == 255] = -1
        return label

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return idx

