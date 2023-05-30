import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
import numpy as np
from skimage import io
from PIL import Image
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = "TRUE"

def logit(x):
    a = torch.tensor(x)
    return torch.special.logit(a, eps=1e-6).numpy()

def inv_logit(x):
    return 0.5*(1. + np.sign(x)*(2./(1. + np.exp(-np.abs(x))) - 1.))

def get_target_bboxes(gray_img):
    coords = np.column_stack(np.where(gray_img == 255))
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x+w, y+h

def get_all_target_bboxes(gray_img):
    ret, thresh = cv2.threshold(gray_img, 2, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 20 or h < 20:
            continue
        target_boxes.append((x, y, x+w, y+h))
    return target_boxes

def SAM(args):
    folderA_path = args.SAM_A
    folderB_path = args.SAM_B
    mask_path = args.mask
    save_path = args.SAMlabel
    os.makedirs(save_path, exist_ok=True)

    sam_checkpoint = args.SAM_weight
    device = "cuda"
    model_type = "vit_h"
    sys.path.append("..")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    samPredictor = SamPredictor(sam)

    for file_name in os.listdir(folderA_path):
        if file_name.endswith('.png'):
            mask = cv2.imread(os.path.join(mask_path, file_name), 0)
            imageA = cv2.imread(os.path.join(folderA_path, file_name))
            imageB = cv2.imread(os.path.join(folderB_path, file_name))
            mask = cv2.resize(mask, (256,256))

            input_box = get_all_target_bboxes(mask)
            input_box = np.array(input_box)
            if len(input_box) == 0:
                continue

            image = imageB - imageA    
            samPredictor.set_image(image)

            input_box = torch.from_numpy(input_box)
            all_input_boxes = samPredictor.transform.apply_boxes_torch(input_box, imageA.shape[:2])

            masks, scores, logits = samPredictor.predict_torch(
                point_coords = None,
                point_labels = None,
                boxes = all_input_boxes.to(device=device),
                multimask_output = False,
            )

            mask_ = np.zeros((256,256))
            for mask in masks:
                mask = mask.cpu().numpy()
            # mask = np.array(mask)
            mask = mask.astype(np.uint8)[0,:,:]
            mask[mask == 1] = 255
            mask_ = mask_ + mask

        # 保存影像
        save_file_name = f'{file_name}'
        save_path1 = os.path.join(save_path, save_file_name)
        cv2.imwrite(save_path1, mask_)

if __name__ == '__main__':
    SAM(None)
