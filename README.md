# CS-WSCDNet

## Citation
CS-WSCDNet: Class Activation Mapping and Segment Anything Model-Based Framework for Weakly Supervised Change Detection [https://ieeexplore.ieee.org/document/10310006]

## Prerequisite
- Python >= 3.8, pytorch >= 1.8.0


## Usage (BCD)
### Step 1: Prepare the dataset.
- Download the BCD dataset from the [official website](https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html).
- Crop the downloaded images to a size of 256×256.

### Step 2: Train a CAM model.
- Use the `train_cam.py` and `train_recam.py` scripts in the `step` folder to train a change Class Activation Map (CAM) model.

### Step 3: Convert CAM to initial pseudo-labels.
- Use the `make_recam`,`cam_to_ir_label.py` and `cam_to_label.py` scripts in the `step` folder to convert the CAM outputs to initial pseudo-labels.
- DenseCRF is used for post-processing. Please note that the pseudo-labels might not be accurate if the class assignments are randomized. You can modify `cam_to_label.py` to assign the correct label as 255.

### Step 4: Apply SAM to refine pseudo-labels.
- Extract the image pairs with changes from the dataset and remove the unchanged image pairs.
- Use the `SAM_label.py` script in the `step` folder to refine the pseudo-labels using the SAM (Segment Anything Model).

### Step 5: Train the segmentation model.
- Train a segmentation model using the refined pseudo-labels and the image pairs with changes as supervised data.
- The segmentation model is typically based on U-Net architecture with attention mechanisms. You can choose from various publicly available models.

Please follow the steps provided above to use the CS-WSCDNet for building weakly supervised change detection on the BCD dataset.

## Acknowledgment
This code is borrowed from [IRN](https://github.com/jiwoon-ahn/irn), [ReCAM](https://github.com/zhaozhengChen/ReCAM) and [SAM](https://github.com/facebookresearch/segment-anything). Special thanks to Jiwoon Ahn, Zhaozheng Chen and Meta AI Research, FAIR for their contributions.

