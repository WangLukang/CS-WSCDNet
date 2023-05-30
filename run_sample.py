import argparse
import os
import numpy as np
import os.path as osp

from misc import pyutils

if __name__ == '__main__':
   

    parser = argparse.ArgumentParser()

    # Environment
    # parser.add_argument("--num_workers", default=os.cpu_count()//2, type=int)
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--CAM_root", default='dataset/BCD', type=str)
    parser.add_argument("--SAM_A", default="./dataset/BCDDDDD/BCD_removeblank_split/train_seg/A/", type=str)
    parser.add_argument("--SAM_B", default="./dataset/BCDDDDD/BCD_removeblank_split/train_seg/B/", type=str, 
                        help = "Remove the unchanged pixel pairs in the predicted mask, and only use SAM for the changed pixel pairs")
    parser.add_argument("--SAM_weight", default="./pth/sam_vit_h_4b8939.pth", type=str, 
                        help = "sam_vit_h_4b8939.pth")

    # Dataset
    parser.add_argument("--train_list", default="dataset/BCD/imagename.txt", type=str)
    parser.add_argument("--val_list", default="dataset/BCD/imagename.txt", type=str)

    # Class Activation Map
    parser.add_argument("--cam_network", default="net.resnet50_cam", type=str)
    parser.add_argument("--feature_dim", default=2048, type=int)
    parser.add_argument("--cam_crop_size", default=224, type=int)
    parser.add_argument("--cam_batch_size", default=16, type=int)
    parser.add_argument("--cam_num_epoches", default=5, type=int)
    parser.add_argument("--cam_learning_rate", default=0.05, type=float)
    parser.add_argument("--cam_weight_decay", default=1e-4, type=float)
    parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5, 2.0),
                        help="Multi-scale inferences")
    # ReCAM
    parser.add_argument("--recam_num_epoches", default=7, type=int)
    parser.add_argument("--recam_learning_rate", default=0.0005, type=float)
    parser.add_argument("--recam_loss_weight", default=0.7, type=float)

    # Mining Inter-pixel Relations
    parser.add_argument("--conf_fg_thres", default=0.35, type=float)
    parser.add_argument("--conf_bg_thres", default=0.1, type=float)

 

    # Output Path
    parser.add_argument("--work_space", default="result", type=str) # set your path
    parser.add_argument("--log_name", default="sample_train", type=str)
    parser.add_argument("--cam_weights_name", default="res50_cam.pth", type=str)
    parser.add_argument("--cam_out_dir", default="cam", type=str)
    parser.add_argument("--ir_label_out_dir", default="DenseCRF_label", type=str)
    parser.add_argument("--recam_weight_dir", default="recam_weight", type=str)
    parser.add_argument("--mask", default="mask", type=str)
    parser.add_argument("--SAMlabel", default="SAMlabel", type=str)

    # Step
    parser.add_argument("--train_cam_pass", default=True)
    parser.add_argument("--train_recam_pass", default=True)
    parser.add_argument("--make_recam_pass", default=True)
    parser.add_argument("--cam_to_ir_label_pass", default=True) 
    parser.add_argument("--cam_to_mask_pass", default=True)
    parser.add_argument("--SAM_label_pass", default=True)  
 
    args = parser.parse_args()
    args.log_name = osp.join(args.work_space,args.log_name)
    args.cam_weights_name = osp.join(args.work_space,args.cam_weights_name)
    args.cam_out_dir = osp.join(args.work_space,args.cam_out_dir)
    args.ir_label_out_dir = osp.join(args.work_space,args.ir_label_out_dir)
    args.recam_weight_dir = osp.join(args.work_space,args.recam_weight_dir)
    args.mask = osp.join(args.work_space,args.mask)
    args.SAMlabel = osp.join(args.work_space,args.SAMlabel)

    os.makedirs(args.work_space, exist_ok=True)
    os.makedirs(args.cam_out_dir, exist_ok=True)
    os.makedirs(args.ir_label_out_dir, exist_ok=True)
    os.makedirs(args.recam_weight_dir, exist_ok=True)
    os.makedirs(args.mask, exist_ok=True)
    os.makedirs(args.SAMlabel, exist_ok=True)
    pyutils.Logger(args.log_name + '.log')
    print(vars(args))


    if args.train_cam_pass is True:
        import step.train_cam

        timer = pyutils.Timer('step.train_cam:')
        step.train_cam.run(args)
    
    
    if args.train_recam_pass is True:
        import step.train_recam

        timer = pyutils.Timer('step.train_recam:')
        step.train_recam.run(args)
    
    if args.make_recam_pass is True:
        import step.make_recam

        timer = pyutils.Timer('step.make_recam:')
        step.make_recam.run(args)
        
    if args.cam_to_ir_label_pass is True:
        import step.cam_to_ir_label

        timer = pyutils.Timer('step.cam_to_ir_label:')
        step.cam_to_ir_label.run(args)


    if args.cam_to_mask_pass is True:
        import step.cam_to_label

        timer = pyutils.Timer('step.cam_to_label:')
        step.cam_to_label.cam_to_label(args)

    if args.SAM_label_pass is True:
        import step.SAM_label

        timer = pyutils.Timer('step.cam_to_label:')
        step.SAM_label.SAM(args)
  