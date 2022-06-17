from mmpose.apis import init_pose_model
import torch

def get_pose_net(cfg, is_train=True, device='cuda:0'):
    # cfg : not used.

    backbone_config="/workspace/learnable_triangulation/mmpose/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py"
#     backbone_checkpoint='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_384x288_dark-741844ba_20200812.pth'
    backbone_checkpoint='https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'
    
    
    # https://github.com/open-mmlab/mmpose/blob/master/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py
    # https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#higherhrnet-cvpr-2020
    # HRNET : ASSOCICATIVE EMBEDDING + HIGHERHRNET on COCO
    
    backbone_config = '/workspace/voxelpose_HICT/data/mmpose_config/configs/body/2d_kpt_sview_rgb_img/associative_embedding/coco/higherhrnet_w48_coco_512x512.py'
    backbone_checkpoint = '/workspace/voxelpose_HICT/data/higher_hrnet48_coco_512x512-60fedcbc_20200712.pth'
    
    
    pose_model = init_pose_model(backbone_config, backbone_checkpoint)#, device=args.device.lower()
    
    if is_train:
        pose_model.train()
    else:
        pose_model.eval()
        
        
#     pose_model.keypoint_head.final_layer = torch.nn.Conv2d(48, 15, kernel_size=(1, 1), stride=(1, 1)).to(device)
    
    # hrnet, associcative embedding, higherhrnet, coco
    pose_model.keypoint_head.final_layers[1] = torch.nn.Conv2d(48, 15, kernel_size=(1, 1), stride=(1, 1)).to(0)

    return pose_model
