# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from models import pose_hrnet # -yk

from models import pose_resnet
from models.cuboid_proposal_net_INFERENCE import CuboidProposalNet
from models.pose_regression_net_INFERENCE import PoseRegressionNet
from core.loss import PerJointMSELoss
from core.loss import PerJointL1Loss

import time
import numpy as np


class MultiPersonPoseNet(nn.Module):
    def __init__(self, backbone, cfg):
        super(MultiPersonPoseNet, self).__init__()
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_joints = cfg.NETWORK.NUM_JOINTS

        self.backbone = backbone
        self.root_net = CuboidProposalNet(cfg)
        self.pose_net = PoseRegressionNet(cfg)

        self.USE_GT = cfg.NETWORK.USE_GT
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET
        
        self.model_name = cfg.BACKBONE_MODEL
        
        self.mean_heatmap_time = []
        self.mean_rootnet_time = []
        self.mean_v2vnet_time = []
        self.mean_count = 0

    def forward(self, views=None, meta=None, targets_2d=None, weights_2d=None, targets_3d=None, input_heatmaps=None, tmp_dataset=None):
        
        start_time = time.time()

        views_tensor = torch.stack(views, dim=1) # torch.Size([batch, 6, 3, 512, 960])
        _batch, _view, _ch, _h, _w = views_tensor.shape
        views_tensor = views_tensor.reshape(_batch*_view, _ch, _h, _w)
        heatmaps = self.backbone.backbone(views_tensor)
        heatmaps = self.backbone.keypoint_head(heatmaps)
        heatmaps = heatmaps[1] # [1]: torch.Size([1, 15, 256, 480])
        _, _hc, _hh, _hw = heatmaps.shape
        heatmaps = heatmaps.reshape(_batch, _view, _hc, _hh, _hw)
        
        all_heatmaps = []
        for _v in range(_view):
            all_heatmaps.append(heatmaps[:,_v,:,:,:])

#         all_heatmaps = []
#         for view in views:
#             if self.model_name == 'pose_hrnet':
#                 heatmaps = self.backbone.backbone(view)
#                 heatmaps = self.backbone.keypoint_head(heatmaps)
#                 heatmaps = heatmaps[1] # [0]: torch.Size([1, 34, 128, 240]), [1]: torch.Size([1, 15, 256, 480])

#             else:
#                 heatmaps = self.backbone(view)

#             all_heatmaps.append(heatmaps)

        ########################################################################################
        heatmap_time =  time.time() - start_time
        self.mean_heatmap_time.append(heatmap_time)    
        if self.mean_count%10 == 0:
            print('mean heatmap time is : {}'.format(np.mean(self.mean_heatmap_time)))
        start_time_2 = time.time()
        ########################################################################################

        # all_heatmaps = targets_2d
        device = all_heatmaps[0].device
        batch_size = all_heatmaps[0].shape[0]
        root_cubes, grid_centers = self.root_net(all_heatmaps, meta)
              
        ########################################################################################            
        rootnet_time = time.time() - start_time_2
        self.mean_rootnet_time.append(rootnet_time)
        if self.mean_count%10 == 0:
            print('mean rootnet time is : {}'.format(np.mean(self.mean_rootnet_time)))
        start_time_3 = time.time()
        ########################################################################################

        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)  # matched gt
        
        
        
        for n in range(self.num_cand):
            index = (pred[:, n, 0, 3] >= 0)
            if torch.sum(index) > 0:
                single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n])
                pred[:, n, :, 0:3] = single_pose.detach()
                del single_pose
        
        ########################################################################################            
        v2vnet_time = time.time() - start_time_3
        self.mean_v2vnet_time.append(v2vnet_time)
        if self.mean_count%10 == 0:
            print('mean rootnet time is : {}'.format(np.mean(self.mean_v2vnet_time)))
        self.mean_count += 1
        ########################################################################################
        
        
        return pred, all_heatmaps, grid_centers



def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + '.get_pose_net')(cfg, is_train=is_train)
    else:
        backbone = None
    model = MultiPersonPoseNet(backbone, cfg)
    return model
