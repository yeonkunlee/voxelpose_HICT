# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from utils.transforms import affine_transform, get_scale, get_affine_transform
import torch

def make_augmented_inputs(_meta, _transform, random_range=[0.2, 1.0]):
    _inputs_new = []
    _inputs_vis = []
    
    _num_cam = len(_meta)
    _batch_size = len(_meta[0]['image'])
    
    for _cam_i in range(_num_cam):
        input_batch_holder = []
        for _batch_i in range(_batch_size):
    #         print(_meta[_cam_i]['image'][_batch_i])
            data_numpy = cv2.imread(_meta[_cam_i]['image'][_batch_i], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
            height, width, _ = data_numpy.shape

            # CAHNGE INPUT FOCAL
            fx = _meta[_cam_i]['camera']['fx'][_batch_i].detach().cpu().numpy()
            fy = _meta[_cam_i]['camera']['fy'][_batch_i].detach().cpu().numpy()
            cx = _meta[_cam_i]['camera']['cx'][_batch_i].detach().cpu().numpy()
            cy = _meta[_cam_i]['camera']['cy'][_batch_i].detach().cpu().numpy()

            f_random_ratio = np.random.rand()*(random_range[1]-random_range[0]) + random_range[0] # (0.2 ~ 1.0)
    #         target_fx = 533.3333
    #         target_fy = 533.3333
            target_fx = f_random_ratio * fx
            target_fy = f_random_ratio * fy

            # affine = np.array([[target_fx/fx,0.,cx/2.-960/2.],
            #                    [0.,target_fy/fy,cy/2.-512/2.]])

            affine = np.array([[target_fx/fx,0.,(cx - cx * (target_fx/fx))],
                               [0.,target_fy/fy,(cy - cy * (target_fy/fy))]])

            data_numpy = cv2.warpAffine(
                        data_numpy,
                        affine, (0, 0),
                        flags=cv2.INTER_LINEAR)

    #         _meta[_cam_i]['camera']['fx'] *= target_fx/fx
    #         _meta[_cam_i]['camera']['fy'] *= target_fy/fy
            _meta[_cam_i]['camera']['fx'][_batch_i] *= target_fx/fx
            _meta[_cam_i]['camera']['fy'][_batch_i] *= target_fy/fy

            c = np.array([width / 2.0, height / 2.0])
            s = get_scale((width, height), [960,512])
            r = 0
            trans = get_affine_transform(c, s, r, [960,512])

            input = cv2.warpAffine(
                        data_numpy,
                        trans, (int(960), int(512)),
                        flags=cv2.INTER_LINEAR)
            input = _transform(input) # 3, 512, 960
            input_batch_holder.append(input)
            
        input_batch_holder = torch.stack(input_batch_holder, dim=0)

#         input = input.unsqueeze(0)

        _inputs_new.append(input_batch_holder)
        _inputs_vis.append(data_numpy)
        
    return _inputs_new, _inputs_vis