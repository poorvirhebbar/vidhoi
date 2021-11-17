import time
import numpy as np
import math
import sys
import os
import pickle as pkl
from slowfast.datasets import loader
import slowfast.utils.logging as logging
from tqdm import tqdm
import torch
import gc
from torch import nn
import torch
from slowfast.models import build_model

from src.detectron2.detectron2_repo.detectron2 import model_zoo
from src.detectron2.detectron2_repo.detectron2.modeling import build_model
from src.detectron2.detectron2_repo.detectron2.engine import DefaultPredictor
from src.detectron2.detectron2_repo.detectron2.config import get_cfg
from src.detectron2.detectron2_repo.detectron2.utils.visualizer import Visualizer

from src.detectron2.detectron2_repo.detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

import pdb; pdb.set_trace()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("vidor_train",)
cfg.DATASETS.TEST = ("",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
model = build_model(cfg)

resnext101 = torch.hub.load('pytorch/vision:v0.4.0','resnext101_32x8d',pretrained=True)
resnext101 = nn.Sequential(*(list(resnext101.children())[:-3]))
resnext101.eval()
resnext101.cuda()

def get_gpu_usage(show):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)/1024**3
    a = torch.cuda.memory_allocated(0)/1024**3
    f = r-a  # free inside reserved
    if(show):
        print("cached : ",r, 'GB')
        print("allocated : ",a, 'GB')
        print("available : ",f, 'GB')
    return r

def print_time(t):
    days = t // 86400
    hours = t // 3600 % 24
    minutes = t // 60 % 60
    seconds = t % 60
    #print(days,"days",hours,":",minutes,":",seconds)
    print(hours,":",minutes,":",seconds)


def forward_step(cfg,mode):
    print("LOADING DATA FOR : ",mode)
    data_loader = loader.construct_loader(cfg,mode)
    features = {}
    video_to_files = {} #video_to_files have video_idx, sec, to filename(where features are saved) mapping.
                        #To be used to map features to corresponding annotations
    j=1# will have next filename
    file_name = f'feat{j:04d}.pkl'
    print("Generating Feature......")
    for i,(imgs,meta) in tqdm(enumerate(data_loader), total=len(data_loader)):
        '''
        (Pdb) imgs.shape
        torch.Size([16, 3, 8, 224, 224]) -> [batch_size, channels, num_frames, H, W]
        (pdb) meta.keys()
        dict_keys(['boxes', 'ori_boxes', 'metadata', 'obj_classes', 'obj_classes_lengths', 'action_labels', 'trajectories'])
        (Pdb) meta['boxes'].shape
        torch.Size([57, 5])
        (Pdb) meta['obj_classes'].shape
        torch.Size([57, 2])
        (Pdb) len(meta['obj_classes_lengths'])
        16
        (Pdb) meta['obj_classes_lengths']                                               
        tensor([3, 4, 6, 2, 2, 3, 4, 5, 3, 5, 4, 4, 4, 4, 2, 2], dtype=torch.int32)
        (Pdb) sum(meta['obj_classes_lengths'])
        tensor(57, dtype=torch.int32)
        (Pdb) meta['action_labels'].shape                                               
        torch.Size([225, 52])
        (Pdb) meta['trajectories'].shape                                                
        torch.Size([57, 8, 5])

        '''
        batch_size = imgs[0].shape[0]
        num_frames = imgs[0].shape[2]
        feat_dim = 1024

        imgs = imgs[0].permute(0,2,1,3,4).reshape(batch_size*num_frames, imgs[0].shape[1], imgs[0].shape[3],imgs[0].shape[4])
        imgs = imgs.cuda()
        resnext_feats = resnext101(imgs).cpu().detach()
        imgs = imgs.cpu().detach()
        H = resnext_feats.shape[-2]
        W = resnext_feats.shape[-1]
        resnext_feats = resnext_feats.reshape(batch_size, num_frames, feat_dim, H, W).permute(0,2,1,3,4).float()

        # to get sec of video, holds index of 1st object of video segment in stream of objects
        # import pdb; pdb.set_trace()
        idx = 0
        for b in range(batch_size):
            video_id = meta['video_id'][b]
            num_objs = meta['obj_classes_lengths'][b]
            sec = meta['metadata'][idx][1].item()
            
            '''
            ## test code ##
            key = video_id + "_" + str(sec)
            assert key not in features.keys()
            features[key]= resnext_feats[b]
            
            assert key not in video_to_files.keys()
            video_to_files[key] = file_name
            '''
            '''
            if video_id not in features.keys():
                features[video_id] = {}
            features[video_id][sec] = resnext_feats[b]
            
            if video_id not in video_to_files.keys():
                video_to_files[video_id] = {}
            video_to_files[video_id][sec] = file_name
            
            idx += num_objs
            '''
            # import pdb; pdb.set_trace()
            video_folder,video = video_id.split('/')
            
            video_folder_path = os.path.join(cfg.VIDOR.FEAT_DIR,mode,video_folder)
            if not os.path.isdir(video_folder_path):
                os.mkdir(video_folder_path)

            video_path = os.path.join(video_folder_path, video)
            if not os.path.isdir(video_path):
                os.mkdir(video_path)
            
            filename = f'{sec:04d}.pkl'
            destfile = os.path.join(video_path, filename)
            with open(destfile,'wb') as f:
                temp = {}
                temp['features'] = resnext_feats[b]
                temp['trajectory_boxes'] = meta['trajectory_boxes'][idx:idx+num_objs,:]
                pkl.dump(temp,f)
                del temp
            
            idx += num_objs
        '''
        if i!=0 and i%50 == 0 or i == len(data_loader)-1:
            feat_file = open(os.path.join(cfg.VIDOR.FEAT_DIR,mode,file_name),"wb")
            pkl.dump(features,feat_file)
            feat_file.close()
            j+=1
            file_name = f'feat{j:04d}.pkl'
            del features
            features = {}
        '''
        #if j==3:
        #    break

    #pkl.dump(video_to_files,open(os.path.join(cfg.VIDOR.FEAT_DIR,mode,"video_to_files.pkl"),"wb"))

def generate_features(cfg):
    #forward_step(cfg,"train")
    forward_step(cfg,"val")

