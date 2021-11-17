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

res50 = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
res50 = torch.nn.Sequential(*(list(res50.children())[:-1]))
res50.cuda()
res50.eval()

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

def crop_roi(cfg, imgs, num_objs, trajectories):
    batch_size = imgs.shape[0]
    num_frames = imgs.shape[2]
    max_objs = cfg.LIGHTEN.MAX_OBJS
    
    start_idx = 0
    # setting dim max_objs to num_objs will only work when batch_size 1
    batch_roi = np.zeros([batch_size, num_frames, num_objs[0], 3, 224, 224])*1.0
    #batch_roi = np.zeros([batch_size, num_frames, max_objs, 3, 224, 224])*1.0
    for b,n in enumerate(num_objs):
        n = n.item()
        img = imgs[b]
        bboxes = trajectories[start_idx:start_idx+n,:,1:]
        for j in range(n):
            for f in range(num_frames):
                x1, y1, x2, y2 = math.floor(bboxes[j,f,0]), math.floor(bboxes[j,f,1]), math.floor(bboxes[j,f,2]), math.floor(bboxes[j,f,3])
                roi = img[:,f][:, y1:y2, x1:x2]

                if roi.shape[1] == 0 or roi.shape[2] == 0:
                    continue

                import pdb; pdb.set_trace()
                max_size = np.max(roi.shape[1], roi.shape[2])
                scale = 224 / max_size
                roi_scaled = cv2.resize(roi.transpose(1,2,0), (scale, scale))
                h, w, _ = roi_scaled.shape
                center = 112  
                temp_image = np.zeros(224, 224, 3)
                temp_image[center-h//2: center+h//2, center-w//2: center+w//2, :] = roi_scaled
                batch_roi[b, f, j] = roi_scaled.transpose(2, 0, 1) 
                
                # batch_roi[b, f, j, :, x1:x2, y1:y2] = roi
                #batch_roi[b, f, j] = np.resize(roi, (3, 224, 224))
    return batch_roi
        
"""
    batch_cropped_images = np.zeros([batch_size, 7, num_frames, 224, 224, 3])*1.0
    for b in range(batch_size):
        num_nodes = len(batch_bboxes[b])
        for n in range(num_nodes) :
            ifor f in range(num_frames) :
                box = batch_bboxes[b, n, f]
                x1, y1, x2, y2 = math.floor(box[0]), math.floor(box[1]), math.floor(box[2]), math.floor(box[3])
                batch_cropped = batch_images[b, f][x1:x2, y1:y2, :]
                if batch_cropped.shape[1] == 0 or batch_cropped.shape[2] == 0:
                    continue
                batch_segments[b, n, f] = resize(batch_cropped, (224, 224, 3))

    img_mean = np.array([0.485, 0.456, 0.406])
    img_std = np.array([0.229, 0.224, 0.225])
    batch_segments[:, :num_nodes] = (batch_segments[:, :num_nodes] - img_mean)/img_std
    return batch_segments
"""

def forward_step(cfg,mode):
    global res50
    print("LOADING DATA FOR : ",mode)
    data_loader = loader.construct_loader(cfg,mode)
    feat_dim = 2048
    feat_file = open(os.path.join(cfg.VIDOR.ANNOTATION_DIR,"resnet_feats.pkl"),"wb")
    features = {}
    print("Generating Feature......")
    for (imgs,meta) in tqdm(data_loader):
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
        num_objs = meta['obj_classes_lengths']
        video_id = meta['video_id'][0]
        batch_roi = crop_roi(cfg, imgs[0], meta['obj_classes_lengths'], meta['trajectories'])
        batch_roi = batch_roi.reshape(batch_size*num_frames,num_objs[0], 3, 224, 224)
        
        batch_roi = torch.from_numpy(batch_roi).float().cuda()
        res50_feats = res50(batch_roi).reshape(batch_size, num_frames, num_objs[0], feat_dim).permute(0,2,1,3).cpu().detach()
        if video_id in features.keys():
            features[video_id].append(res50_feats)
        else:
            features[video_id] = [res50_feats]
        
        #torch.cuda.empty_cache()
        #pkl.dump({video_id,res50_feats},feat_file)

    pkl.dump(features,feat_file)
    feat_file.close()

def generate_features(cfg):
    forward_step(cfg,"train")
