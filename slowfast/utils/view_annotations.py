#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import pickle as pkl
import cv2
import sys
import os
from matplotlib import pyplot as plt
import numpy as np
from math import sqrt

def save_preds(sel_anno, idx=0):
    for i, x in enumerate(sel_anno):
        frame_path = frame_root_path + '/' +  vid_id + "_" + x['frame_id'] + '.jpg'
        frame = cv2.imread(frame_path)
        bbox_start = (x['person_box']['xmin'],x['person_box']['ymin'])
        bbox_end = (x['person_box']['xmax'],x['person_box']['ymax'])
        frame = cv2.rectangle(frame,bbox_start, bbox_end,(0,0,255),2)
        cv2.putText(frame, 'subject' , (bbox_start[0], bbox_start[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(frame, 'person' , (bbox_start[0], bbox_start[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        bbox_start = (x['object_box']['xmin'],x['object_box']['ymin'])
        bbox_end = (x['object_box']['xmax'],x['object_box']['ymax'])
        frame = cv2.rectangle(frame,bbox_start, bbox_end,(255,0,255),2)
        cv2.putText(frame, 'object' , (bbox_start[0], bbox_start[1]-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
        cv2.putText(frame, obj[x['object_class']] , (bbox_start[0], bbox_start[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig.add_subplot(4, 4, i+1)
        plt.imshow(frame)
        plt.axis('off')
        plt.title(x['frame_id'] + ', ' +pred[x['action_class']])
    plt.savefig(f'vis_val_{idx}.png')


def save_pred_imgs(cfg, results, mode):
    '''
    (Pdb) results.keys()
    dict_keys(['preds_score', 'preds', 'preds_bbox_pair_ids', 'proposal_scores', 'proposal_boxes', 'proposal_classes', 'gt_boxes', 'ori_boxes', 'gt_action_labels', 'gt_obj_classes', 'gt_bbox_pair_ids'])
    '''
    pred = pkl.load(open(os.path.join(cfg.VIDOR.ANNOTATION_DIR, 'idx_to_pred.pkl'),'rb'))
    obj = pkl.load(open(os.path.join(cfg.VIDOR.ANNOTATION_DIR, 'idx_to_obj.pkl'),'rb'))

    fig = plt.figure(figsize=(18,18))
    # fig.subplots_adjust(hspace=0.5, wspace=0)
    
    assert len(results['orig_video_idx']) == 1

    orig_video_id = results['orig_video_idx'][0]
    #'0085/5018581116/5018581116_000241'
    
    if mode in ['val', 'test']:    
        frame_path = os.path.join(cfg.VIDOR.FRAME_DIR,'validation',f'{orig_video_id}.jpg')
    
    if mode in ['train']:    
        frame_path = os.path.join(cfg.VIDOR.FRAME_DIR,'training',f'{orig_video_id}.jpg')

    # no. of rows and cols in subplot
    sub_c = int(sqrt(len(results['gt_bbox_pair_ids'])))
    sub_r = len(results['gt_bbox_pair_ids'])//sub_c

    for i, pair_id in enumerate(results['gt_bbox_pair_ids']):
        # import pdb; pdb.set_trace()
        frame = cv2.imread(frame_path)
        if frame is None:
            break
        # frame = imgs[0,:,num_frames//2,:,:]
        
        import pdb; pdb.set_trace() 

        #bbox = results['ori_boxes'][pair_id[0]]
        bbox = results['gt_boxes'][pair_id[0]]
        bbox_start = (int(bbox[1]),int(bbox[2]))
        bbox_end = (int(bbox[3]),int(bbox[4]))
        frame = cv2.rectangle(frame,bbox_start, bbox_end,(0,0,255),2)
        # cv2.putText(frame, 'subject' , (bbox_start[0], bbox_start[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
        # cv2.putText(frame, 'person' , (bbox_start[0], bbox_start[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)

        
        #bbox = results['ori_boxes'][pair_id[1]]
        bbox = results['gt_boxes'][pair_id[1]]
        bbox_start = (int(bbox[1]),int(bbox[2]))
        bbox_end = (int(bbox[3]),int(bbox[4]))
        frame = cv2.rectangle(frame,bbox_start, bbox_end,(255,0,255),2)
        #cv2.putText(frame, 'object' , (bbox_start[0], bbox_start[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 1)
        #cv2.putText(frame, obj[results['proposal_classes'][pair_id[1]][1]] , (bbox_start[0], bbox_start[1]+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
        
        if sub_c * sub_r < i+1:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fig.add_subplot(sub_c, sub_r, i+1)
        plt.imshow(frame)
        plt.axis('off')
        
        gt_action_label = [ pred[j] for j,x in enumerate(results['gt_action_labels'][i]) if x==1]
        # pred_action_label = [ pred[j] for j,x in enumerate(results['preds'][i]) if x]
        #import pdb; pdb.set_trace()
        gt_action_label_count = len(gt_action_label)
        pred_action_index = sorted(enumerate(results['preds_score'][i]), key= lambda x : x[1])[-gt_action_label_count:]
        
        pred_action_label = [pred[x] for x,y in pred_action_index]
        
        #pred_action_label = [ pred[results['preds_score'][i].index(max(results['preds_score'][i]))]]
        
        title = "GT: " + ','.join(gt_action_label) + "  " + "PRED: " + ','.join(pred_action_label)
        
        plt.title(title, fontsize=15)

    orig_video_id = orig_video_id.replace("/","_")
    plt.subplot_tool()
    VISUALIZE_DIR = os.path.join(cfg.OUTPUT_DIR, "visualize")
    if not os.path.exists(VISUALIZE_DIR):
        os.mkdir(VISUALIZE_DIR)
    plt.savefig(os.path.join(VISUALIZE_DIR, f'{orig_video_id}.png'))

if __name__ == "__main__":
    root_dir = sys.argv[1]

    anno = json.load(open(os.path.join(root_dir, 'val_frame_annots.json')))
    pred = pkl.load(open(os.path.join(root_dir, 'idx_to_pred.pkl'),'rb'))
    obj = pkl.load(open(os.path.join(root_dir, 'idx_to_obj.pkl'),'rb'))

    vid_fld = '0004'
    vid_id = '3022452780'
    sel_anno = [ x for x in anno if x['video_folder']==vid_fld and x['video_id']==vid_id]


    fig = plt.figure(figsize=(18, 18))
    fig.subplots_adjust(hspace=0.5, wspace=0)

    frame_root_path = os.path.join(root_dir, 'frames', 'validation', '0004', '3022452780')


    save_preds(sel_anno, vid_fld)

    # In[ ]:




