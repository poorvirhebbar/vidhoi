#!/usr/bin/env python
# coding: utf-8

# In[62]:


import os
import json
import sys

result_dir = sys.argv[1]
result_json_name = sys.argv[2]
demo_result_json = sys.argv[3]
with open(os.path.join(result_dir, result_json_name), 'r') as f:
    all_results = json.load(f)

# For computing Temporal-mAP
only_evaluate_temporal_predicates = False # should be True when visualizing correct prediction!

### Settings for visualizing demo images purpose only
is_demo = True  #False
is_demo_incorrect_hois = True  #False
is_demo_save_imgs = True #False
is_demo_show_imgs = False
is_demo_top_10 = True #False # may be used with incorrect hois

only_demo_specific_videos = True #False
only_demo_specific_frames = False
to_demo_video_names = []
to_demo_frame_names = []

if is_demo:
#     assert only_evaluate_temporal_predicates ^ is_demo_incorrect_hois # sanity check!
    demo_vis_name = 'GT'
    #demo_vis_name = 'SLOWFAST'
    #demo_vis_name = 'VANILLA'
    #demo_vis_name = 'vidhoi_2D' # 'vidor_TP'
#     if is_demo_incorrect_hois:
#         demo_vis_name += '_wrong'
    if only_evaluate_temporal_predicates:
        demo_vis_name += '_onlytemp'
    if only_demo_specific_videos:
        to_demo_video_names = [
            '0024/5328004991',
            '0041/2604394962',
            '0030/8594314852',
            '0040/6711090395',
            '0052/2535384528',
            '0044/11565498775',
            '0060/11849091804',
            '0063/2435100235',
            '0091/5296635780',
            '1001/4335807873',
            '1001/5343885791',
            #'1110/2584172238',#theirs
        ]
        demo_vis_name += '_specvids'
    elif only_demo_specific_frames:
        to_demo_frame_names = [
#             '0080/9439876127',
#             '1009/2975784201_000106',
#             '1009/2975784201_000136',
#             '1009/2975784201_000166',
#             '1009/2975784201_000196',
#             '1009/2975784201_000226',
#             '1009/2975784201_000256',
#             '1009/4488998616',
#             '1009/4896969617_000016',
#             '1009/4896969617_000046',
#             '1009/4896969617_000076',
#             '1009/4896969617_000226',
#             '1009/4896969617_000256',
#             '1009/4896969617_000286',
#             '1017/4518113460_000376',
#             '1017/4518113460_000406',
#             '1017/4518113460_000436',
#             '1017/4518113460_000466',
#             '1017/4518113460_000496',
#             '1017/2623954636_000076',
#             '1017/2623954636_000136',
#             '1017/2623954636_000166',
#             '1017/2623954636_000196',
#             '1017/2623954636_000226',
#             '1017/2623954636_000256',
#             '1017/2623954636_000706',
#             '1017/2623954636_000736',
#             '1017/2623954636_000856',
#             '1017/2623954636_000886',
#             '1017/2623954636_000916',
#             '1009/7114553643_000736',
#             '1009/7114553643_000766',
#             '1009/7114553643_000796',
#             '1009/7114553643_000826',
#             '1009/7114553643_000856',
#             '1009/7114553643_000886',
#             '1009/7114553643_000916',
#             '1018/3155382178',
#             '1101/6305304857',
#             '1101/6443512089_000676',
#             '1101/6443512089_000706',
#             '1101/6443512089_000736',
#             '1101/6443512089_000766',
#             '1101/6443512089_000796',
#             '1101/6443512089_000826',
#             '1101/6443512089_000856',
            '1110/2584172238_000202',
            '1110/2584172238_000226',
            '1110/2584172238_000250',
            '1110/2584172238_000274',
            '1110/2584172238_000298',
            '1110/2584172238_000322',
            '1110/2584172238_000346',
        ]
        demo_vis_name += '_specvids'

# In[30]:


import pickle
with open('slowfast/datasets/vidor/idx_to_pred.pkl', 'rb') as f:
    idx_to_pred = pickle.load(f)
with open('slowfast/datasets/vidor/idx_to_obj.pkl', 'rb') as f:
    idx_to_obj = pickle.load(f)


# In[31]:


import pickle
with open('slowfast/datasets/vidor/pred_to_idx.pkl', 'rb') as f:
    pred_to_idx = pickle.load(f)
# pred_to_idx


# In[32]:


temporal_predicates = [
    'towards',
    'away',
    'pull',
    'caress',
    'push',
    'press',
    'wave',
    'hit',
    'lift',
    'pat',
    'grab',
    'chase',
    'release',
    'wave_hand_to',
    'squeeze',
    'kick',
    'shout_at',
    'throw',
    'smell',
    'knock',
    'lick',
    'open',
    'close',
    'get_on',
    'get_off',
]

if only_evaluate_temporal_predicates:
    temporal_predicates_idx = [pred_to_idx[pred] for pred in temporal_predicates]


# In[34]:

'''
all_results[0].keys()
'''

# In[36]:


# Visualization
import json
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

#json_file = './output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-spa_conf/all_results_vidor_checkpoint_epoch_00020.pyth_proposal_less-168-examples_demo-all.json'
json_file = demo_result_json
with open(json_file, 'r') as f:
    res = json.load(f)
print(len(res))


import cv2
import numpy as np
import math
# from slowfast.datasets.cv2_transform import scale

def scale(size, image):
    """
    Scale the short side of the image to size.
    Args:
        size (int): size to scale the image.
        image (array): image to perform short side scale. Dimension is
            `height` x `width` x `channel`.
    Returns:
        (ndarray): the scaled image with dimension of
            `height` x `width` x `channel`.
    """
    height = image.shape[0]
    width = image.shape[1]
    if (width <= height and width == size) or (
        height <= width and height == size
    ):
        return image
    new_width = size
    new_height = size
    if width < height:
        new_height = int(math.floor((float(height) / width) * size))
    else:
        new_width = int(math.floor((float(width) / height) * size))
    img = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
    )
    return img

def vis_detections_allclss(im, dets, dets_clss, vidor_clss, thresh=0.5, 
                           proposal_scores=None, sub_obj_pair=False, pred_cls=-1):
    """Visual debugging of detections."""
    for i in range(len(dets)):
        bbox = tuple(int(np.round(x)) for x in dets[i])
        class_name = vidor_clss[int(dets_clss[i])]
        if sub_obj_pair: # if evaluating on HOI pair
            class_name = class_name + '_s' if i == 0 else class_name + '_o'
        
        color = (0, 204, 0) if i == 0 else (0, 0, 204)

        if proposal_scores is not None:
            print('proposal_scores', proposal_scores)
            score = proposal_scores[i]
            if score > thresh:
#                 print(bbox)
                cv2.rectangle(im, bbox[0:2], bbox[2:4], color, 2)
#                 print(class_name, bbox, score)
                #cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                 #           1.0, (0, 0, 255), thickness=2)
        else:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], color, 2)
#             print(class_name, bbox)
            #cv2.putText(im, '%s' % (class_name), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
            #            1.0, (0, 0, 255), thickness=2)
    
    if pred_cls != -1:
        pred_name = idx_to_pred[pred_cls]
        box1_x1y1 = list(int(np.round(x)) for x in dets[0])[:2]
        box2_x1y1 = list(int(np.round(x)) for x in dets[1])[:2]
        box1_box2_mid = (np.array(box1_x1y1) + np.array(box2_x1y1)) / 2
        box1_box2_mid = tuple(int(np.round(x)) for x in box1_box2_mid)

        #cv2.line(im, tuple(box1_x1y1), tuple(box2_x1y1), (255, 0, 0), 2)
        #cv2.putText(im, pred_name, box1_box2_mid, cv2.FONT_HERSHEY_PLAIN,
        #                    1.0, (0, 0, 255), thickness=2)
#         cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

    return im

vidor_classes = list(idx_to_obj.values())
'''
# img = cv2.resize(img, (224,224))
img_vis = scale(224, img)

# visualize gt boxes
# img_vis = vis_detections_allclss(img, res[idx]['gt_boxes'], res[idx]['gt_obj_classes'], vidor_classes)

# visualize proposal boxes (same as gt boxes when in ORACLE model)
img_vis = vis_detections_allclss(img_vis, [x[1:] for x in res[idx]['proposal_boxes']], [x[1] for x in res[idx]['proposal_classes']], vidor_classes, 0.2, proposal_scores=[x[1] for x in res[idx]['proposal_scores']])

# Can use all other result file as idx remains in the same order
# img_vis = vis_detections_allclss(img_vis, [x[1:] for x in all_results[idx]['proposal_boxes']], [x[1] for x in all_results[idx]['proposal_classes']], vidor_classes, 0.2, proposal_scores=[x[1] for x in all_results[idx]['proposal_scores']])

plt.imshow(img_vis)
plt.show()
'''

# In[39]:


def vis_hoi(img_idx, sub_cls, pred_cls, obj_cls, gt_sub_box, gt_obj_box):
    # img_path = 'slowfast/datasets/vidor/frames/' + res[img_idx]['orig_video_idx'][0] + '.jpg'
    new_idx = f"{int(res[img_idx]['orig_video_idx'][0].split('_')[-1])-15:06d}"
    frame_name = f"{res[img_idx]['orig_video_idx'][0].split('_')[0] + '_' + new_idx}"
    img_path = f"slowfast/datasets/vidor/frames/validation/{frame_name}.jpg"
    
    img = plt.imread(img_path)
    img = scale(224, img)
    img = vis_detections_allclss(img, [gt_sub_box, gt_obj_box], [sub_cls, obj_cls], vidor_classes, sub_obj_pair=True, pred_cls=pred_cls)
    return img, '/'.join([frame_name.split('/')[0], frame_name.split('/')[2]]) # frame_name # '/'.join(frame_name.split('/')[:-1])

def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# In[66]:


from tqdm import tqdm
from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

tp, fp, scores, sum_gt = {}, {}, {}, {}

# Construct dictionaries of triplet class 
for result in tqdm(all_results):
    bbox_pair_ids = result['gt_bbox_pair_ids']
    
    for idx, action_label in enumerate(result['gt_action_labels']):
        # action_label != 0.0
        action_label_idxs = [i for i in range(50) if action_label[i] == 1.0]
        for action_label_idx in action_label_idxs:
            if only_evaluate_temporal_predicates and action_label_idx not in temporal_predicates_idx:
                continue
            
            subject_label_idx = 0 # person idx = 0
            object_label_idx = int(result['gt_obj_classes'][bbox_pair_ids[idx][1]][1])
            triplet_class = (subject_label_idx, action_label_idx, object_label_idx)
            
            if triplet_class not in tp: # should also not exist in fp, scores & sum_gt
                tp[triplet_class] = []
                fp[triplet_class] = []
                scores[triplet_class] = []
                sum_gt[triplet_class] = 0
            sum_gt[triplet_class] += 1
            
# Collect true positive, false positive & scores
import math

gt_labels = {}
for img_idx, result in enumerate(tqdm(all_results)): # for each keyframe
    frame_gt_labels = {}
    if is_demo:
        new_idx = f"{int(res[img_idx]['orig_video_idx'][0].split('_')[-1])-15:06d}"
        frame_name = f"{res[img_idx]['orig_video_idx'][0].split('_')[0] + '_' + new_idx}"
        frame_name = '/'.join([frame_name.split('/')[0], frame_name.split('/')[2]])
        video_name = frame_name.split('_')[0]
        
        if only_demo_specific_videos and video_name not in to_demo_video_names:
            continue
        if only_demo_specific_frames and frame_name not in to_demo_frame_names:
            continue
    gt_bbox_pair_ids = result['gt_bbox_pair_ids']
    for k, gt_bbox_pair_id in enumerate(gt_bbox_pair_ids):  # for each ground truth HOI
        gt_sub_cls = int(result['gt_obj_classes'][gt_bbox_pair_id[0]][1])
        gt_obj_cls = int(result['gt_obj_classes'][gt_bbox_pair_id[1]][1])
        
        gt_sub_box = result['gt_boxes'][gt_bbox_pair_id[0]][1:]
        gt_obj_box = result['gt_boxes'][gt_bbox_pair_id[1]][1:]
        
        pred_sub_cls = gt_sub_cls
        pred_obj_cls = gt_obj_cls
        
        for j in range(50): # for each j-th action
            gt_rel_cls = result['gt_action_labels'][k][j]
             
            if gt_rel_cls == 1.0:
                sub_cls = idx_to_obj[pred_sub_cls].replace('/','or') + str(gt_bbox_pair_id[0])
                pred_cls = idx_to_pred[j]
                obj_cls = idx_to_obj[pred_obj_cls].replace('/','or') + str(gt_bbox_pair_id[1])
                
                sub_obj_pair = f'{sub_cls}-{obj_cls}'
                if sub_obj_pair not in frame_gt_labels.keys():
                    frame_gt_labels[sub_obj_pair] = []
                frame_gt_labels[sub_obj_pair].append(pred_cls)
                
                '''
                save_dir = f'demo/{demo_vis_name}/{frame_name.split("/")[0]}'
                save_path = f'demo/{demo_vis_name}/{frame_name}' + '_' + f'{sub_cls}-{pred_cls}-{obj_cls}.jpg'
                    
                if (is_demo_save_imgs and not os.path.exists(save_path)) or is_demo_show_imgs:
                    img_vis, _ = vis_hoi(img_idx,pred_sub_cls,j,pred_obj_cls,gt_sub_box,gt_obj_box)
                    plt.imshow(img_vis)
                    plt.axis('off')
                    if is_demo_save_imgs: #and not os.path.exists(save_path):
                        if not os.path.isdir(save_dir):
                            os.makedirs(save_dir)

                        plt.savefig(save_path, bbox_inches='tight')
                    if is_demo_show_imgs:
                        print('save_path:', save_path)
                        plt.show() # (Optional) Show the figure. THIS SHOULD COME AFTER plt.savefig
                '''
        frame_id = frame_name.split("/")[-1]
        if video_name not in gt_labels.keys():
            gt_labels[video_name] = {}
        if frame_id not in gt_labels[video_name].keys():
            gt_labels[video_name][frame_id] = {}

        gt_labels[video_name][frame_id] = frame_gt_labels

with open(f'demo/gt_labels.json','w') as f:
    json.dump(gt_labels,f)
