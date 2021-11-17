import json
import argparse
import torch
import pickle
from slowfast.utils.meters import get_accuracy
from slowfast.utils.vidor_eval_helper import evaluate_vidor, search_thresholds
from slowfast.visualization.utils import get_confusion_matrix, plot_confusion_matrix
from torch.nn.functional import one_hot
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('val_results')
args = parser.parse_args()
val_file = args.val_results
#val_file = "output/SLOWFAST_32x2_R50_SHORT_SCRATCH_EVAL_GT_trajectory-toipool/all_results_vidor_checkpoint_epoch_00010.pyth_proposal_less-168-examples.json"

data = json.load(open(val_file))
"""
>>> data[0].keys()
dict_keys(['preds_score', 'preds', 'preds_bbox_pair_ids', 'proposal_scores', 'proposal_boxes', 'proposal_classes', 'gt_boxes', 'gt_action_labels', 'gt_obj_classes', 'gt_bbox_pair_ids'])
"""
preds_score = [x['preds_score'] for x in data]
"""
>>>len(preds_score)
22808
"""
preds = [x['preds'] for x in data]
preds_bbox_pair_ids = [x['preds_bbox_pair_ids'] for x in data]
proposal_scores = [x['proposal_scores'] for x in data]
proposal_boxes = [x['proposal_boxes'] for x in data]
proposal_classes= [x['proposal_classes'] for x in data]
gt_boxes = [x['gt_boxes'] for x in data]
gt_action_labels = [x['gt_action_labels'] for x in data]
gt_obj_classes = [x['gt_obj_classes'] for x in data]
gt_bbox_pair_ids = [x['gt_bbox_pair_ids'] for x in data]

print(val_file)

#thresh_vec = search_thresholds(preds, preds_score, preds_bbox_pair_ids, proposal_scores, proposal_boxes, proposal_classes, gt_boxes, gt_action_labels, gt_obj_classes, gt_bbox_pair_ids)
#mAP,_,_,_,_ = evaluate_vidor(preds, preds_score, preds_bbox_pair_ids, proposal_scores, proposal_boxes, proposal_classes, gt_boxes, gt_action_labels, gt_obj_classes, gt_bbox_pair_ids, thresh_vec)
#mAP,_,_,_,_ = search_thresholds(preds, preds_score, preds_bbox_pair_ids, proposal_scores, proposal_boxes, proposal_classes, gt_boxes, gt_action_labels, gt_obj_classes, gt_bbox_pair_ids)
#thresh_vec = search_thresholds(preds, preds_score, preds_bbox_pair_ids, proposal_scores, proposal_boxes, proposal_classes, gt_boxes, gt_action_labels, gt_obj_classes, gt_bbox_pair_ids)
#print(thresh_vec)
#accuracy = get_accuracy(preds_score, gt_action_labels)

#print("VAL mAP = ",mAP)
#print("VAL accuracy = ",accuracy) 

mAP,_,_,_,_ = evaluate_vidor(preds, preds_score, preds_bbox_pair_ids, proposal_scores, proposal_boxes, proposal_classes, gt_boxes, gt_action_labels, gt_obj_classes, gt_bbox_pair_ids, mode="rare")
print("VAL rare  mAP = ",mAP)

mAP,_,_,_,_ = evaluate_vidor(preds, preds_score, preds_bbox_pair_ids, proposal_scores, proposal_boxes, proposal_classes, gt_boxes, gt_action_labels, gt_obj_classes, gt_bbox_pair_ids, mode="non_rare")
print("VAL non rare mAP = ",mAP)

'''
#import pdb; pdb.set_trace()
preds_score = [torch.Tensor(x) for x in preds_score]
preds_labels = [one_hot(torch.argmax(x,dim=1),num_classes=50) for x in preds_score]

#preds_score = [torch.Tensor(x) for x in preds_score]
gt_action_labels = [torch.Tensor(x) for x in gt_action_labels]

last_idx = 20

with open('slowfast/datasets/vidor/idx_to_pred.pkl','rb') as f:
    idx_to_pred = pickle.load(f)
predicate_labels = []
for i in range(last_idx):
    predicate_labels.append(idx_to_pred[i].split('(')[0])
print(predicate_labels)
# get confusion matrx
#cmtx = get_confusion_matrix(preds_labels, gt_action_labels, 50)
from sklearn.metrics import multilabel_confusion_matrix
preds_labels = torch.cat(preds_labels, dim=0)
gt_action_labels = torch.cat(gt_action_labels, dim=0)
#import pdb; pdb.set_trace()
preds_labels = preds_labels[:,:last_idx]
gt_action_labels = gt_action_labels[:,:last_idx]
cmtx = get_confusion_matrix(preds_labels, gt_action_labels, last_idx)
fig = plot_confusion_matrix(cmtx, last_idx, predicate_labels, [19.2, 13.4]) #[19.2, 13.4])
cmtxpath = val_file.split('.')[:-1]
cmtxpath.append(f'first_{last_idx}')
cmtxpath = '.'.join(cmtxpath) + '.png'
plt.savefig(cmtxpath)
#cmtx = multilabel_confusion_matrix(gt_action_labels, preds_labels)
#import pdb; pdb.set_trace()
#print(cmtx)
'''
