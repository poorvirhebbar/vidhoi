import pickle as pkl
import json
import os
import torch.nn.functional as F

"""
This file contains functions to load static data files like multi_label_action_prior.pkl file.
"""

def get_prior_matrix(cfg):
    with open(os.path.join(cfg.VIDOR.ANNOTATION_DIR, cfg.VIDOR.ACTION_PRIOR_FILE),'rb') as f:
        adjacency_matrix = pkl.load(f)
    
    return adjacency_matrix

def update_prior_to_pred_score(adjacency_matrix, pred_score):
    #import pdb; pdb.set_trace()
    # not implemented
    return pred_score
