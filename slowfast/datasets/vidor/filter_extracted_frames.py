import os
import json
import pandas as pd
from tqdm import tqdm

DATA_ROOT = "/mnt/data/apoorva/HOI_vidhoi/slowfast/datasets/vidor"

FRAMES_LIST_DIR = os.path.join(DATA_ROOT,'frame_lists')
TRAIN_FRAMES_LIST_PATH = os.path.join(FRAMES_LIST_DIR,'train_full.csv')
VAL_FRAMES_LIST_PATH = os.path.join(FRAMES_LIST_DIR,'val_full.csv')

def match_frame_count(TRAJ, FRAMES_DF, SPLIT):
    extra_frames_count=0
    max_diff = 0
    min_diff = 1000
    filtered_frames = {
            'original_vido_id':[],
            'video_id':[],
            'frame_id':[],
            'path':[],
            'labels':[]
            }
    
    FRAMES_GROUPED = FRAMES_DF.groupby(by='original_vido_id',as_index=False,sort=False)
    FRAMES_COUNT_DF = FRAMES_GROUPED.count()
    for i in tqdm(FRAMES_COUNT_DF.index):
        video_id = FRAMES_COUNT_DF['original_vido_id'][i]
        frames_count = FRAMES_COUNT_DF['frame_id'][i]
        traj_count = len(TRAJ[video_id])
        diff = frames_count-traj_count
        max_diff = max(max_diff,diff)
        min_diff = min(min_diff,diff)
        extra_frames_count+=diff
        if diff!=0: #removing last frame as it is extra
            filtered_frames['original_vido_id'] += list(FRAMES_GROUPED.get_group(video_id)['original_vido_id'])[:-1]
            filtered_frames['video_id'] += list(FRAMES_GROUPED.get_group(video_id)['video_id'])[:-1]
            filtered_frames['frame_id'] += list(FRAMES_GROUPED.get_group(video_id)['frame_id'])[:-1]
            filtered_frames['path'] += list(FRAMES_GROUPED.get_group(video_id)['path'])[:-1]
            filtered_frames['labels'] += list(FRAMES_GROUPED.get_group(video_id)['labels'])[:-1]
        else:
            filtered_frames['original_vido_id'] += list(FRAMES_GROUPED.get_group(video_id)['original_vido_id'])
            filtered_frames['video_id'] += list(FRAMES_GROUPED.get_group(video_id)['video_id'])
            filtered_frames['frame_id'] += list(FRAMES_GROUPED.get_group(video_id)['frame_id'])
            filtered_frames['path'] += list(FRAMES_GROUPED.get_group(video_id)['path'])
            filtered_frames['labels'] += list(FRAMES_GROUPED.get_group(video_id)['labels'])
    print(extra_frames_count)
    print(max_diff)
    print(min_diff)
    pd.DataFrame(filtered_frames).to_csv(os.path.join(FRAMES_LIST_DIR,SPLIT+'.csv'),sep=' ',index=False)
if __name__=='__main__':
    
    TRAIN_TRAJECTORIES = json.load(open(os.path.join(DATA_ROOT,'train_trajectories.json'),'r'))
    TRAIN_FRAMES_DF = pd.read_csv(open(TRAIN_FRAMES_LIST_PATH,'r'),sep=" ")

    print("checking training frames")
    match_frame_count(TRAIN_TRAJECTORIES,TRAIN_FRAMES_DF,'train')

    VAL_TRAJECTORIES = json.load(open(os.path.join(DATA_ROOT,'val_trajectories.json'),'r'))
    VAL_FRAMES_DF = pd.read_csv(open(VAL_FRAMES_LIST_PATH,'r'),sep=" ")

    print("checking validation frames")
    match_frame_count(VAL_TRAJECTORIES,VAL_FRAMES_DF,'val')
