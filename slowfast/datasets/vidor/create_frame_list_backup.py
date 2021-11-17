import os
import json
import pandas as pd
from tqdm import tqdm

DATA_ROOT = '/mnt/data/apoorva/HOI_vidhoi/slowfast/datasets/vidor'
FRAME_PATH = os.path.join(DATA_ROOT,'frames')
TRAIN_ANNO_FILE_PATH = os.path.join(DATA_ROOT,'train_frame_annots.json')
VAL_ANNO_FILE_PATH = os.path.join(DATA_ROOT,'val_frame_annots.json')
FRAME_LIST_DIR = os.path.join(DATA_ROOT,'frame_lists')
TRAIN_CSV = os.path.join(FRAME_LIST_DIR,'train.csv')
VAL_CSV = os.path.join(FRAME_LIST_DIR,'val.csv')

if not os.path.exists(FRAME_LIST_DIR):
    print("Created frame list directory")
    os.mkdir(FRAME_LIST_DIR)

def create_save_frame_list(DATA,SPLIT):
    FRAME_LIST = {'original_vido_id' : [],
                    'video_id' : [],
                    'frame_id' : [],
                    'path' : [],
                    'labels' : []}
    
    if SPLIT=='train':
        FRAME_DIR = os.path.join(FRAME_PATH,'training')
    else:
        FRAME_DIR = os.path.join(FRAME_PATH,'validation')

    for blob in tqdm(DATA):
        video_id = blob['video_folder'] + '/' + blob['video_id']
        frame_path = blob['video_folder'] + '/' + blob['video_id'] + '/' + blob['video_id'] + '_' + blob['frame_id'] + '.jpg'
        path = os.path.join(FRAME_DIR,frame_path)
        
        FRAME_LIST['original_vido_id'].append(video_id)
        FRAME_LIST['video_id'].append(video_id)
        FRAME_LIST['frame_id'].append(blob['frame_id'])
        FRAME_LIST['path'].append(path)
        FRAME_LIST['labels'].append("")
        
    FRAME_LIST_DF = pd.DataFrame(FRAME_LIST)
    FRAME_LIST_DF.drop_duplicates(keep='first',inplace=True)

    FRAME_LIST_DF.to_csv(os.path.join(FRAME_LIST_DIR,SPLIT+'.csv'),sep=' ',index=False)

if __name__ == '__main__':
    TRAIN_ANNO = json.load(open(TRAIN_ANNO_FILE_PATH,'r'))
    VAL_ANNO = json.load(open(VAL_ANNO_FILE_PATH,'r'))

    print("Generating Train File")
    create_save_frame_list(TRAIN_ANNO,'train')
    
    print("Generating Val File")
    create_save_frame_list(VAL_ANNO,'val')
