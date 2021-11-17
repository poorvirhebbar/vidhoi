import json
import os

def count(anno_file,mode):
    annots = json.load(open(anno_file,'r'))
    frames = dict()
    videos = set()
    multi_label_videos = set()
    multi_label_frames = set()
    multi_label_hopair = set()
    for annot in annots:
        video_id = annot['video_folder'] + "/" + annot['video_id']
        frame_id = annot['video_folder'] + "/" + annot['video_id'] + "/" + annot['frame_id']
        hopair = str(annot['person_id'])+"_"+str(annot['object_id'])
        hopair_id = frame_id + "/" + hopair
        videos.add(video_id)
        if frame_id in frames.keys():
            if hopair in frames[frame_id]:
                multi_label_videos.add(video_id)
                multi_label_frames.add(frame_id)
                multi_label_hopair.add(hopair_id)
                #import pdb; pdb.set_trace()
            else:
                frames[frame_id].append(hopair)
        else:
            frames[frame_id] = [hopair]
    
    #import pdb; pdb.set_trace()
    print("***",mode,"***")
    
    print("count of VIDEO with multi label:",len(multi_label_videos))
    print("total VIDEO:",len(videos))
    print("fraction of VIDEO with multi label:",len(multi_label_videos)/len(videos))
    print("")
    print("count of FRAMES with multi label:",len(multi_label_frames))
    print("total FRAMES:",len(frames.keys()))
    print("fraction of FRAMES with multi label:",len(multi_label_frames)/len(frames.keys()))
    print("")
    print("count of HOPAIR with multi label:",len(multi_label_hopair))
    print("total HOPAIR:",len(annots))
    print("fraction of HOPAIR with multi label:",len(multi_label_hopair)/len(annots))
    print("")

if __name__ == "__main__":
    #count("debug_dataset_395_frames.json","train")
    
    count("train_frame_annots.json","train")
    count("val_frame_annots.json","val")
