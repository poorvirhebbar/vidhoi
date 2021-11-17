import json
import os
import pickle as pkl

def count(anno_file,mode):
    annots = json.load(open(anno_file,'r'))
    triplet = dict()
    rare = list()
    non_rare = list()
    for annot in annots:
        
        hopair = (0,annot['action_class'],annot['object_class'])
        
        if hopair not in triplet.keys():
            triplet[hopair] = 1
        else:
            triplet[hopair]+=1
    
    for hopair in triplet.keys():
        if triplet[hopair] >= 25:
            non_rare += [hopair]
        else:
            rare += [hopair]
     
    print("Rare count : ", len(rare))
    print("Non-Rare count : ", len(non_rare))
    print("total : ", len(triplet.keys()))
    with open('rare_triplets.pkl','wb') as f:
        pkl.dump(rare,f)
    with open('non_rare_triplets.pkl','wb') as f:
        pkl.dump(non_rare,f)
    #import pdb; pdb.set_trace()

if __name__ == "__main__":
    #count("debug_dataset_395_frames.json","train")
    
    #count("train_frame_annots.json","train")
    count("val_frame_annots.json","val")
