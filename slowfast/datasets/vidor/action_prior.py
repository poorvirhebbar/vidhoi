import json
import pickle as pkl
import torch

with open('train_frame_annots.json','r') as f:
    instances = json.load(f)

num_classes = 50 #action label count
adjacency_matrix = torch.zeros(num_classes, num_classes).long()

data = {}# has hopair and corresponding list of action labels

for instance in instances:
    frame_id = instance['video_folder'] + "/" + instance['video_id'] + "/" + instance['frame_id']
    hopair = str(instance['person_id'])+"_"+str(instance['object_id'])
    hopair_id = frame_id + "/" + hopair
    if hopair_id not in data.keys():
        data[hopair_id] = [instance['action_class']]
    else:
        data[hopair_id] += [instance['action_class']]


for hopair in data.keys():
    labels = data[hopair]
    for i in range(len(labels)):
        for j in range(i+1,len(labels)-i):
            adjacency_matrix[labels[i]][labels[j]]+=1
            adjacency_matrix[labels[j]][labels[i]]+=1

with open('multi_label_action_prior.pkl','wb') as f:
    pkl.dump(adjacency_matrix,f)
