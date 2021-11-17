import json
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_label_distribution(mode):
    with open(f'{mode}_frame_annots.json','rb') as f:
        data = json.load(f)

    # with open(f'idx_to_pred.pkl','rb') as f:
    #     idx_to_pred = pkl.load(f)
    
    label_freq = {}
    
    for d in data:
        # label = idx_to_pred[d['action_class']]
        label = d['action_class']
        if label in label_freq.keys():
            label_freq[label] += 1
        else:
            label_freq[label] = 1
    
    label_freq = dict(sorted(label_freq.items(), key = lambda item: item[1], reverse=True))
    print(label_freq)

    pkl.dump(label_freq,open('train_action_frequency_class.pkl','wb'))

get_label_distribution("train")
