import json
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

def plot_label_distribution(mode):
    with open(f'{mode}_frame_annots.json','rb') as f:
        data = json.load(f)

    #with open(f'debug_dataset_395_frames.json','rb') as f:
    #    data = json.load(f)
    
    with open(f'idx_to_pred.pkl','rb') as f:
        idx_to_pred = pkl.load(f)
    
    label_freq = {}
    save_freq = [0]*50
    
    for d in data:
        label = idx_to_pred[d['action_class']]
        #if len(label.split('_')) > 2:
        #    label = label.split('_')[:-1]
        #    label = '_'.join(label)
        save_freq[d['action_class']] += 1
        if label in label_freq.keys():
            label_freq[label] += 1
        else:
            label_freq[label] = 1

    with open('predicate_frequencies.pkl', 'wb') as f:
        pickle.dump(save_freq, f)

    #import pdb; pdb.set_trace()
    
    label_freq = dict(sorted(label_freq.items(), key = lambda item: item[1], reverse=True))

    labels = list(label_freq.keys())
    freq = list(label_freq.values())
    
    
    weights = [1]*len(freq)
    '''
    weights[3] = 0.05
    weights[7] = 0.1
    
    #weights[1] = weights[4] = 0.5
    factor = 1
    if mode == 'val':
        factor = 10
    for i,v in enumerate(freq):
        if v > 300000/factor :
            weights[i] = 0.1
        elif v > 100000/factor:
            weights[i] = 0.5
        elif v < 10000/factor and v > 5000/factor:
            weights[i] = 5
        elif v < 5000/factor and v > 1000/factor:
            weights[i]  = 10
        elif v < 1000/factor and v > 100/factor:
            weights[i] = 50
        elif v < 100/factor:
            weights[i] = 100
    
    print(weights) 
    print(len(weights))
    print(len(freq))
    '''
    for l in idx_to_pred.values():
        if l not in labels:
            print(f'{l} not in {mode}')
    
    weighted_freq = [weights[i] * freq[i] for i in range(len(freq))]
    fontsize = 80
    selected_idx = [x  for x in range(50)if x<5 or x>44 or x%4==0]

    for i in range(1):

        #data = {"Predicate" : labels,
        #        "Frequency" : weighted_freq}
        if i==0:
            labels = [x for i,x in enumerate(labels) if i in selected_idx]
            weighted_freq = [x for i,x in enumerate(weighted_freq) if i in selected_idx]
            pred = [x.split('(')[0] for x in labels]
            predicates = ['_'.join(x.split('_')[:2]) if 'front' not in x else x for x in pred ]
                
        if i==1:
            predicates = ['_'.join(x.split('_')[:2]) for x in labels]
    

        data = {"Predicate" : predicates,
                "Frequency" : weighted_freq}
        
        #data = {"Predicate" : predicates[i*25:(i+1)*25],
        #        "Frequency" : weighted_freq[i*25:(i+1)*25]}
        



        data_df = pd.DataFrame(data, columns=['Predicate','Frequency']) 
        plt.figure(figsize = (50, 25))
        
        #plt.bar(labels, freq, width=0.4)
        plots = sns.barplot(x="Predicate",y="Frequency",data=data_df)
        for bar in plots.patches:
            plots.annotate(str(int(bar.get_height())), 
                            (bar.get_x() + bar.get_width()/2, bar.get_height()), 
                            ha="center", va="center", rotation=90, xytext=(0,120), 
                            textcoords="offset points", size=65)

        #plt.title(f'{mode} split predicate frequency distribution'.capitalize(), size=fontsize)
        plt.xlabel("Predicates",size=fontsize)
        plt.ylabel('Frequency',size=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xticks(rotation=85, fontsize=fontsize)
        #for i, v in enumerate(freq):
        #    plt.text(v, i, str(v))
        plt.tight_layout()
        #plt.savefig(f'{mode}_label_distribution.png')
        plt.savefig(f'{mode}_label_distribution_{i}.png')
        #plt.savefig(f'debug_dataset_label_distribution.png')

#plot_label_distribution("val")
plot_label_distribution("train")
