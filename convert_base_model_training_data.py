import pickle as pkl
import random

with open('dataset/data_nhq_triplet/tagop_roberta_cached_train.pkl','rb') as f:
    train = pkl.load(f)

mix_train = []

oq, hq, nhq = 0, 0, 0

for line in train:
    mix_train.append(line[0])
    oq += 1
    if line[1]:
        mix_train.append(line[1])
        hq += 1
    if line[2]:
        mix_train.append(line[2])
        nhq += 1

random.shuffle(mix_train)

with open('dataset/data_nhq_mix/tagop_roberta_cached_train.pkl','wb') as f:
    pkl.dump(mix_train, f)
    
print("Finished converting data, oq {} hq {} nhq {}".format(oq, hq, nhq))

with open('dataset/data_nhq_triplet/tagop_roberta_cached_dev.pkl','rb') as f:
    dev = pkl.load(f)

with open('dataset/data_nhq_mix/tagop_roberta_cached_dev.pkl','wb') as f:
    pkl.dump(dev, f)
    
    
