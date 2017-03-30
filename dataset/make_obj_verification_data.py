import os
import json
import unicodedata
import numpy as np
import json
import random

# loading metaata and instances
random_train = "random_train.txt"
random_val = "random_val.txt"
metadata_file = "metadata.json"
instance_file = "instances.json"

# get quantity given index
def get_quantity(idx):
    quantity = 0
    if metadata[idx]:
        quantity = metadata[idx]['EXPECTED_QUANTITY']
    return quantity

# get an array that shows if the images are 0 < quantity < 6
# True or False
def get_moderate(split_file):
    print("loading random split file")
    moderate = np.zeros(N,bool)
    with open(split_file) as f:
        for line in f.readlines():
            idx = int(line)-1
            quantity = get_quantity(idx)
            if quantity > 0 and quantity < 6:
                moderate[idx] = True
    return moderate 

# making split train list
# each element, it contains (image idx, object list)
# for each objects, it contains list of indices of images that contain the object
def get_train_list(train_candidates):
    train_list = []
    for idx in range(N):
        if train_candidates[idx]:
            bin_info = metadata[idx]['BIN_FCSKU_DATA']
            bin_keys = bin_info.keys()
            object_list = []
            # iterate over objects in the bin
            for j in range(0,len(bin_info)):
                asin = bin_info[bin_keys[j]]['asin']
                if asin==None:
                    continue
                repeat_in = 0
                target_idx_list = []
                # how many this object repeted in the candidate images?
                for target_idx in instances[asin]['bin_list']:
                    if train_candidates[target_idx]==True and target_idx!=idx:
                        repeat_in = repeat_in + 1
                        target_idx_list.append(target_idx)
                # if repeat > 1, then at least twice showed up, we can make a
                # positive pair with this object
                if repeat_in > 1:
                    object_list.append([asin,target_idx_list])
            if object_list: 
                train_list.append([idx,object_list])
    return train_list

# making split val list(for positive pair)
# each element, it contains (image idx, instance asin, pos or neg, target index list),
def get_val_pos_list(train_candidates, val_candidates):
    val_list = []
    for idx in range(N):
        if val_candidates[idx]:
            bin_info = metadata[idx]['BIN_FCSKU_DATA']
            bin_keys = bin_info.keys()
            # iterate over objects in the bin
            for j in range(0,len(bin_info)):
                asin = bin_info[bin_keys[j]]['asin']
                if asin==None:
                    continue
                target_idx_list = []
                for target_idx in instances[asin]['bin_list']:
                    if train_candidates[target_idx]==True and target_idx!=idx:
                        target_idx_list.append(target_idx)
                if target_idx_list:
                    val_list.append([idx, asin, 1, target_idx_list])
    return val_list 

# making split val list(for negative pair)
def get_val_neg_list(train_candidates, val_candidates):
    val_list = []
    for idx in range(N):
        if val_candidates[idx]:
            # pick up random object asin
            asin = instance_keys[random.randint(0,N_inst-1)]
            if asin==None:
                continue
            inst = instances[asin]
            target_idx_list = []
            for target_idx in inst['bin_list']:
                if train_candidates[target_idx]==True and target_idx!=idx:
                    target_idx_list.append(target_idx)
            if len(target_idx_list) > 0:
                val_list.append([idx, asin, 0, target_idx_list])
    return val_list 

if __name__ == "__main__":
    print("loading metadata!")
    with open(metadata_file) as json_file:
        metadata = json.load(json_file)
    print("loading instance data!")
    with open(instance_file) as json_file:
        instances = json.load(json_file)
    instance_keys = instances.keys()
    N = len(metadata)
    N_inst = len(instance_keys)

    moderate_train = get_moderate(random_train)
    moderate_val = get_moderate(random_val)

    # building training sets
    train_list = get_train_list(moderate_train)

    # building validataion sets
    val_pos_list = get_val_pos_list(moderate_train, moderate_val)
    val_neg_list = get_val_neg_list(moderate_train, moderate_val)
    pos_samples = random.sample(val_pos_list,20000)
    neg_samples = random.sample(val_neg_list,20000)
    val_list = pos_samples + neg_samples
    random.shuffle(val_list)
    
    print("dumping train and val sets into json file")
    with open('obj_verification_train.json','wb') as f:
        json.dump(train_list,f)
    with open('obj_verification_val.json','wb') as f:
        json.dump(val_list,f)
