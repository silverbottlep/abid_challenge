import os
import json
import unicodedata
import numpy as np
import json
import random

# loading metaata
random_train = "random_train.txt"
random_val = "random_val.txt"
metadata_file = "metadata.json"

def get_quantity(idx):
    quantity = 0
    if metadata[idx]:
        quantity = metadata[idx]['EXPECTED_QUANTITY']
    return quantity

def get_moderate_list(split_file):
    print("loading random split file")
    train_list = []
    with open(split_file) as f:
        for line in f.readlines():
            idx = int(line)-1
            quantity = get_quantity(idx)
            if quantity < 6:
                train_list.append([idx,quantity])
    return train_list 

if __name__ == "__main__":
    print("loading metadata!")
    with open(metadata_file) as json_file:
        metadata = json.load(json_file)
    N = len(metadata)

    train_list = get_moderate_list(random_train)
    val_list = get_moderate_list(random_val)

    print("dumping train and val sets into json file")
    with open('counting_train.json','wb') as f:
        json.dump(train_list,f)
    with open('counting_val.json','wb') as f:
        json.dump(val_list,f)
