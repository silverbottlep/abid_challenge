import os
from os import listdir
import os.path
import numpy as np
import random
import json

img_dir= "data/public_images/"
meta_dir = "data/metadata/"

# getting whole metadata list 
def get_metadata(img_dir,meta_dir):
    metadata=[]
    n_images=0
    img_list = listdir(img_dir)
    N = len(img_list)
    for i in range(N):
        if i%1000 == 0:
            print("get_metadata: processing (%d/%d)..." % (i,N))
        jpg_path = '%s%05d.jpg' % (img_dir,i+1)
        json_path = '%s%05d.json' % (meta_dir,i+1)
        if os.path.isfile(jpg_path) and os.path.isfile(json_path):
            d = json.loads(open(json_path).read())
            metadata.append(d)
            n_images = n_images + 1
        else:
            metadata.append({})
    print("get_metadata: Available Images: %d" % n_images)
    return metadata,n_images

# getting instance list
def get_instance_data(metadata):
    instances={}
    N = len(metadata)
    for i in range(N):
        if i%1000 == 0:
            print("get_instance_data: processing (%d/%d)..." % (i,N))
        if metadata[i]:
            quantity = metadata[i]['EXPECTED_QUANTITY']
            if quantity>0:
                bin_info = metadata[i]['BIN_FCSKU_DATA']
                bin_keys = bin_info.keys()
                for j in range(0,len(bin_info)):
                    instance_info = bin_info[bin_keys[j]]
                    asin = instance_info['asin']
                    if asin in instances:
                        # occurance
                        instances[asin]['repeat'] = instances[asin]['repeat'] + 1
                        # quantity
                        instances[asin]['quantity'] = instances[asin]['quantity'] + instance_info['quantity']
                        instances[asin]['bin_list'].append(i)
                    else:
                        instances[asin]={}
                        instances[asin]['repeat'] = 1
                        instances[asin]['quantity'] = instance_info['quantity']
                        instances[asin]['name'] = instance_info['name']
                        bin_list = []
                        bin_list.append(i)
                        instances[asin]['bin_list'] = bin_list
    return instances


if __name__ == '__main__':
    metadata,n_images = get_metadata(img_dir, meta_dir)
    instances = get_instance_data(metadata)
    # dumping out all metadata into a file
    print("dumping metadata.json...")
    with open('metadata.json','w') as fp:
        json.dump(metadata,fp)
    print("dumping instances.json...")
    with open('instances.json','w') as fp:
        json.dump(instances,fp)
