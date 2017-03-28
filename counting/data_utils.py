import torch.utils.data as data
import torch

from PIL import Image
import numpy as np
import os
import os.path
import json

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolderCounting(data.Dataset):
    def __init__(self, root, list_file, transform=None, target_transform=None,
                 loader=default_loader):
        print('loading data_list json file...')
        with open(list_file) as f:
          data_list = json.loads(f.read())
        print('finished loading data_list json file')
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.data_list = data_list
        self.N = len(self.data_list)
    
    def __getitem__(self, index):
        # pick up the image
        item = self.data_list[index]
        target = int(item[1])
        img_path = '%05d.jpg' % (item[0]+1)
        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        return img, target
    
    def __len__(self):
        return self.N
