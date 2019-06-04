import os
import re
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision as tv


def ICIAR_loader():
    ''' 
    split dataset into train-set and valid-set
    valid-set : [091 ~ 100].tif for each class
    '''
    train_dataset = ICIAR_dataset(phase='train')
    valid_dataset = ICIAR_dataset(phase='valid')
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    return train_loader, valid_loader


class ICIAR_dataset(Dataset):
    def __init__(self, phase='train'):
        super(ICIAR_dataset, self).__init__()
        self.parent_dir = os.path.join('..','..','ICIAR2018_BACH_Challenge','NormPhotos')
        self.csv_file = os.path.join(self.parent_dir,'microscopy_ground_truth.csv')
            
        self.input_transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
        ])  
            
        self.dataset_tmp = []
        self.get_dataset_tmp(phase)
            
            
    def __getitem__(self, index):
        file, cls = self.dataset_tmp[index]
        filepath = os.path.join(self.parent_dir, cls, file)
            
        input = load_image(filepath)
        input = self.input_transform(input)
        target = load_target(cls)
           
        return input, target
            
        
    def __len__(self):
        return len(self.dataset_tmp)

    
    def get_dataset_tmp(self, phase):
        with open(self.csv_file,'r') as f:
            for line in f:
                file, cls, _ = re.split(',|\n',line)
                if phase=='train' and is_file_in_train_set(file):
                    self.dataset_tmp.append((file, cls))
                elif phase=='valid' and not is_file_in_train_set(file):
                    self.dataset_tmp.append((file, cls))
    

def is_file_in_train_set(file):
    # file = 'n001.tif' --> number = 1
    number = int(file[-7:-4])
    if number >=1 and number<=90:
        return True
    else:
        return False

def load_image(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def load_target(cls):
    # (Normal, Benign, InSitu, Invasive) --> (0,1,2,3)
    target = torch.tensor(0)
    if cls == 'Normal':
        target = torch.tensor(0)
    elif cls == 'Benign':
        target = torch.tensor(1)
    elif cls == 'InSitu':
        target = torch.tensor(2)
    elif cls == 'Invasive':
        target = torch.tensor(3)
    else:
        raise Exception("class should be in ['Normal', 'Benign', 'InSitu', 'Invasive']")
    return target
