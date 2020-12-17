

# Mounting on google drive


from google.colab import drive
drive.mount('/gdrive')
!ls
# !tar -zxvf CUB_200_2011.tgz  (for downloading dataset,Mounting is better since it is a large dataset)

batch_size = 28

# Imports

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset,DataLoader
from collections import OrderedDict

"""Connecting to GPU"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Dataset Class"""

import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset


class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transforms.Compose([transforms.Resize(255), 
                                       transforms.CenterCrop(224),  
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(), 
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 
        self.loader = default_loader
        self.train = train
        self._load_metadata()

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img']) 

        data = images.merge(image_class_labels, on='img_id')

        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

train_dataset = Cub2011('/gdrive/My Drive',train = True)
test_dataset  = Cub2011('/gdrive/My Drive',train = False)

len(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=28)
batch = next(iter(train_loader))

test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = 28)

batch[0].size()