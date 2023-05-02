import os

import numpy as np
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data import Dataset


class MyDs(Dataset):
    def __init__(self, data_path, pos_files, neg_files, tfm=None):
        self.data_path = data_path
        self.pos_files = pos_files
        self.neg_files = neg_files
        self.tfm = tfm

    def __len__(self):
        return len(self.pos_files) + len(self.neg_files)

    def __getitem__(self, i):
        if i < len(self.pos_files):
            pf = self.data_path / 'def_front' / self.pos_files[i]
            lbl = 1
        else:
            pf = self.data_path / 'ok_front' / self.neg_files[i - len(self.pos_files)]
            lbl = 0
        image = Image.open(pf)
        if self.tfm is not None:
            image = self.tfm(image)
        return image, lbl
    
def create_datasets(data_path):
    pos_files = sorted(os.listdir(data_path / 'def_front'))
    neg_files = sorted(os.listdir(data_path / 'ok_front'))
    np.random.seed(0)
    np.random.shuffle(pos_files)
    np.random.shuffle(neg_files)
    _N = int(len(pos_files) * 0.8)
    trn_pos_files, val_pos_files = pos_files[:_N], pos_files[_N:]
    _N = int(len(neg_files) * 0.8)
    trn_neg_files, val_neg_files = neg_files[:_N], neg_files[_N:]
    _normalize = TF.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tfm = TF.Compose([TF.Resize((256,256)), TF.ToTensor(), _normalize])
    trn_ds = MyDs(data_path, trn_pos_files, trn_neg_files, tfm=tfm)
    val_ds = MyDs(data_path, val_pos_files, val_neg_files, tfm=tfm)
    return trn_ds, val_ds