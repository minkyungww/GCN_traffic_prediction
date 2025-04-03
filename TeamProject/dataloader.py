from utils import *

import pandas as pd
import numpy as np
import torch
import random

import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch_geometric.data import Data

def fix_seed(seed_value = 42):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed_value)

class CustomDataset(Dataset): 
  def __init__(self, **kwargs):

    self.del_feature = None

    for key, value in kwargs.items():
        setattr(self, key, value)

    self.df_T = make_time_series_data(self.df, T = self.T, t = self.t)
    self.edge_index = make_edge_relationship(df = self.df_T)
    self.dataloader_1 = make_dataloader(self.df_T, edge_index=self.edge_index, device = self.device, T = self.T, del_feature = self.del_feature)

    self.df_add_T = self.df_add[self.T:-self.t]
    if self.full_edges:
       self.edge_index_add = [[i, j] for i in range(33) for j in range(33)]
    else:
       self.edge_index_add = torch.load('./data/address_relations.pt') # it was preprocessed
    self.dataloader_2 = self.make_dataloader_add(self.df_add_T, self.edge_index_add, del_feature = self.del_feature)

  def del_feature_add(self, x, d, del_feature):
    dict_ = {
       'rain' : torch.arange(0, 1),
       'temp' : torch.arange(1, 2), 
       'humid' : torch.arange(2, 3),
       'wind' : torch.arange(3, 4)
    }

    if del_feature not in dict_.keys():
        d = 1
        return x, d
    else:
        d = 2
        x[:, dict_[del_feature]] = torch.randn(x.size(0), 1).to(self.device)
        return x, d
  
  def make_dataloader_add(self, df_add_T, edge_index_add, del_feature):
    dataloader = []
    d = 0
    for i in range(df_add_T.shape[0]):
      x = torch.Tensor(df_add_T[i]).to(self.device)
      if del_feature is not None:
         x, d = self.del_feature_add(x, d, del_feature)

      edge_index = torch.tensor(edge_index_add, dtype=torch.long).to(self.device)

      data = Data(x=x, edge_index=edge_index.t().contiguous())
      dataloader.append(data)
    
    if d == 1:
        # print(f'Feature deletion did not work on dataloader_2')
    if d == 2:
        # print(f'Feature deletion did work on {del_feature}')
    return dataloader

  def __len__(self): 
    return len(self.dataloader_1)

  def __getitem__(self, idx): 
    x = self.dataloader_1[idx]
    y = self.dataloader_2[idx]
    return x, y
  

def make_dataset(type_, T, t, del_feature, device):
    if type_ == 'train':
        df_train = pd.read_parquet('./data/df_train.parquet', engine = 'pyarrow')
        df_train_add = np.load('./data/df_train_add.npy')

        train_dataset = CustomDataset(df = df_train, 
                                    df_add = df_train_add, 
                                    T = T, 
                                    t = t, 
                                    full_edges = False,
                                    del_feature = del_feature,
                                    device = device
                                    )
        return train_dataset
    
    if type_ == 'val':
        df_val = pd.read_csv('./data/df_val.csv', index_col=0)
        df_val_add = np.load('./data/df_val_add.npy')

        val_dataset = CustomDataset(df = df_val, 
                                    df_add = df_val_add, 
                                    T = T, 
                                    t = t, 
                                    full_edges = False,
                                    del_feature = del_feature,
                                    device = device
                                    )
        return val_dataset
    
    if type_ == 'test':
        df_test = pd.read_csv('./data/df_test.csv', index_col=0)
        df_test_add = np.load('./data/df_test_add.npy')

        test_dataset = CustomDataset(df = df_test, 
                                    df_add = df_test_add, 
                                    T = T, 
                                    t = t, 
                                    full_edges = False,
                                    del_feature = del_feature,
                                    device = device
                                    )
        return test_dataset
