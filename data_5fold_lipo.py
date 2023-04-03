import os
import glob
import time
import math
import h5py
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


def get_tensor(dir_path):
    with h5py.File(dir_path, 'r') as hf:
      data_raw = hf['elem'][:]
    return torch.from_numpy(data_raw).float()

def get_long_tensor(dir_path):
    with h5py.File(dir_path, 'r') as hf:
      data_raw = hf['elem'][:]
    return torch.from_numpy(data_raw).long()


class GetData(Dataset):
    def __init__(self, fold, knn):

      init_train_time = time.time()
      fold_list = [1, 2, 3, 4, 5]
      fold_list.pop(fold)
      train_fold = fold_list
      valid_fold = [fold+1]
      
      self.node_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/node_lipo.h5')[:,:,:-3] for f in train_fold], dim = 0)
      self.positions = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/node_lipo.h5')[:,:,-3:] for f in train_fold], dim = 0)
      self.node_shuffle_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/node_shuffle_lipo.h5')[:,:,:-3] for f in train_fold], dim = 0)
      self.edge_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/bond_lipo.h5') for f in train_fold], dim = 0)
      self.edge_3d_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/edge_feature_3d_'+str(knn)+'nn_lipo.h5').unsqueeze(-1) for f in train_fold], dim = 0)
      self.labels_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/y_lipo.h5') for f in train_fold], dim = 0)
      self.rational_raw = torch.cat([get_long_tensor('./dataset/lipo5fold/'+str(f)+'/rational_lipo.h5') for f in train_fold], dim = 0)
      self.distance_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/distance_lipo.h5') for f in train_fold], dim = 0)
      self.edge_mask_2d, self.atom_mask = self.get_adj(self.edge_raw)
      self.edge_mask_3d,  _ = self.get_adj(self.edge_3d_raw)

      print('init_train_time', time.time() - init_train_time)



    def __getitem__(self, index):
      data = {}
      data["node_fea"] = self.node_raw[index]
      data["positions"] = self.positions[index]
      data["node_shuffle_fea"] = self.node_shuffle_raw[index]
      data["edge_fea"] = self.edge_raw[index]
      # data["edge_3d_fea"] = self.edge_3d_raw[index]
      data["labels"] = self.labels_raw[index]
      data["rational"] = self.rational_raw[index]
      data["distance"] = self.distance_raw[index]
      data["edge_mask_2d"] = self.edge_mask_2d[index]
      data["edge_mask_3d"] = self.edge_mask_3d[index]
      data["atom_mask"] = self.atom_mask[index]

      return data

    def __len__(self):
      return len(self.node_raw)
      
    def get_shape(self):
      return self.node_raw.shape, self.labels_raw.shape

    def get_adj(self, edge_root):
      adj_time = time.time()
      edge_root = edge_root.sum(dim=3)
      zero_vec = torch.zeros_like(edge_root)
      one_vec = torch.ones_like(edge_root)
      adj = torch.where(edge_root > 0, one_vec, zero_vec)
      del zero_vec, one_vec

      atom_num = []
      for i in range(adj.size(0)):
        adj_sum = adj[i].sum(0)
        for j in range(adj.size(1)):
          num = j
          if adj_sum[j] == 0:
            break
        atom_num.append(num)
      atom_mask = torch.ones(edge_root.size(0),edge_root.size(1))
      for i in range(atom_mask.size(0)):
        atom_mask[i,atom_num[i]:] = 0

      del atom_num
      print('adj_time', time.time() - adj_time)
      return adj, atom_mask

class GetValData(Dataset):
    def __init__(self, fold, knn):

      init_train_time = time.time()
      fold_list = [1, 2, 3, 4, 5]
      train_fold = fold_list.pop(fold)
      valid_fold = [fold+1]
      
      self.node_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/node_lipo.h5')[:,:,:-3] for f in valid_fold], dim = 0)
      self.positions = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/node_lipo.h5')[:,:,-3:] for f in valid_fold], dim = 0)
      self.node_shuffle_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/node_shuffle_lipo.h5')[:,:,:-3] for f in valid_fold], dim = 0)
      self.edge_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/bond_lipo.h5') for f in valid_fold], dim = 0)
      self.edge_3d_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/edge_feature_3d_'+str(knn)+'nn_lipo.h5').unsqueeze(-1) for f in valid_fold], dim = 0)
      self.labels_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/y_lipo.h5') for f in valid_fold], dim = 0)
      self.rational_raw = torch.cat([get_long_tensor('./dataset/lipo5fold/'+str(f)+'/rational_lipo.h5') for f in valid_fold], dim = 0)
      self.distance_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/distance_lipo.h5') for f in valid_fold], dim = 0)
      self.edge_mask_2d, self.atom_mask = self.get_adj(self.edge_raw)
      self.edge_mask_3d,  _ = self.get_adj(self.edge_3d_raw)

      print('init_train_time', time.time() - init_train_time)



    def __getitem__(self, index):
      data = {}
      data["node_fea"] = self.node_raw[index]
      data["positions"] = self.positions[index]
      data["node_shuffle_fea"] = self.node_shuffle_raw[index]
      data["edge_fea"] = self.edge_raw[index]
      # data["edge_3d_fea"] = self.edge_3d_raw[index]
      data["labels"] = self.labels_raw[index]
      data["rational"] = self.rational_raw[index]
      data["distance"] = self.distance_raw[index]
      data["edge_mask_2d"] = self.edge_mask_2d[index]
      data["edge_mask_3d"] = self.edge_mask_3d[index]
      data["atom_mask"] = self.atom_mask[index]

      return data

    def __len__(self):
      return len(self.node_raw)
      
    def get_shape(self):
      return self.node_raw.shape, self.labels_raw.shape

    def get_adj(self, edge_root):
      adj_time = time.time()
      edge_root = edge_root.sum(dim=3)
      zero_vec = torch.zeros_like(edge_root)
      one_vec = torch.ones_like(edge_root)
      adj = torch.where(edge_root > 0, one_vec, zero_vec)
      del zero_vec, one_vec

      atom_num = []
      for i in range(adj.size(0)):
        adj_sum = adj[i].sum(0)
        for j in range(adj.size(1)):
          num = j
          if adj_sum[j] == 0:
            break
        atom_num.append(num)
      atom_mask = torch.ones(edge_root.size(0),edge_root.size(1))
      for i in range(atom_mask.size(0)):
        atom_mask[i,atom_num[i]:] = 0

      del atom_num
      print('adj_time', time.time() - adj_time)
      return adj, atom_mask


class GetTestData(Dataset):
    def __init__(self, fold, knn):

      init_train_time = time.time()
      test_fold = ['test']
      
      self.node_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/node_lipo.h5')[:,:,:-3] for f in test_fold], dim = 0)
      self.positions = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/node_lipo.h5')[:,:,-3:] for f in test_fold], dim = 0)
      self.node_shuffle_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/node_shuffle_lipo.h5')[:,:,:-3] for f in test_fold], dim = 0)
      self.edge_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/bond_lipo.h5') for f in test_fold], dim = 0)
      self.edge_3d_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/edge_feature_3d_'+str(knn)+'nn_lipo.h5').unsqueeze(-1) for f in test_fold], dim = 0)
      self.labels_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/y_lipo.h5') for f in test_fold], dim = 0)
      self.rational_raw = torch.cat([get_long_tensor('./dataset/lipo5fold/'+str(f)+'/rational_lipo.h5') for f in test_fold], dim = 0)
      self.distance_raw = torch.cat([get_tensor('./dataset/lipo5fold/'+str(f)+'/distance_lipo.h5') for f in test_fold], dim = 0)
      self.edge_mask_2d, self.atom_mask = self.get_adj(self.edge_raw)
      self.edge_mask_3d,  _ = self.get_adj(self.edge_3d_raw)

      print('init_train_time', time.time() - init_train_time)



    def __getitem__(self, index):
      data = {}
      data["node_fea"] = self.node_raw[index]
      data["positions"] = self.positions[index]
      data["node_shuffle_fea"] = self.node_shuffle_raw[index]
      data["edge_fea"] = self.edge_raw[index]
      # data["edge_3d_fea"] = self.edge_3d_raw[index]
      data["labels"] = self.labels_raw[index]
      data["rational"] = self.rational_raw[index]
      data["distance"] = self.distance_raw[index]
      data["edge_mask_2d"] = self.edge_mask_2d[index]
      data["edge_mask_3d"] = self.edge_mask_3d[index]
      data["atom_mask"] = self.atom_mask[index]

      return data

    def __len__(self):
      return len(self.node_raw)
      
    def get_shape(self):
      return self.node_raw.shape, self.labels_raw.shape

    def get_adj(self, edge_root):
      adj_time = time.time()
      edge_root = edge_root.sum(dim=3)
      zero_vec = torch.zeros_like(edge_root)
      one_vec = torch.ones_like(edge_root)
      adj = torch.where(edge_root > 0, one_vec, zero_vec)
      del zero_vec, one_vec

      atom_num = []
      for i in range(adj.size(0)):
        adj_sum = adj[i].sum(0)
        for j in range(adj.size(1)):
          num = j
          if adj_sum[j] == 0:
            break
        atom_num.append(num)
      atom_mask = torch.ones(edge_root.size(0),edge_root.size(1))
      for i in range(atom_mask.size(0)):
        atom_mask[i,atom_num[i]:] = 0

      del atom_num
      print('adj_time', time.time() - adj_time)
      return adj, atom_mask
