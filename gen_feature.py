import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdBase,Draw
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
from rdkit.Chem.Scaffolds import rdScaffoldNetwork, MurckoScaffold

import deepchem as dc
import numpy as np
import h5py
import random
import argparse
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

featurizer = dc.feat.WeaveFeaturizer(
    graph_distance=True, 
    explicit_H = True, 
    use_chirality = True, 
    max_pair_distance = 1
    )
tasks_lipo, datasets_lipo,\
 transformers_lipo = dc.molnet.load_lipo(featurizer=featurizer, splitter='scaffold')
train_dataset, valid_dataset, test_dataset = datasets_lipo
max = 0
lent = 0
out = []
train_len, valid_len, test_len = 0, 0, 0
for split_i, data in enumerate([train_dataset, valid_dataset, test_dataset]):
  temp_X = data.X
  temp_y = data.y
  temp_smi = data.ids
  out = []
  for i, smi in enumerate(temp_smi):
    mol = Chem.MolFromSmiles(smi)
    num_heav = mol.GetNumHeavyAtoms()
    if num_heav > 60:
      out.append(i)
      continue
    lent += 1
    if num_heav > max :
      max = num_heav
  out.reverse()
  print(len(temp_X))
  print(len(temp_X)-len(out))
  if split_i == 0 : train_len = len(temp_X)-len(out)
  if split_i == 1 : valid_len = len(temp_X)-len(out)
  if split_i == 2 : test_len = len(temp_X)-len(out)

max_num_atoms = max
data_all = [train_dataset, valid_dataset, test_dataset]
for split, split_data in enumerate(data_all):
  print(split, split_data)

  X = split_data.X
  y = split_data.y
  smiles = split_data.ids
  if split == 0 : fold_len = train_len
  if split == 1 : fold_len = valid_len
  if split == 2 : fold_len = test_len

  lipo_atom_test3D = np.zeros((fold_len,max_num_atoms,76))
  lipo_atom_test_shuffle = np.zeros((fold_len,max_num_atoms,76))
  lipo_bond_test = np.zeros((fold_len,max_num_atoms,max_num_atoms,18))
  lipo_y_test = np.zeros((fold_len,1))
  lipo_rational_test = np.zeros((fold_len,max_num_atoms))


  index = -1
  for i in range(fold_len):

    smi = smiles[i]
    print(smi)
    mol = Chem.MolFromSmiles(smi)
    num_heav = mol.GetNumHeavyAtoms()
    if num_heav > max_num_atoms:
      out.append(i)
      continue
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
    try:
      pos_x = [mol.GetConformer().GetAtomPosition(0).x]
    except: continue
    index += 1
    atoms = mol.GetAtoms()
    num_heav = mol.GetNumHeavyAtoms()
    node_feat3D = np.zeros((max_num_atoms,76))
    node_shuffle = np.zeros((max_num_atoms,76))
    rationality_one = np.zeros((max_num_atoms))
    rationality_one[0:num_heav] = 1
    atom_list = []
    for j, atom in enumerate(atoms):
      if j >= num_heav: break
      atom_list.append(atom.GetSymbol())
      raw_feat = X[i].get_atom_features()[j]
      pos_x = [mol.GetConformer().GetAtomPosition(j).x]
      pos_y = [mol.GetConformer().GetAtomPosition(j).y]
      pos_z = [mol.GetConformer().GetAtomPosition(j).z]
      one_node_feat = np.array(list(raw_feat)+pos_x+pos_y+pos_z)
      node_feat3D[j] = one_node_feat
    lipo_atom_test3D[index] = node_feat3D
    
    atom_list_raw = copy.deepcopy(atom_list)
    shuffle_index = random.sample(range(num_heav), int(num_heav/2)+1)
    temp = atom_list[shuffle_index[0]]
    for si in range(len(shuffle_index)):
      atom_list[shuffle_index[si]] = \
      atom_list[shuffle_index[(si+1) % len(shuffle_index)]]
    atom_list[shuffle_index[-1]] = temp
    if atom_list_raw==atom_list:
      node_shuffle = node_feat3D
    else:
      for sh_i, sh_index in enumerate(shuffle_index):
        rationality_one[sh_index] = 2
      node_shuffle = copy.deepcopy(node_feat3D)
      temp = node_feat3D[shuffle_index[0]]
      for si in range(len(shuffle_index)):
        node_shuffle[shuffle_index[si]][:-3] = \
        node_shuffle[shuffle_index[(si+1) % len(shuffle_index)]][:-3]
      node_shuffle[shuffle_index[-1]][:-3] = temp[:-3]
    lipo_atom_test_shuffle[index] = node_shuffle
    lipo_rational_test[index] = rationality_one

    bond_feature = X[i].get_pair_features()
    pair_edges = X[i].get_pair_edges()
    for jj in range(pair_edges.shape[1]):
      lipo_bond_test[index,pair_edges[0,jj],pair_edges[1,jj]] = \
      bond_feature[jj]
      lipo_bond_test[index,pair_edges[1,jj],pair_edges[0,jj]] = \
      bond_feature[jj]
    lipo_y_test[index] = y[i]

  with h5py.File('../egnn-main/lipo_scaffold/'+str(split)+'/node_lipo.h5', 'w') as hf:
    hf.create_dataset('elem', data=lipo_atom_test3D,
                      compression='gzip', compression_opts=9)
  with h5py.File('../egnn-main/lipo_scaffold/'+str(split)+'/node_shuffle_lipo.h5', 'w') as hf:
    hf.create_dataset('elem', data=lipo_atom_test_shuffle,
                      compression='gzip', compression_opts=9)
  with h5py.File('../egnn-main/lipo_scaffold/'+str(split)+'/bond_lipo.h5', 'w') as hf:
    hf.create_dataset('elem', data=lipo_bond_test,
                      compression='gzip', compression_opts=9)
  with h5py.File('../egnn-main/lipo_scaffold/'+str(split)+'/y_lipo.h5', 'w') as hf:
    hf.create_dataset('elem', data=lipo_y_test,
                      compression='gzip', compression_opts=9)
  with h5py.File('../egnn-main/lipo_scaffold/'+str(split)+'/rational_lipo.h5', 'w') as hf:
    hf.create_dataset('elem', data=lipo_rational_test,
                      compression='gzip', compression_opts=9)
