# import rdkit_installer
# rdkit_installer.install()
import os
import sys
import random
import string
import h5py
import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from time import time
import gzip

Symbol = ['C', 'N', 'O', 'F']
TotalHs = [0, 1, 2, 3, 4, 6]
Degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Hybridization = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
Bool_list = ['True', 'False']
ChiralTag = ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW']
ImplicitValence = [0, 1, 2, 3, 4, 5, 6]

Bond_type = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
Stereo = ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS']
ValenceContrib = [1.0, 1.5, 2.0, 3.0]

dim_node = 38
dim_edge = 12


def one_of_k_encoding_unk(x, allowable_set):
  if x not in allowable_set:
    x = allowable_set[-1]
  return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise ValueError("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))


def atomFeatures(atom):
  results = one_of_k_encoding_unk(
    atom.GetSymbol(),Symbol) + one_of_k_encoding(atom.GetDegree(),
                            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
            [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
            one_of_k_encoding_unk(atom.GetHybridization(), [
              Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
              Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
              Chem.rdchem.HybridizationType.SP3D2
            ]) + [atom.GetIsAromatic()]

  
  results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])

  try:
    results = results + one_of_k_encoding_unk(
        atom.GetProp('_CIPCode'),
        ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
  except:
    results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

  return np.array(results).astype(float)

def bondFeatures(bond):
  bt = bond.GetBondType()
  bond_feats = [
      bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
      bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
      bond.GetIsConjugated(),
      bond.IsInRing()
  ]
  bond_feats = bond_feats + one_of_k_encoding_unk(
      str(bond.GetStereo()), Stereo)
  return np.array(bond_feats).astype(float)


def normalize_adj(mx):
  """Row-normalize sparse matrix"""
  rowsum = mx.sum(-1)
  r_inv = torch.pow(rowsum, -1).flatten()
  r_inv[torch.isinf(r_inv)] = 0.
  r_inv = r_inv.reshape(mx.size(0),-1)

  r_mat_inv = torch.zeros_like(mx)
  for i in range(r_inv.size(0)):
    r_mat_inv[i] = torch.diag(r_inv[i], diagonal=0)
  mx = torch.matmul(r_mat_inv,mx)
  return mx

def get_adj(edge_root):
  edge_root = torch.from_numpy(np.array(edge_root)).sum(dim=3)
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
  for i in range(adj.size(0)):
    for j in range(0,atom_num[i]):
      adj[i,j,j] += 1
  adj = normalize_adj(adj)

  return adj


def get_2d_fea(smiles_list, n_nodes):
  max_atom = n_nodes
  batch_size = len(smiles_list)
  atom_set = np.zeros((batch_size, max_atom, dim_node), dtype=float)
  bond_set = np.zeros((batch_size, max_atom, max_atom, dim_edge), dtype=float)
  for i, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smiles)
    node = np.zeros((max_atom, dim_node), dtype=float)
    atoms = mol.GetAtoms()
    for j, atom in enumerate(atoms):
        atom = mol.GetAtomWithIdx(j)
        node[j, :] = atomFeatures(atom)
    atom_set[i] = node
    #print("molecule {} set done".format(i))

    # gen molecule edge dataset
    edge = np.zeros((max_atom, max_atom, dim_edge), dtype=float)
    n_atom = mol.GetNumHeavyAtoms()
    for j in range(n_atom - 1):
      for k in range(j + 1, n_atom):
        bond = mol.GetBondBetweenAtoms(j, k)
        if bond is not None:
          edge[j, k, :] = bondFeatures(bond)
          edge[k, j, :] = edge[j, k, :]
    bond_set[i] = edge
  adj = get_adj(bond_set)


  return torch.from_numpy(atom_set).float(), torch.from_numpy(bond_set).float(), adj