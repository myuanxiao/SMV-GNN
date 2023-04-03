import copy
import torch
import random


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


def collate_fn(batch):
    """
    Collation function that collates datapoints into the batch format for cormorant

    Parameters
    ----------
    batch : list of datapoints
        The data to be collated.

    Returns
    -------
    batch : dict of Pytorch tensors
        The collated data.
    """

    batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}

    to_keep = (batch['charges'].sum(0) > 0)

    batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}

    atom_mask = batch['charges'] > 0
    batch['atom_mask'] = atom_mask

    #Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    #mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask

    #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    # Distance
    positions = batch['positions']
    charges = batch['charges']
    distance_set = torch.zeros(positions.shape[0], positions.shape[1], positions.shape[1])
    edge_mask_3d_set = torch.zeros((batch_size, n_nodes, n_nodes), dtype=torch.bool)
    shuff_charges_set = torch.zeros_like(charges)
    rationality_set = torch.zeros_like(charges)
    for i in range(positions.shape[0]):
      num_atom = 0
      node_pos = positions[i]
      for num in range(node_pos.shape[0]):
        if node_pos[num,0] == 0:
          break
        num_atom += 1
      node_repeated_in_chunks = node_pos.repeat_interleave(positions.shape[1], dim=0)
      node_repeated_alternating = node_pos.repeat(positions.shape[1], 1)
      dis = torch.sum((node_repeated_in_chunks -
                      node_repeated_alternating).pow(2), 1)
      dis_s = torch.sqrt(dis)
      dis = torch.reshape(torch.sqrt(dis), (positions.shape[1], positions.shape[1]))
      for a_num in range(dis.shape[0]):
        if a_num >= num_atom:
          dis[a_num,:] = 0
          dis[:,a_num] = 0
      # print("molecule {} distance shape is {}".format(i, dis.shape))
      distance_set[i, :, :] = dis

      # Knn neighbor
      inf_vec = torch.ones_like(dis) * 999
      dis = torch.where(dis==0, inf_vec, dis).numpy().tolist()
      neighbor_one_list = []
      edge_num = 5
      if num_atom <= 5:
        edge_num = int(num_atom/3)+1
      Inf = 999
      for i_m in range(num_atom):
        temp=[]
        atom_dis = dis[i_m]
        for i_t in range(edge_num):
          temp.append(atom_dis.index(min(atom_dis)))
          atom_dis[atom_dis.index(min(atom_dis))] = Inf
        neighbor_one_list.append(temp)
      for no_i, neighbor_one in enumerate(neighbor_one_list):
        for n_i, nei_one in enumerate(neighbor_one):
          edge_mask_3d_set[i, no_i, nei_one] = True

      # Shuffle
      shuffle_index = random.sample(range(num_atom), int(num_atom/2)+1)
      shuffle_list = []
      rationality = torch.zeros(charges.shape[1])
      for r_i in range(num_atom):
        rationality[r_i] = 1
      for sh_i in shuffle_index:
        shuffle_list.append(charges[i,sh_i].item())
      shuffle_list_raw = copy.deepcopy(shuffle_list)
      shuffle_list.insert(0,shuffle_list.pop())
      if shuffle_list_raw == shuffle_list:
        shuff_charges_set[i] = charges[i]
      else:
        shuff_charges = copy.deepcopy(charges[i])
        shuffle_index_raw = copy.deepcopy(shuffle_index)
        shuffle_index.insert(0,shuffle_index.pop())
        for shu_i, shuff_atom_index in enumerate(shuffle_index):
          rationality[shuff_atom_index] = 2
          shuff_charges[shuff_atom_index] = charges[i][shuffle_index_raw[shu_i]]
        shuff_charges_set[i] = shuff_charges
      rationality_set[i] = rationality

    batch['rationality'] = rationality_set
    batch['shuff_charges'] = shuff_charges_set
    batch['distance'] = distance_set
    batch['edge_mask_3d'] = edge_mask_3d_set.view(batch_size * n_nodes * n_nodes, 1)

    included_species = torch.tensor([1, 6, 7, 8, 9])
    batch['shuff_one_hot'] = batch['shuff_charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)



    return batch
