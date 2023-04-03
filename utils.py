import os
import matplotlib
matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
 
def kl_graph(embd, n_node, n_nf):
    embd = embd.view(-1,n_node,n_nf) # 2,3,4
    graph = 1/n_node * torch.sum(embd,dim=1) #2,4
    #print(embd[0][0], graph[0])
    repeat_graph = graph.unsqueeze(1).repeat(1,n_node,1) #2,3,4
    #print(repeat_graph[0],a[0])
    norm_embd = torch.pow(torch.norm(repeat_graph - embd, p=2, dim=[2]), 2) + 1
    norm_embd = torch.pow(norm_embd, -1)
    #print(norm_embd)
    #print(norm_embd.shape)
    sum_embd = torch.sum(norm_embd,dim=1).unsqueeze(-1).repeat(1,n_node) #2,3 -> 2 -> 2,3
    #print(sum_embd)
    #print(sum_embd.shape)
    div_embd = norm_embd / sum_embd
    #print(div_embd)
    #print(div_embd.shape)
    return div_embd
    #embd_2d = embs_2d.view(-1, n_node, n_nf)
    #embd_3d = embd_3d.view(-1, n_node, n_nf)
    #graph_2d = 1/n_nodes * torch.sum(embd_2d, dim=1)  
    #graph_3d = 1/n_nodes * torch.sum(embd_3d, dim=1)
    #repeate_graph_2d = graph_2d.unsqueeze(1).repeate(1,n_node,1)
    #repeate_graph_3d = graph_3d.unsqueeze(1).repeate(1,n_node,1)
    #div_2d = torch.pow(troch.pow(repeate_graph_2d - embd_2d, 2)+1, -1)
    #div_3d = torch.pow(troch.pow(repeate_graph_3d - embd_2d, 2)+1, -1)
    #sum_2d = torch.sum(div_2d, dim=1) 
    #sum_3d = torch.sum(div_3d, dim=1)



def create_folders(args):
    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name)
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name + '/images_recon')
    except OSError:
        pass

    try:
        os.makedirs(args.outf + '/' + args.exp_name + '/images_gen')
    except OSError:
        pass

def makedir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass

def normalize_res(res, keys=[]):
    for key in keys:
        if key != 'counter':
            res[key] = res[key] / res['counter']
    del res['counter']
    return res

def plot_coords(coords_mu, path, coords_logvar=None):
    if coords_mu is None:
        return 0
    if coords_logvar is not None:
        coords_std = torch.sqrt(torch.exp(coords_logvar))
    else:
        coords_std = torch.zeros(coords_mu.size())
    coords_size = (coords_std ** 2) * 1

    plt.scatter(coords_mu[:, 0], coords_mu[:, 1], alpha=0.6, s=100)


    #plt.errorbar(coords_mu[:, 0], coords_mu[:, 1], xerr=coords_size[:, 0], yerr=coords_size[:, 1], linestyle="None", alpha=0.5)

    plt.savefig(path)
    plt.clf()

def filter_nodes(dataset, n_nodes):
    new_graphs = []
    for i in range(len(dataset.graphs)):
        if len(dataset.graphs[i].nodes) == n_nodes:
            new_graphs.append(dataset.graphs[i])
    dataset.graphs = new_graphs
    dataset.n_nodes = n_nodes
    return dataset

def adjust_learning_rate(optimizer, epoch, lr_0, factor=0.5, epochs_decay=100):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_0 * (factor ** (epoch // epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
