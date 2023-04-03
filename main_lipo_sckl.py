from torch.nn.functional import fold
from qm9 import dataset
from qm9.models import EGNN, PredictionKL, GNN, SSL_3d, SSL_2d
from data_scaffold_lipo import GetData, GetValData, GetTestData
from torch.utils.data import Dataset, DataLoader
from leaky_mib import Solver

#from qm9 import get_2d_fea
import torch
from torch import nn, optim
import argparse
from qm9 import utils as qm9_utils
from time import time
import utils
import json
import glob
import os
import numpy as np

parser = argparse.ArgumentParser(description='QM9 Example')
parser.add_argument('--exp_name', type=str, default='sckl', metavar='N',
                    help='experiment_name')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--alpha', type=float, default=1.0, metavar='N',
                    help='weight of loss_2d')
parser.add_argument('--beta', type=float, default=0.1, metavar='N',
                    help='weight of loss_3d')
parser.add_argument('--eta', type=float, default=0.003, metavar='N',
                    help='weight of loss1 in KL')
parser.add_argument('--mu', type=float, default=0.01, metavar='N',
                    help='weight of loss2 in KL')
parser.add_argument('--lr', type=float, default=0.0003, metavar='N',
                    help='learning rate')
parser.add_argument('--fold', type=int, default=0, metavar='N',
                    help='learning rate')
parser.add_argument('--knn', type=int, default=10, metavar='N',
                    help='k of knn')
parser.add_argument('--nf', type=int, default=128, metavar='N',
                    help='learning rate')
parser.add_argument('--z_dim', type=int, default=128, metavar='N',
                    help='dim of z feature')
parser.add_argument('--outf', type=str, default='lipo/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--weight_decay', type=str, default=1e-18, metavar='N',
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=1, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--attention', type=int, default=1, metavar='N',
                    help='attention in the ae model')
parser.add_argument('--n_layers', type=int, default=7, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--property', type=str, default='homo', metavar='N',
                    help='label to predict: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve')
parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                    help='number of workers for the dataloader')
parser.add_argument('--charge_power', type=int, default=2, metavar='N',
                    help='maximum power to take into one-hot features')
parser.add_argument('--dataset_paper', type=str, default="cormorant", metavar='N',
                    help='cormorant, lie_conv')
parser.add_argument('--node_attr', type=int, default=0, metavar='N',
                    help='node_attr or not')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
dtype = torch.float32
print(args)
print(device)

utils.makedir(args.outf)
utils.makedir(args.outf + "/" + args.exp_name)

start_time = time()
# dataloaders, charge_scale = dataset.retrieve_dataloaders(args.batch_size, args.num_workers)

data_all = GetData(args.fold, args.knn)
dataloaders_train = DataLoader(data_all, batch_size = args.batch_size, shuffle = True, drop_last=False, num_workers=2)

data_valid = GetValData(args.fold, args.knn)
dataloaders_valid = DataLoader(data_valid, batch_size = 8, shuffle = True, drop_last=False, num_workers=2)

data_test = GetTestData(args.fold, args.knn)
dataloaders_test = DataLoader(data_test, batch_size = 10, shuffle = True, drop_last=False, num_workers=2)

print('Dataloaders time:', time()-start_time)


model = EGNN(in_node_nf=73, in_edge_nf=0, hidden_nf=args.nf, device=device,
                 n_layers=args.n_layers, coords_weight=1.0, attention=args.attention, node_attr=args.node_attr)

model_2d = GNN(input_dim=73, hidden_nf=args.nf, edges_in_nf=18, device=device, act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False)

ssl_3d = SSL_3d(hidden_nf=args.nf, device=device)

ssl_2d = SSL_2d(hidden_nf=args.nf, device=device)

prediction = PredictionKL(in_node_nf=73, in_nf=2*args.nf, hidden_nf=args.nf, device=device,
                 n_layers=args.n_layers, coords_weight=1.0, attention=args.attention, node_attr=args.node_attr)
kl_solver = Solver(args, device=device)

weight_decay = torch.tensor(float(args.weight_decay))
optimizer = optim.Adam(
          [{'params': model.parameters()},
          {'params': model_2d.parameters()},
          {'params': ssl_2d.parameters()},
          {'params': ssl_3d.parameters()},
          {'params': kl_solver.encoder_H2d.parameters()},
          {'params': kl_solver.encoder_H3d.parameters()},
          {'params': kl_solver.mi_estimator1.parameters()},
          {'params': kl_solver.mi_estimator2.parameters()},
          {'params': prediction.parameters()}],
          lr=args.lr,
          weight_decay=weight_decay)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
loss_l1 = nn.L1Loss()
loss_mse = nn.MSELoss()
loss_ce = nn.CrossEntropyLoss()


def train(epoch, loader, partition='train'):
    lr_scheduler.step()
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    if partition=='train': loader = dataloaders_train
    else: loader = dataloaders_valid
    for batch, data in enumerate(loader):

        n_nodes = data["node_fea"].size(1) # 23
        labels = data["labels"].to(device, dtype)
        batch_size = labels.size(0)
        rationality = data["rational"].to(device).long()
        distance = data["distance"].to(device, dtype)

        edge_mask_2d = data["edge_mask_2d"].view(batch_size * n_nodes * n_nodes, -1).to(device, dtype)
        shuff_nodes = data["node_shuffle_fea"].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_2d = data["edge_fea"].view(batch_size * n_nodes * n_nodes, -1).to(device, dtype)

        nodes = data["node_fea"].view(batch_size * n_nodes, -1).to(device, dtype)
        atom_positions = data["positions"].view(batch_size * n_nodes, -1).to(device, dtype)
        edge_mask = data["edge_mask_3d"].view(batch_size * n_nodes * n_nodes, -1).to(device, dtype)
        atom_mask = data["atom_mask"].to(device, dtype).view(batch_size * n_nodes, -1).to(device, dtype)

        rows, cols = [], []
        for batch_idx in range(batch_size):
            for i_node in range(n_nodes):
                for j_node in range(n_nodes):
                    rows.append(i_node + batch_idx*n_nodes)
                    cols.append(j_node + batch_idx*n_nodes)
        edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]

        if partition == 'train':
            model.train()#.half()
            model_2d.train()#.half()
            ssl_2d.train()#.half()
            kl_solver.train()
            prediction.train()#.half()
            ssl_3d.train()#.half()
            optimizer.zero_grad()

        else:
            model.eval()#.half()
            model_2d.eval()#.half()
            kl_solver.eval()
            ssl_2d.eval()#.half()
            prediction.eval()#.half()
            ssl_3d.eval()#.half()
            optimizer.zero_grad()


        embd_3d = model(h0=nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)

        embd_3d_ssl = model(h0=shuff_nodes, x=atom_positions, edges=edges, edge_attr=None, node_mask=atom_mask, edge_mask=edge_mask,
                     n_nodes=n_nodes)

        embd_2d = model_2d(nodes=nodes, edges=edges, edge_mask=edge_mask_2d, edge_attr=edge_2d)

        distance_pre = ssl_2d(h=embd_2d.view(batch_size, n_nodes, -1), node_mask=atom_mask, n_nodes=n_nodes)
        ration_pre = ssl_3d(h = embd_3d_ssl, node_mask=atom_mask, n_nodes=n_nodes)

        H_2d = torch.sum((embd_2d*atom_mask).view(-1,n_nodes,args.nf),dim=1) # B,N,F->B,F
        H_3d = torch.sum((embd_3d*atom_mask).view(-1,n_nodes,args.nf),dim=1) # B,N,F->B,F
        kl_loss, Z_2d, Z_3d = kl_solver._compute_loss(H_2d, H_3d)
        
        fusion_embs = torch.cat((Z_2d.view(-1,args.nf), Z_3d.view(-1,args.nf)),dim=1)
        pred = prediction(h=fusion_embs)


        if partition == 'train':
            loss = loss_mse(pred, labels.squeeze(1))
            loss_ssl_3d = loss_ce(ration_pre.view(-1,3), rationality.view(-1,1).squeeze(1))
            loss_ssl_2d = loss_mse(distance_pre, distance)
            loss_sum = loss + args.alpha*loss_ssl_2d + args.beta*loss_ssl_3d + kl_loss
            loss_sum.backward()
            optimizer.step()
        else:
            loss = loss_mse(pred, labels.squeeze(1))


        res['loss'] += torch.sqrt(loss).item() * batch_size
        res['counter'] += batch_size
        res['loss_arr'].append(torch.sqrt(loss).item())

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        print(prefix + "Epoch %d \t Iteration %d \t rmse %.4f \t knn %d \t alpha %f \t beta %f \t lr %f" % (epoch, batch, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:]), args.knn, args.alpha, args.beta, args.lr))
    return res['loss'] / res['counter']



if __name__ == "__main__":

    best_result = 1000
    res = {'args': str(args), 'epochs': [], 'train_loss': [], 'val_loss': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

    for epoch in range(args.epochs):
        train_loss = train(epoch, dataloaders_train, partition='train')

        if epoch % 5 == 0: #args.test_interval == 0:
            val_loss = train(epoch, dataloaders_valid, partition='valid')
            res['epochs'].append(epoch)
            res['train_loss'].append(train_loss)
            res['val_loss'].append(val_loss)

            if val_loss < res['best_val']:
                res['best_val'] = val_loss
                res['best_epoch'] = epoch
                test_loss = train(epoch, dataloaders_test, partition='test')
                res['best_test'] = test_loss
            print("Val loss: %.4f  \t epoch %d" % (val_loss, epoch))
            print("Best: val loss: %.4f  \t epoch %d" % (res['best_val'], res['best_epoch']))
            print("Best: test loss: %.4f  \t epoch %d" % (res['best_test'], res['best_epoch']))


        json_object = json.dumps(res, indent=4)
        pref = args.outf + "/" + args.exp_name + "/"+ str(args.fold) + "/" + str(args.lr) + "/" + str(args.knn) \
                + "/" + str(args.weight_decay) + "/" + str(args.alpha) + "_" + str(args.beta) + "_" + str(args.mu) + "_" +str(args.eta)  
        if not os.path.exists(pref):
            os.makedirs(pref) 
        with open(pref + "/losess.json", "w") as outfile:
            outfile.write(json_object)
