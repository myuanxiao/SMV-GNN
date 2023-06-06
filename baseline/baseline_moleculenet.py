import argparse
import csv
import deepchem as dc
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--graph_conv_layers", type=int,  default=[64,64], help="graph_conv_layers")
parser.add_argument("--batch_size", type=int, default=16,  help="batch_size")
parser.add_argument("--dropout", type=float,  default=0.0, help="dropout")
parser.add_argument("--predictor_hidden_feats", type=int,  default=128, help="predictor_hidden_feats")
parser.add_argument("--predictor_dropout", type=float,  default=0.0, help="predictor_dropout")
parser.add_argument("--number_atom_features", type=int,  default=30, help="number_atom_features")
parser.add_argument("--learning_rate", type=float,  default=0.01, help="learning_rate")
args = parser.parse_args()

avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
valid_result = []
test_result = []


with open("./train_lipo.csv") as file_name:
    train_dataset_list = list(csv.reader(file_name))
with open("./valid_lipo.csv") as file_name:
    valid_dataset_list = list(csv.reader(file_name))
with open("./test_lipo.csv") as file_name:
    test_dataset_list = list(csv.reader(file_name))
featurizer = dc.feat.MolGraphConvFeaturizer()
train_x = featurizer.featurize([smiles[0] for smiles in train_dataset_list])
valid_x = featurizer.featurize([smiles[0] for smiles in valid_dataset_list] )
test_x = featurizer.featurize([smiles[0] for smiles in test_dataset_list] )
train_set = dc.data.NumpyDataset(X=train_x, y = [float(label[1]) for label in train_dataset_list])
valid_set = dc.data.NumpyDataset(X=test_x, y = [float(label[1]) for label in valid_dataset_list])
test_set = dc.data.NumpyDataset(X=test_x, y = [float(label[1]) for label in test_dataset_list])

for seed in range(3):

    # GCN
    model = dc.models.GCNModel(mode='regression', n_tasks=1,
                 batch_size=args.batch_size,
                 learning_rate=args.learning_rate,
                 dropout=args.dropout,
                 predictor_hidden_feats=args.predictor_hidden_feats,
                 predictor_dropout=args.predictor_dropout,
                 number_atom_features=args.number_atom_features)
    loss = model.fit(train_set)
    valid_scores = model.evaluate(valid_set, [avg_rms])
    test_scores = model.evaluate(test_set, [avg_rms])
    valid_result.append(valid_scores['valid_mean-rms_score'])
    test_result.append(test_scores['test_mean-rms_score'])
    print("GCN valid: ", valid_scores)
    print("GCN test: ", test_scores)

print(args)
print(np.mean(valid_result),np.std(valid_result))
print(np.mean(test_result),np.std(test_result))

