import torch
import deepchem as dc
import numpy as np

tasks, datasets, transformers = dc.molnet.load_lipo(splitter=None)
# train_dataset, valid_dataset, test_dataset = datasets
# X_all = np.concatenate((train_dataset.X, valid_dataset.X),axis=0)
# X_all = np.concatenate((X_all, test_dataset.X),axis=0)
# y_all = np.concatenate((train_dataset.y, valid_dataset.y),axis=0)
# y_all = np.concatenate((y_all, test_dataset.y),axis=0)
# ids_all = np.concatenate((train_dataset.ids, valid_dataset.ids),axis=0)
# ids_all = np.concatenate((ids_all, test_dataset.ids),axis=0)
# Xs = np.zeros(len(ids_all))
# Ys = np.ones(len(ids_all))
# dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=Ys,w=np.zeros(len(ids_all)),ids=ids_all)
scaffoldsplitter = dc.splits.ScaffoldSplitter()
fold = scaffoldsplitter.k_fold_split(datasets[0],5)
avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
# result = []
# for set in fold:
#     train_dataset, test_dataset = set
#     featurizer = dc.feat.MolGraphConvFeaturizer()
#     train_x = featurizer.featurize(train_dataset.ids)
#     test_x = featurizer.featurize(test_dataset.ids)
#     train_set = dc.data.NumpyDataset(X=train_x, y=train_dataset.y)
#     test_set = dc.data.NumpyDataset(X=test_x, y=test_dataset.y)
#     # GCN
#     model = dc.models.GCNModel(mode='regression', n_tasks=1,
#                  batch_size=16, learning_rate=0.001)
#     loss = model.fit(train_set)
#     test_scores = model.evaluate(test_set, [avg_rms], transformers)
#     result.append(test_scores['mean-rms_score'])
#     print("GCN: ", test_scores)
    
# print(np.mean(result),np.std(result))

result = []
for set in fold:
    train_dataset, test_dataset = set
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    train_x = featurizer.featurize(train_dataset.ids)
    test_x = featurizer.featurize(test_dataset.ids)
    train_set = dc.data.NumpyDataset(X=train_x, y=train_dataset.y)
    test_set = dc.data.NumpyDataset(X=test_x, y=test_dataset.y)
    print(train_set)
    # AttentiveFPModel
    model = dc.models.AttentiveFPModel(mode='regression', n_tasks=1,
                 batch_size=16, learning_rate=0.001)
    loss = model.fit(train_set, nb_epoch=5)
    test_scores = model.evaluate(test_set, [avg_rms], transformers)
    result.append(test_scores['mean-rms_score'])
    print("AttentiveFPModel: ", test_scores)
    
print(np.mean(result),np.std(result))
