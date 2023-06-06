import torch
import csv
import deepchem as dc
import numpy as np
import pandas as pd

# with open("./train_lipo.csv") as file_name:
#     train_dataset_list = list(csv.reader(file_name))
# for data in train_dataset_list:
     # print(data[1:])
# print(train_dataset_list[1][1:])
tasks, datasets, transformers = dc.molnet.load_lipo(splitter=None,featurizer="ECFP")
scaffoldsplitter = dc.splits.ScaffoldSplitter()
fold = scaffoldsplitter.train_valid_test_split(datasets[0])
df0 = {'smiles':fold[0].ids, 'y':[token[0] for token in fold[0].y]}
df0 = pd.core.frame.DataFrame(df0)
df1 = {'smiles':fold[1].ids, 'y':[token[0] for token in fold[1].y]}
df1 = pd.core.frame.DataFrame(df1)
df2 = {'smiles':fold[2].ids, 'y':[token[0] for token in fold[2].y]}
df2 = pd.core.frame.DataFrame(df2)
print(df0)

df0.to_csv("./train_lipo.csv", header=None, index=None)
df1.to_csv("./valid_lipo.csv", header=None, index=None)
df2.to_csv("./test_lipo.csv", header=None, index=None)

# with open("valid_lipo.csv", "w", encoding="utf-8") as f:
#      writer = csv.writer(f)
#      writer.writerow(fold[1].ids)
#      f.close()
# with open("test_lipo.csv", "w", encoding="utf-8") as f:
#      writer = csv.writer(f)
#      writer.writerow(fold[2].ids)
#      f.close()
# avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
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

# result = []
# for set in fold:
#     train_dataset, test_dataset = set
#     featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
#     train_x = featurizer.featurize(train_dataset.ids)
#     test_x = featurizer.featurize(test_dataset.ids)
#     train_set = dc.data.NumpyDataset(X=train_x, y=train_dataset.y)
#     test_set = dc.data.NumpyDataset(X=test_x, y=test_dataset.y)
#     print(train_set)
#     # MPNNModel
#     model = dc.models.torch_models.MPNNModel(mode='regression', n_tasks=1,
#                  batch_size=16, learning_rate=0.001)
#     loss = model.fit(train_set, nb_epoch=5)
#     test_scores = model.evaluate(test_set, [avg_rms], transformers)
#     result.append(test_scores['mean-rms_score'])
#     print("MPNNModel: ", test_scores)
    
# print(np.mean(result),np.std(result))


# result = []
# for set in fold:
#     train_dataset, test_dataset = set
#     featurizer = dc.feat.WeaveFeaturizer()
#     train_x = featurizer.featurize(train_dataset.ids)
#     test_x = featurizer.featurize(test_dataset.ids)
#     train_set = dc.data.NumpyDataset(X=train_x, y=train_dataset.y)
#     test_set = dc.data.NumpyDataset(X=test_x, y=test_dataset.y)
#     print(train_set)
#     # WeaveModel
#     model = dc.models.graph_models.WeaveModel(n_tasks=1, n_weave=2, 
#                         fully_connected_layer_sizes=[2000, 1000], mode="regression")
#     loss = model.fit(train_set, nb_epoch=5)
#     test_scores = model.evaluate(test_set, [avg_rms], transformers)
#     result.append(test_scores['mean-rms_score'])
#     print("WeaveModel: ", test_scores)
    
# print(np.mean(result),np.std(result))


