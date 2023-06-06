import torch
import deepchem as dc
import numpy as np
from deepchem.models.torch_models import MPNNModel
from deepchem.models import AttentiveFPModel

tasks, datasets, transformers = dc.molnet.load_sampl(splitter=None,featurizer="ECFP")
# scaffoldsplitter = dc.splits.RandomSplitter()
scaffoldsplitter = dc.splits.ScaffoldSplitter()
fold = scaffoldsplitter.k_fold_split(datasets[0],5)
avg_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
result = []
for fold_set in fold:
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    # training
    smiles = fold_set[0].ids
    labels = fold_set[0].y
    X = featurizer.featurize(smiles)
    del_list = []
    for xi in range(X.size):
        if (isinstance(X[xi], np.ndarray)):
            del_list.append(xi)
    X = np.delete(X,del_list)
    labels = np.delete(labels,del_list)
    dataset = dc.data.NumpyDataset(X=X, y=labels)
    model = AttentiveFPModel(mode='regression', n_tasks=1,
        batch_size=16, learning_rate=0.001)
    loss =  model.fit(dataset, nb_epoch=5)

    # test
    smiles = fold_set[1].ids
    labels = fold_set[1].y
    X = featurizer.featurize(smiles)
    del_list = []
    for xi in range(X.size):
        if (isinstance(X[xi], np.ndarray)):
            del_list.append(xi)
    X = np.delete(X,del_list)
    labels = np.delete(labels,del_list)
    dataset_test = dc.data.NumpyDataset(X=X, y=labels)
    test_scores = model.evaluate(dataset_test, [avg_rms], transformers)
    print("AttenFP:",test_scores)
    result.append(test_scores["mean-rms_score"])
    print(result,np.mean(result),np.std(result))

print(result,np.mean(result),np.std(result))