import csv
import torch
import deepchem as dc
import numpy as np
from sklearn.metrics import roc_auc_score

result = []
for i in range (3):
    # Load dataset with default 'scaffold' splitting
    tasks, datasets, transformers = dc.molnet.load_lipo(featurizer="ECFP")
    # tasks
    # train_dataset, valid_dataset, test_dataset = datasets
    with open("./train_lipo.csv") as file_name:
        train_dataset_list = list(csv.reader(file_name))

    featurizer=dc.feat.MolecularFeaturizer()
    train_dataset = featurizer.featurize(train_dataset_list)
    print(train_dataset)

    # We want to know the pearson R squared score, averaged across tasks
    # avg_pearson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
    roc_auc = dc.metrics.Metric(roc_auc_score)

    # We'll train a multitask regressor (fully connected network)
    model = dc.models.MultitaskClassifier(
    len(tasks),
    n_features=1024,
    layer_sizes=[500])

    model.fit(train_dataset)

    # We now evaluate our fitted model on our training and validation sets
    train_scores = model.evaluate(train_dataset, [roc_auc], transformers)
    print(train_scores,type(train_scores['roc_auc_score']))
    valid_scores = model.evaluate(valid_dataset, [roc_auc], transformers)
    print(valid_scores)
    test_scores = model.evaluate(test_dataset, [roc_auc], transformers)
    print(test_scores)
    result.append(test_scores['roc_auc_score'])

print(result,np.mean(result),np.std(result))