import torch
import deepchem as dc
import numpy as np
from sklearn.metrics import mean_squared_error

result = []
for i in range (3):
    # Load dataset with default 'scaffold' splitting
    tasks, datasets, transformers = dc.molnet.load_sampl(featurizer="ECFP")
    tasks
    train_dataset, valid_dataset, test_dataset = datasets
    # print(train_dataset, valid_dataset, test_dataset)

    # We want to know the pearson R squared score, averaged across tasks
    avg_pearson_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
    rmse = dc.metrics.Metric(mean_squared_error, np.sqrt)

    # We'll train a multitask regressor (fully connected network)
    model = dc.models.MultitaskRegressor(
    len(tasks),
    n_features=1024,
    layer_sizes=[500])

    model.fit(train_dataset)

    # We now evaluate our fitted model on our training and validation sets
    train_scores = model.evaluate(train_dataset, [avg_pearson_r2,rmse], transformers)
    print(train_scores,type(train_scores['sqrt-mean_squared_error']))
    valid_scores = model.evaluate(valid_dataset, [avg_pearson_r2,rmse], transformers)
    print(valid_scores)
    test_scores = model.evaluate(test_dataset, [avg_pearson_r2,rmse], transformers)
    print(test_scores)
    result.append(test_scores['sqrt-mean_squared_error'])

print(result,np.mean(result),np.std(result))