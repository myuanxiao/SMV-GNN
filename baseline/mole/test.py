import numpy as np
import deepchem as dc
featurizer = dc.feat.WeaveFeaturizer()
X = featurizer(["C", "CC"])
y = np.array([1, 0])
dataset = dc.data.NumpyDataset(X, y)
model = dc.models.WeaveModel(n_tasks=1, n_weave=2, fully_connected_layer_sizes=[2000, 1000], mode="classification")
loss = model.fit(dataset)