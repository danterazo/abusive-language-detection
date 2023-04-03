import numpy as np
from joblib import Parallel, delayed
from sklearn.svm import LinearSVC


def train_model(X, y, seed):
    model = LinearSVC(random_state=seed)
    return model.fit(X, y)


X = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([0, 1])
result = Parallel(n_jobs=4)(delayed(train_model)(X, y, seed) for seed in range(10))

for r in result:
    print(r)
