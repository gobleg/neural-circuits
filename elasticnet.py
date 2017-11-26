import numpy as np
from sklearn.linear_model import ElasticNet

Rsq = lambda a, a_hat: 1 - (a - a_hat).var() / a.var()

def loadData():
    npzfile = np.load('data.npz')
    X = npzfile['arr_0']
    Y = npzfile['arr_1']
    n = X.shape[0]
    n_trn = int(n * 0.8)
    X_trn = X[0:n_trn, :]
    X_test = X[n_trn:n, :]
    Y_trn = Y[0:n_trn]
    Y_test = Y[n_trn:n]
    return X_trn, X_test, Y_trn, Y_test

X_trn, X_test, Y_trn, Y_test = loadData()
clf = ElasticNet()
clf.fit(X_trn, Y_trn)
Y_test_hat = clf.predict(X_test)
print Rsq(Y_test, Y_test_hat)
