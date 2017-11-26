import sys
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

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

def elasticnet(X_trn, X_test, Y_trn, Y_test):
    clf = ElasticNet(alpha=0.1)
    clf.fit(X_trn, Y_trn)
    Y_test_hat = clf.predict(X_test)
    return Rsq(Y_test, Y_test_hat)

def lasso(X_trn, X_test, Y_trn, Y_test):
    clf = Lasso(alpha=0.1)
    clf.fit(X_trn, Y_trn)
    Y_test_hat = clf.predict(X_test)
    return Rsq(Y_test, Y_test_hat)

def rf(X_trn, X_test, Y_trn, Y_test):
    regr = RandomForestRegressor()
    regr.fit(X_trn, Y_trn)
    Y_test_hat = regr.predict(X_test)
    return Rsq(Y_test, Y_test_hat)

def ridge(X_trn, X_test, Y_trn, Y_test):
    clf = Ridge()
    clf.fit(X_trn, Y_trn)
    Y_test_hat = clf.predict(X_test)
    return Rsq(Y_test, Y_test_hat)

def sgd(X_trn, X_test, Y_trn, Y_test):
    clf = SGDRegressor()
    clf.fit(X_trn, Y_trn)
    Y_test_hat = clf.predict(X_test)
    return Rsq(Y_test, Y_test_hat)

def svrlin(X_trn, X_test, Y_trn, Y_test):
    clf = SVR(kernel='linear')
    clf.fit(X_trn, Y_trn)
    Y_test_hat = clf.predict(X_test)
    return Rsq(Y_test, Y_test_hat)

def svrrbf(X_trn, X_test, Y_trn, Y_test):
    clf = SVR(kernel='rbf')
    clf.fit(X_trn, Y_trn)
    Y_test_hat = clf.predict(X_test)
    return Rsq(Y_test, Y_test_hat)

functionDict = {'elasticnet' : elasticnet,
        'lasso' : lasso,
        'rf' : rf,
        'ridge' : ridge,
        'sgd' : sgd,
        'svrlin' : svrlin,
        'svrrbf' : svrrbf,
        'all' : None}


if len(sys.argv) != 2:
    print "Please specify a regressor from the following list:"
    print functionDict.keys()
    sys.exit()
funcStr = sys.argv[1]
if funcStr not in functionDict:
    print "Please specify a regressor from the following list:"
    print functionDict.keys()
    sys.exit()
func = functionDict[funcStr]
X_trn, X_test, Y_trn, Y_test = loadData()
if func is not None:
    print funcStr
    print func(X_trn, X_test, Y_trn, Y_test)
else:
    for name, func in functionDict.items():
        if func is None:
            continue
        print name
        print func(X_trn, X_test, Y_trn, Y_test)
