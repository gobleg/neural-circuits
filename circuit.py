import sys
import os
import numpy as np

def OrCircuit(X):
    n = X.shape[0]
    if n == 1:
        return X
    X1 = np.array([])
    for i in range(0, n, 2):
        X1 = np.append(X1, X[i] or X[i+1])
    return AndCircuit(X1)

def AndCircuit(X):
    n = X.shape[0]
    if n == 1:
        return X
    X1 = np.array([])
    for i in range(0, n, 2):
        X1 = np.append(X1, X[i] and X[i+1])
    return OrCircuit(X1)

def SumCircuit(X):
    return np.sum(X)

def AdderCircuit(X):
    n = X.shape[0]
    X1 = X[0:n/2]
    X2 = X[n/2:n]
    x1 = np.packbits(X1)
    x1 = 256*x1[0] + x1[1]
    x2 = np.packbits(X2)
    x2 = 256*x2[0] + x2[1]
    return x1 + x2

def SemiSum(X):
    n = X.shape[0]
    X1 = np.array([])
    for i in range(0, n, 2):
        X1 = np.append(X1, X[i] or X[i+1])
    return np.sum(X1)

def CreateData(inputSize, numDataPoints, circuitFunction):
    X = np.atleast_2d(np.array([]))
    Y = np.array([])
    for i in range(numDataPoints):
        x = np.atleast_2d(np.random.randint(2, size=inputSize))
        y = circuitFunction(x.ravel())
        if i == 0:
            X = x
        else:
            X = np.append(X, x, axis=0)
        Y = np.append(Y, y)
    return X, Y

functionDict = {'or' : OrCircuit,
                'sum' : SumCircuit,
                'adder' : AdderCircuit,
                'semisum' : SemiSum}

if len(sys.argv) != 2:
    print "Please specify a circuit from the following list:"
    print functionDict.keys()
    sys.exit()
funcStr = sys.argv[1]
if funcStr not in functionDict:
    print "Please specify a circuit from the following list:"
    print functionDict.keys()
    sys.exit()
func = functionDict[funcStr]
X, Y = CreateData(32, 10000, func)
if os.path.exists('data.npz'):
    os.remove('data.npz')
np.savez('data.npz', X, Y)
