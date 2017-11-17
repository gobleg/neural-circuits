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

def CreateData(inputSize, numDataPoints):
    X = np.atleast_2d(np.array([]))
    Y = np.array([])
    for i in range(numDataPoints):
        x = np.atleast_2d(np.random.randint(2, size=inputSize))
        y = OrCircuit(x.ravel())
        if i == 0:
            X = x
        else:
            X = np.append(X, x, axis=0)
        Y = np.append(Y, y)
    return X, Y

X, Y = CreateData(32, 10000)
np.savez('data.npz', X, Y)
