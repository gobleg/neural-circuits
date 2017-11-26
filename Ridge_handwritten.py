import numpy as np

def ridge(x, y, lam):
    n_features = x.shape[1]
    beta_ridge = np.linalg.inv(x.T.dot(x) + lam * np.eye(n_features)).dot(x.T).dot(y)
    return beta_ridge

Rsq = lambda a, a_hat: 1 - (a - a_hat).var() / a.var()

def cv_ridge(x, y, x_test, y_test):
    lambdas = np.logspace(1, 7, 15) # use these lambdas

    n_mc_iters = 50 # let's do 50 Monte Carlo iterations
    n_per_mc_iter = 50 # on each MC iteration, hold out 50 datapoints to be the validation set
    num_training = x.shape[0]

    val_mses = np.zeros((num_training, len(lambdas)))

    for it in range(n_mc_iters):
        indices = np.random.choice(num_training, n_per_mc_iter, replace=True)

        x_trn = np.delete(x, indices, 0)
        y_trn = np.delete(y, indices, 0)

        x_val = x[indices]
        y_val = y[indices]

        for ii in range(len(lambdas)):
            y_val_hat = np.dot(x_val, ridge(x_trn, y_trn, lambdas[ii]))
            val_mses[it,ii] = np.mean((y_val_hat - y_val) ** 2)

    lambda_avg = np.mean(val_mses, axis=0)
    best_lambda = lambdas[np.argmin(lambda_avg)]
    beta = ridge(x, y, best_lambda)
    y_test_hat = np.dot(x_test, beta)
    return Rsq(y_test, y_test_hat)

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
print cv_ridge(X_trn, Y_trn, X_test, Y_test)
