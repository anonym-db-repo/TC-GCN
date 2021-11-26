import numpy
import numpy as np
import scipy.signal as sps
from collections import deque

eps = np.finfo(float).eps

def normalize(a, order=2, axis=-1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def embed_data(x, order, lag):
    ch, N = x.shape
    hidx = np.arange(order * lag, step=lag)
    Nv = N - (order - 1) * lag
    u = np.zeros((order*ch, Nv))
    for i in range(order):
        u[i*ch:(i+1)*ch] = x[:, hidx[i]:hidx[i]+Nv]

    return u.T


def pTE(z, lag=1, model_order=1, to_norm=False):
    """Returns pseudo transfer entropy.

    Parameters
    ----------
    z : array
        array of arrays, containing all the time series.
    lag : integer
        delay of the embedding.
    model_order : integer
        embedding dimension, or model order.
    surr : boolean
        if True it computes the maximum value obtained using 19 times shifted
        surrogates

    Returns
    -------
    pte : array
        array of arrays. The dimension is (# time series, # time series).
        The diagonal is 0, while the off diagonal term (i, j) corresponds
        to the pseudo transfer entropy from time series i to time series j.
    """

    NN, C, T = np.shape(z)
    pte = np.zeros((NN, NN))
    if to_norm:
        z = normalize(sps.detrend(z))
    nodes = np.arange(NN, step=1)

    for i in nodes:
        EmbdDumm = embed_data(z[i], model_order + 1, lag)
        Xtau = EmbdDumm[:, :-C]
        for j in nodes:
            if i != j:
                Yembd = embed_data(z[j], model_order + 1, lag)
                Y = Yembd[:, -C:]
                Ytau = Yembd[:, :-C]
                XtYt = np.concatenate((Xtau, Ytau), axis=1)
                YYt = np.concatenate((Y, Ytau), axis=1)
                YYtXt = np.concatenate((YYt, Xtau), axis=1)

                if model_order > 1 or C > 1:
                    ptedum = np.linalg.det(np.cov(XtYt.T)) * np.linalg.det(np.cov(YYt.T)) / \
                             (np.linalg.det(np.cov(YYtXt.T)) * np.linalg.det(np.cov(Ytau.T)) + eps)
                else:
                    ptedum = np.linalg.det(np.cov(XtYt.T)) * np.linalg.det(np.cov(YYt.T)) / \
                             (np.linalg.det(np.cov(YYtXt.T)) * np.cov(Ytau.T) + eps)

                if ptedum > 0:
                    pte[i, j] = 0.5 * np.log(ptedum)

    return pte

#
# cov = numpy.array(
#     [[1.0, 0.5, 0.5, 0.2],
#      [0.5, 1.0, 0.5, 0.1],
#      [0.5, 0.5, 1.0, 0.3],
#      [0.2, 0.1, 0.3, 1.0]])
#
# z = np.random.multivariate_normal([1, 2, 3, 4], cov, 3000)
# cov_m = np.cov(z.T)
#
# z = np.reshape(z.T, (2, 2, -1))
# causal_matrix = pTE(z, model_order=2, to_norm=True)
# print(causal_matrix)
