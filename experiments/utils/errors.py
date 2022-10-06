import numpy as np


def buildRMSPBE(X, P, R, D, gamma):
    # A = X.T.dot(D).dot(np.eye(X.shape[0]) - gamma * P).dot(X)
    # b = X.T.dot(D).dot(R)
    # C = X.T.dot(D).dot(X)

    # ------- light version of matrix multiplication ------- #
    # A = np.multiply(X.T, D).dot(np.eye(X.shape[0]) - gamma * P).dot(X)
    XTD = np.multiply(X.T, D)
    Mat_XP = np.zeros_like(XTD.dot(X))
    for a in P:
        i, j = int(a[0]), int(a[1])
        Mat_XP += gamma * P[a] * (XTD[:, [i]] * X[[j], :])
    A = XTD.dot(X) - Mat_XP
    b = XTD.dot(R)
    C = XTD.dot(X)

    Cinv = np.linalg.pinv(C)

    def RMSPBE(w):
        v = np.dot(-A, w) + b
        mspbe = v.T.dot(Cinv).dot(v)
        # ----- squared mspbe ------ #
        rmspbe = np.sqrt(mspbe)

        return rmspbe

    return RMSPBE


def buildJ_mu(d_mu, endFeature):
    # d_mu is stationary prob of endFeature
    def J_MU(w):
        j_mu = w.T.dot(endFeature) * d_mu
        return j_mu
    return J_MU
