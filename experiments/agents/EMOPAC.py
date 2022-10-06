import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils.policies import parameterToPolicies, parameterToPolicies_tab, mapStateIndex, mapActionIndex


class EMOPAC:
    def __init__(self, features, params, lmda):
        self.features = features
        self.params = params

        self.gamma = params['gamma']
        self.alpha = params['alpha']
        self.alpha_theta = params['alpha_theta']
        self.beta = params['beta']
        self.eta = params.get('eta', 1)
        self.lmda = lmda
        self.Ft = 0
        # ---- rho_{-1} = 0 --- #
        self.rho = 0
        self.NUM_ACTORS = 9
        self.THETA_DIM = 18
        # -------- mixing matrix for 9 agents -------- #
        # self.C = connectionMatrix(np.array([[1.0, 1.0, 0, 1.0, 0, 0, 0, 0, 0],
        #                                     [1.0, 1.0, 1.0, 0, 1.0, 0, 0, 0, 0],
        #                                     [0, 1.0, 1.0, 0, 0, 1.0, 0, 0, 0],
        #                                     [1.0, 0, 0, 1.0, 1.0, 0, 1.0, 0, 0],
        #                                     [0, 1.0, 0, 1.0, 1.0, 1.0, 0, 1.0, 0],
        #                                     [0, 0, 1.0, 0, 1.0, 1.0, 0, 0, 1.0],
        #                                     [0, 0, 0, 1.0, 0, 0, 1.0, 1.0, 0],
        #                                     [0, 0, 0, 0, 1.0, 0, 1.0, 1.0, 1.0],
        #                                     [0, 0, 0, 0, 0, 1.0, 0, 1.0, 1.0]
        #                                     ]))

        self.C = connectionMatrix(np.array([[1.0, 1.0, 0, 0, 0, 0, 0, 0, 1.0],
                                            [1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0],
                                            [0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0],
                                            [0, 0, 1.0, 1.0, 1.0, 0, 0, 0, 0],
                                            [0, 0, 0, 1.0, 1.0, 1.0, 0, 0, 0],
                                            [0, 0, 0, 0, 1.0, 1.0, 1.0, 0, 0],
                                            [0, 0, 0, 0, 0, 1.0, 1.0, 1.0, 0],
                                            [0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0],
                                            [1.0, 0, 0, 0, 0, 0, 0, 1.0, 1.0]
                                            ]))

        # ------ fully connected graph ------ #
        # self.C = connectionMatrix(np.ones(shape=(self.NUM_ACTORS, self.NUM_ACTORS)))

        # features is the dimension of the feature vector
        # self.w = np.zeros(features); 8 agents
        self.e_theta = np.zeros(self.THETA_DIM)
        self.eligibility = np.zeros(self.features)
        # self.w = np.random.normal(-4, 2, size=(self.NUM_ACTORS, features))
        # --- initial w's of the agents are different --- #
        self.w = np.zeros(shape=(self.NUM_ACTORS, features))
        for i in range(self.NUM_ACTORS):
            # self.w[i] = np.random.normal((-1) ** (i + 1) * (i + 2), 1, size=(1, features))
            # self.w[i] = np.random.normal((-1) ** (i + 1), 1, size=(1, features))
            self.w[i] = np.random.normal(-15, 15, size=(1, features))


        # self.w = np.random.normal(-4, 4, size=(self.NUM_ACTORS, features))
        self.h = np.zeros(shape=(self.NUM_ACTORS, features))
        # self.theta = np.zeros(shape=(2, 5))
        self.theta = np.zeros(shape=(self.NUM_ACTORS, self.THETA_DIM))

    # -- corresponds to the step's arguments in utils/rl_glue.py---#
    def update(self, x, s, a, r, target_para, behavior, xp, step):
        #  here rho depends on theta
        target_tab = parameterToPolicies_tab(target_para, self.theta)
        NUM_AGENT = len(r)
        target = parameterToPolicies(target_para, self.theta)
        # rho = self.target.ratio(self.behavior, self.s, self.a)
        rho = 1
        for i in range(NUM_AGENT):
            rho *= target[i].ratio(behavior, s[i], a[i])
        # self.eligibility = rho * (x + self.gamma * self.lmda * self.eligibility)
        # ---------------- eligibility traces ---------------- #
        Mt_theta = 1 + self.lmda * self.gamma * self.rho * self.Ft
        self.Ft = self.rho * self.gamma * self.Ft + 1
        Mt = (1 - self.lmda) * self.Ft + self.lmda
        self.eligibility = Mt * x + self.gamma * self.lmda * self.eligibility
        v = self.w.dot(x)
        vp = self.w.dot(xp)
        delta = np.zeros(NUM_AGENT)
        for i in range(NUM_AGENT):
            delta[i] = r[i] + self.gamma * vp[i] - v[i]
        # delta_joint = np.mean(delta)
        # delta_joint = np.sum(r) + self.gamma * np.mean(vp) - np.mean(v)
        # delta_joint = np.sum(r) + self.gamma * np.sum(vp) - np.sum(v)
        # delta_hat = self.h.dot(x)
        # delta_he = self.h.dot(self.eligibility)
        delta_hat = self.h.dot(x)
        # s1 = mapStateIndex(s)
        # a1 = mapActionIndex(a)
        gradlog = np.zeros_like(self.theta)
        for i in range(NUM_AGENT):
            # ----- compute gradient ------ #
            gradlog[i] = target_para[s[i], a[i]] - (target_tab[i, s[i], a[i]] * target_para[s[i], a[i]] +
                                                    target_tab[i, s[i], 1 - a[i]] * target_para[s[i], 1 - a[i]])
        # -------- update parameters according to (9) and (8) (algorithm) -----------#
        dw = np.zeros(shape=[NUM_AGENT, self.features])
        # dh = np.zeros(shape=[NUM_AGENT, self.features])
        dtheta = np.zeros(shape=[NUM_AGENT, self.THETA_DIM])
        for i in range(NUM_AGENT):
            dw[i] = rho * delta[i] * self.eligibility
            dtheta[i] = rho * Mt_theta * gradlog[i] * delta[i]
            # dtheta[i] = rho * Mt_theta * gradlog[i] * delta_joint
            # self.e_theta = rho * (Mt * gradlog[i] + self.gamma * self.lmda * self.e_theta)

        # ------ updates before communication ----- #
        # wbc = self.w + self.alpha * dw
        # hbc = self.h + self.eta * self.alpha * dh
        wbc = self.w + self.alpha / (step+1)**(5/8) * dw
        # hbc = self.h + self.eta * self.alpha / (step+1)**(9/16) * dh
        self.w = np.zeros_like(self.w)
        # self.h = np.zeros_like(self.h)
        for i in range(NUM_AGENT):
            for j in range(NUM_AGENT):
                self.w[i] += self.C[i][j] * wbc[j]
                # self.h[i] += self.C[i][j] * hbc[j]

        # self.w[0], self.w[1] = 1/2 * ((self.w[0] + self.alpha * dw[0]) + (self.w[1] + self.alpha * dw[1])), \
        #                        1/2 * ((self.w[0] + self.alpha * dw[0]) + (self.w[1] + self.alpha * dw[1]))
        # self.theta = self.theta + self.alpha_theta * 0.01 * dtheta
        self.theta = self.theta + self.alpha_theta * 0.001 / (step+1) * dtheta
        self.rho = (rho + rho) / 2
        # return self.theta

    def getWeights(self):
        return self.w


def connectionMatrix(adjointMat):
    n = len(adjointMat)
    sum_row = np.sum(adjointMat, axis=1)
    for i in range(n):
        for j in range(n):
            if adjointMat[i, j] != 0:
                adjointMat[i, j] = 1 / sum_row[i]
    for i in range(n):
        for j in range(n):
            if adjointMat[i, j] > adjointMat[j, i]:
                adjointMat[i, j] = adjointMat[j, i]
    for i in range(n):
        adjointMat[i, i] = 1 - np.sum(adjointMat[i]) + adjointMat[i, i]

    return adjointMat
