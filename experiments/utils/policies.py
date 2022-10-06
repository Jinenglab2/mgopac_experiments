from .random import sampleFromDist
import numpy as np


# wraps a function which takes a state and returns a list of probabilities for each action
# helps maintain a consistent API even if policies are generated in different ways
class Policy:
    def __init__(self, probs, NUM_ACTORS):
        self.probs = probs
        self.NUM_ACTORS = NUM_ACTORS

    def selectAction(self, s):
        # np.random.seed(2021)
        # --- !! probs is pi_\theta (can be softmax and so on) -----#
        # action_probabilities = [self.probs(s[i]) for i in range(len(s))]

        # * in behavior policy, each agent i selects its action independently (s[i]) !!! * #
        action_probabilities = [self.probs(s[i]) for i in range(self.NUM_ACTORS)]
        # ------- multiple actors -------- #
        return [sampleFromDist(ap) for ap in action_probabilities]

    # --------- for computing importance weighting pi^i/mu_i --------- #
    def ratio(self, other, s, a):
        # ---------??? other means another policy??? yes---------- #
        # (i1, i2) ---> (j)
        # s1 = mapStateIndex(s)
        # probs = [self.probs(ss) for ss in s]
        # a1 = mapActionIndex(a)
        # ------- other.probs is the behavior policy for ONE agent ------ #
        # behaviorpolicy_prob = 1
        # targetpolicy_prob = 1
        # for i, ss in enumerate(s):
        # -------- compute pi^i_theta / mu^i -------- #
        behaviorpolicy_prob = other.probs(s)[a]
        targetpolicy_prob = self.probs(s)[a]
        return targetpolicy_prob / behaviorpolicy_prob

# ----- [s1, s2] ---> s ----- #
def mapStateIndex(s):
    if len(s) == 2:
        mapTabular = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        s1 = mapTabular[s[0]][s[1]]
        return s1

# ----- [a1, a2] ---> a ----- #
def mapActionIndex(a):
    if len(a) == 2:
        mapTabularA = [[0, 1], [2, 3]]
        a1 = mapTabularA[a[0]][a[1]]
        return a1

def matrixToPolicy(probs, NUM_ACTORS):
    probsmap = lambda s: probs[s]
    return Policy(probsmap, NUM_ACTORS)


def actionArrayToPolicy(probs, NUM_ACTORS):
    return Policy(lambda s: probs, NUM_ACTORS)


def softmax(x):
    # !!! input x is an array; axis=-1,
    # e = np.exp(x - np.amax(x, keepdims=True))
    e = np.zeros_like(x)
    for i in range(len(x)):
        e[i] = np.exp(x[i] - np.amax(x[i], keepdims=True))
        e[i] = e[i] / np.sum(e[i])
    # [e[i] / np.sum(e[i], keepdims=True) for i in range(len(e))]
    return e

# ------ create policies of number len(theta) ------- #
def parameterToPolicies(target_parameters, theta):
    NUM_ACTORS = theta.shape[0]
    NUM_STATE, NUM_ACTION, _ = target_parameters.shape
    probs = np.zeros(shape=(NUM_ACTORS, NUM_STATE, NUM_ACTION))
    for i in range(NUM_ACTORS):
        probs[i] = softmax(np.dot(target_parameters, theta[i], out=None))
    # probs = softmax(np.dot(target_parameters, theta, out=None))
    # ----- states 1, 2, 3, 6 do not appear in the process ------- #
    # probs[1] = 0
    # probs[2] = 0
    # probs[3] = 0
    # probs[6] = 0
    # return probs
    # --- return the joint policy based on joint state!!! (9 x 4) --- #
    return [Policy(lambda s: probs[i][s], NUM_ACTORS) for i in range(NUM_ACTORS)]


def parameterToPolicies_tab(target_parameters, theta):

    NUM_ACTORS = theta.shape[0]
    NUM_STATE, NUM_ACTION, _ = target_parameters.shape
    probs = np.zeros(shape=(NUM_ACTORS, NUM_STATE, NUM_ACTION))
    for i in range(NUM_ACTORS):
        probs[i] = softmax(np.dot(target_parameters, theta[i], out=None))
    return probs
    # return Policy(lambda s: probs[s])
