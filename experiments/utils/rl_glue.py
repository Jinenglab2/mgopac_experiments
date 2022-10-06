import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from RlGlue.agent import BaseAgent

# ties together a learning algorithm, a behavior policy, a target policy, and a representation
# stores the previous state and representation of that state for 1-step bootstrapping methods (all methods in this project)
# this wrapper allows easy API compatibility with RLGlue, while allowing the learning code to remain small and simple (the complexity due to bookkeeping is pushed into this class)
class RlGlueCompatWrapper(BaseAgent):
    # ----- corresponds to Agent's argument in figure1.py (target_para replaces target)------#
    def __init__(self, learner, behavior, target_para, representation, num_actors):
        self.learner = learner
        self.behavior = behavior
        # self.target = target
        self.target_para = target_para
        self.theta = None
        # ------ here representation is the transferred rep.encode ------- #
        self.representation = representation
        self.NUM_ACTORS = num_actors
        self.s = None
        self.a = None

        # because representations are always deterministic in this project
        # just store the past representation to reduce compute
        self.x = None

    # called on the first step of the episode
    def start(self, s):
        self.s = s
        self.x = self.representation(s)
        # ----- self.a is joint actor of multiple actors ------- #
        self.a = self.behavior.selectAction(s)
        return self.a

    # def softmax(x):
    #     e = np.exp(x - np.amax(x, axis=-1, keepdims=True))
    #     return e / np.sum(e, axis=-1, keepdims=True)

    # tf.Variable()
    # def
    #     actor_variables = tf.random.normal(shape=(state_size, action_size), dtype=tf.float64), dtype=tf.float64)

    def step(self, r, s, step):
        # ---------- here attain the feature vector (phi(s)) --------- #
        xp = self.representation(s)
        # target = parameterToPolicy(self.target_para)
        # # rho = self.target.ratio(self.behavior, self.s, self.a)
        # rho = target.ratio(self.behavior, self.s, self.a)
        # #qsa = target_para(self.s, self.a)
        self.learner.update(self.x, self.s, self.a, r, self.target_para, self.behavior, xp, step)
        self.s = s
        # ------ joint actor of dimenstion self.NUM_ACTORS
        self.a = self.behavior.selectAction(s)

        self.x = xp
        # self.target_para = ta

        return self.a

    # called on the terminal step of the episode
    def end(self, r, step):
        # rho = self.target.ratio(self.behavior, self.s, self.a)

        # there is no next-state on the terminal state, so
        # encode the "next-state" as a zero vector to avoid accidental bootstrapping
        xp = np.zeros_like(self.x)
        # xp = self.representation([2] * self.NUM_ACTORS)
        self.learner.update(self.x, self.s, self.a, r, self.target_para, self.behavior, xp, step)

        self.x = xp
