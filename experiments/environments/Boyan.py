import numpy as np
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from RlGlue.environment import BaseEnvironment

# Constants
RIGHT = 0
SKIP = 1


class Boyan(BaseEnvironment):
    def __init__(self):
        # self.states = 12; joinAgentStates for compute joint J_mu
        self.NUM_ACTORS = 9
        self.Num_SingleState = 5
        self.lastIndexJointAgentStates = self.Num_SingleState ** self.NUM_ACTORS - 1
        self.state = 0

    def start(self):
        # --- should have a vector joint state, e.g., [0, 1], [1, 2] --- #
        self.state = [0] * self.NUM_ACTORS
        return self.state

    def mapobstoLocalRewards(self):
        # if s == [0, 0]:
        #     return -3
        # if s == [1, 0]:
        #     return -2
        # if s == [2, 0]:
        #     return 0
        # if s == [0, 1]:
        #     return -6
        # if s == [1, 1]:
        #     return -4
        # if s == [2, 1]:
        #     return 0
        # if s == [0, 2]:
        #     return -9
        # if s == [1, 2]:
        #     return -6
        # if s == [2, 2]:
        #     return 0
        # -------- * local rewards assignments * ------- #
        occurences0 = self.state.count(0)
        if occurences0 == self.NUM_ACTORS:
            return np.array([-3] * self.NUM_ACTORS)

        rewards_standard = [-3] * (self.NUM_ACTORS - 2) + [-2] + [0]
        rewards = np.zeros(self.NUM_ACTORS)
        for i in range(self.NUM_ACTORS):
            max_position = max(self.state[:i] + self.state[i + 1:])
            rewards[i] = rewards_standard[i] * (max_position + 1)

        # rewards = np.zeros(self.NUM_ACTORS)
        # occurences1 = self.state.count(1)
        # if occurences1 == self.NUM_ACTORS:
        #     rewards = np.array([-4] * self.NUM_ACTORS)
        # else:
        #     indices = [index for index, element in enumerate(self.state) if element == 1]
        #     rewards[indices] = -6
        return rewards

        # M = self.lastIndexJointAgentStates
        # InterStates = self.helpIndex(2, 3)
        # R = np.zeros(M + 1)
        # # R[364] = -24
        # R[0] = -3 * 2
        # for a in InterStates:
        #     a_ternary = self.ternaryIndex(a)
        #     occurences = a_ternary.count(1)
        #     if occurences == 2:
        #         R[a] = -4 * 2
        #     else:
        #         R[a] = -6 * occurences
        # print('states outside domain!')

        # if s == [[1, 0], [1, 0]]:
        #     return -3
        # if s == [[0.5, 0.5], [1, 0]]:
        #     return -2
        # if s == [[0, 1], [1, 0]]:
        #     return 0
        # if s == [[1, 0], [0.5, 0.5]]:
        #     return -6
        # if s == [[0.5, 0.5], [0.5, 0.5]]:
        #     return -4
        # if s == [[0, 1], [0.5, 0.5]]:
        #     return 0
        # if s == [[1, 0], [0, 1]]:
        #     return -9
        # if s == [[0.5, 0.5], [0, 1]]:
        #     return -6
        # if s == [[0, 1], [0, 1]]:
        #     return 0
        # print('states outside domain!')

    def step(self, a):
        # ------ the reward is -3 for each move before state 12------#
        # reward = -3
        rewards = self.mapobstoLocalRewards()
        # reward2 = self.mapobstoLocalRewards([self.state[1], self.state[0]])
        # reward = reward1 + reward2

        terminal = [False] * self.NUM_ACTORS
        for i in range(self.NUM_ACTORS):
            # --- should be self.state > 10 ? state 11 and state 12 do not have skip action? ---
            # if a[i] == SKIP and self.state[i] > 10:
            if a[i] == SKIP and self.state[i] > self.Num_SingleState - 3:
                print("Double right action is not available in state 10 or state 11... Exiting now.")
                exit()
            # --- here use voting to decide skip or right  ---#
            if a[i] == RIGHT:
                self.state[i] = self.state[i] + 1
            elif a[i] == SKIP:
                self.state[i] = self.state[i] + 2

            # if (self.state[i] == 12):
            #     terminal[i] = True
            #     reward = -2
            if self.state[i] >= self.Num_SingleState - 1:
                self.state[i] = self.Num_SingleState - 1
                terminal[i] = True
                # reward = -2
        term = np.multiply.reduce(terminal)

        # reward1, reward2,
        return rewards, self.state, term

    # ----- find the indices of valid joint states of N agents ------ #
    # def helpIndex(self, N, Num_SingleState):
    #     def computeIndex(N):
    #         if N < 0:
    #             return [0]
    #         collect = []
    #         # --- when N = 2, [1, 1], [1, 2], [2, 1], [2, 2] --- #
    #         for i in range(1, Num_SingleState):
    #             for a in computeIndex(N - 1):
    #                 # ----- the index for state [i, a] ------ #
    #                 collect.append(i * (Num_SingleState ** N) + a)
    #         return collect

    # --- find the feasible actions given current N-ary joint states --- #
    def possible_Actions(self, Current_Point):
        def possible_Actions_help(a, b):
            if a == [] and b == []:
                return [[]]
            collect_Actions = []
            for subaction in possible_Actions_help(a[1:], b[1:]):
                if a[0] >= 0:
                    collect_Actions.append([a[0]] + subaction)

                if b[0] > 0:
                    collect_Actions.append([b[0]] + subaction)
            return collect_Actions

        # -- if some a[i] = 0 after forloop, then computeIndex(newstartpoint) return (None) - #
        # # --- a indicates move to the next or not; b indicates skip or not --- #
        N = self.NUM_ACTORS
        a = [0] * N
        b = [0] * N
        for i in range(N):
            if Current_Point[i] < self.Num_SingleState - 1:
                # --- move one step --- #
                a[i] = 1
            if Current_Point[i] < self.Num_SingleState - 2:
                # --- move two steps --- #
                b[i] = 2
        Possible_Actions = possible_Actions_help(a, b)
        return Possible_Actions

    # ----- find the N-ary indices of valid joint states of N agents ------ #
    def helpIndex(self, N, Num_SingleState):
        collect = []
        possible_Actions = self.possible_Actions
        # def possible_Actions(Current_Point):
        #     def possible_Actions_help(a, b):
        #         if a == [] and b == []:
        #             return [[]]
        #         collect_Actions = []
        #         for subaction in possible_Actions_help(a[1:], b[1:]):
        #             if a[0] >= 0:
        #                 collect_Actions.append([a[0]] + subaction)
        #
        #             if b[0] > 0:
        #                 collect_Actions.append([b[0]] + subaction)
        #         return collect_Actions
        #
        #     # -- if some a[i] = 0 after forloop, then computeIndex(newstartpoint) return (None) - #
        #     # # --- a indicates move to the next or not; b indicates skip or not --- #
        #     a = [0] * N
        #     b = [0] * N
        #     for i in range(N):
        #         if Current_Point[i] < Num_SingleState - 1:
        #             # --- move one step --- #
        #             a[i] = 1
        #         if Current_Point[i] < Num_SingleState - 2:
        #             # --- move two steps --- #
        #             b[i] = 2
        #     Possible_Actions = possible_Actions_help(a, b)
        #     return Possible_Actions

        def computeIndex(Current_Point):
            if list(Current_Point) in collect:
                return

            # ----- if each element is less= than Num_SingleState - 1 ----- #
            if np.multiply.reduce(Current_Point <= Num_SingleState - 1):
                collect.append(list(Current_Point))

                # if Start_point == [Num_SingleState - 1] * N:
                #     return

                Possible_Actions = possible_Actions(Current_Point)
                for action in Possible_Actions:
                    computeIndex(Current_Point + np.array(action))
                    # computeIndex([a[0], a[1]], N)
                    # computeIndex([a[0], b[1]], N)
                    # computeIndex([b[0], a[1]], N)
                    # computeIndex([b[0], b[1]], N)
            return

        # --- for joint states of N agents --- #
        Start_point = np.array([0] * N)
        computeIndex(Start_point)
        return collect

    def ternaryIndex(self, N):
        if -1 < N < 3:
            return [0] * (self.NUM_ACTORS - 1) + [N]
        # Indexternary = np.zeros(6)
        Indexternary = [0] * self.NUM_ACTORS
        i = 0
        while N >= 3:
            Indexternary[(self.NUM_ACTORS - 1) - i] = N % 3
            N = N // 3
            i += 1
        Indexternary[(self.NUM_ACTORS - 1) - i] = N % 3
        return Indexternary

    def naryIndex(self, N, N_SingleStates):
        if -1 < N < N_SingleStates:
            return [0] * (self.NUM_ACTORS - 1) + [N]
        # Indexternary = np.zeros(6)
        Indexternary = [0] * self.NUM_ACTORS
        i = 0
        while N >= N_SingleStates:
            Indexternary[(self.NUM_ACTORS - 1) - i] = N % N_SingleStates
            N = N // N_SingleStates
            i += 1
        Indexternary[(self.NUM_ACTORS - 1) - i] = N % N_SingleStates
        return Indexternary

    def naryToDecimal(self, L, N_SingleStates):
        N = len(L)
        ans = 0
        for i in range(N):
            ans += (N_SingleStates ** i) * L[N - 1 - i]
        return ans

    def getXPRD(self, target_para, rep):
        M = self.lastIndexJointAgentStates
        # --- build the state * feature matrix; X is the state feature matrix (collection of phi(s)) --- #
        # add an extra state at the end to encode the "terminal" state; here 2 is
        # j can be 0, 1, 2
        # i = M // 3
        # j = M % 3
        # Indexternary = self.ternaryIndex(M)
        # X = [
        #     rep.encode([ii, jj]) for ii in range(i + 1)
        #     for jj in range(j + 1)
        # ]
        # X = np.array([
        #     rep.encode(self.ternaryIndex(s)) for s in range(M + 1)
        # ])
        X = np.array([
            rep.encode(self.naryIndex(s, self.Num_SingleState)) for s in range(M + 1)
        ])

        # build a transition dynamics matrix
        # following policy "target"
        # P = np.zeros((M + 1, M + 1))
        # ------ light version of transition matrix ------- #
        P = {}
        InterStates = self.helpIndex(self.NUM_ACTORS, self.Num_SingleState)
        for s in InterStates:
            actions = self.possible_Actions(s)
            N_nonzero = len(actions)
            for a in actions:
                S_IndexDecimal = self.naryToDecimal(s, self.Num_SingleState)
                Sp_IndexDecimal = self.naryToDecimal(np.array(s) + np.array(a), self.Num_SingleState)
                # P[S_IndexDecimal, Sp_IndexDecimal] = 1.0 / N_nonzero
                # ------ light version of transition matrix ------- #
                P[str(S_IndexDecimal) + str(Sp_IndexDecimal)] = 1.0 / N_nonzero
            # P[a, M] = 1

        # P[0, 4] = 0.25
        # P[0, 5] = 0.25
        # P[0, 7] = 0.25
        # P[0, 8] = 0.25
        # P[4, 8] = 1
        # P[5, 8] = 1
        # P[7, 8] = 1
        # P[8, 8] = 1
        # for i in range(11):
        #     P[i, i + 1] = .5
        #     P[i, i + 2] = .5
        #
        # P[10, 11] = 1
        # P[11, 12] = 1

        # build the average reward vector
        # R = np.array([-3] * 10 + [-2, -2, 0])
        # -------- here -8 = -2 + (-6) -------- #
        # --------- expected rewards ---------- #
        # R = [-3 * 2, -100, -100, -100, -4 * 2, -6, -100, -6, 0]
        R = np.zeros(M + 1)
        # R[364] = -24
        # R[0] = -3 * self.NUM_ACTORS
        reward_Transition = [-3] * (self.Num_SingleState - 2) + [-2] + [0]
        for a in InterStates:
            reward_a = 0
            for i in range(len(a)):
                # --- the most advanced state other than the state of agent i --- #
                Index_MostAdvanced = np.max(a[:i] + a[i + 1:])
                # ---  reward for transition from state a[i] --- #
                reward_a += reward_Transition[a[i]] * (Index_MostAdvanced + 1)
            a_IndexDecimal = self.naryToDecimal(a, self.Num_SingleState)
            R[a_IndexDecimal] = reward_a

            # a_ternary = self.ternaryIndex(a)
            # occurences = a_ternary.count(1)
            # if occurences == self.NUM_ACTORS:
            #     R[a] = -4 * self.NUM_ACTORS
            # else:
            #     R[a] = -6 * occurences
        # for i in range(M + 1):

        # D = np.diag(
        #     [0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308, 0.07692308,
        #      0.07692308, 0.07692308, 0.07692308, 0.07692308])
        # ------ here state 1, 2, 3, 6 have probability 0 ------ #
        # D = np.diag([0.2, 0, 0, 0, 0.2, 0.2, 0, 0.2, 0.2])
        # D = np.diag([0.2, 0, 0, 0, 0.2, 0.2, 0, 0.2, 0.2])
        # State_visit = [0]
        # State_visit.extend(InterStates)
        NJointStates = self.lastIndexJointAgentStates + 1
        # D = np.zeros(shape=(NJointStates, NJointStates))
        # --- light version of D --- #
        D = np.zeros(NJointStates)
        # --- set the diagonal elements in uniform distribution --- #
        prob_state = 1.0 / len(InterStates)
        for a in InterStates:
            a_IndexDecimal = self.naryToDecimal(a, self.Num_SingleState)
            # D[a_IndexDecimal, a_IndexDecimal] = prob_state
            # --- light version of D --- #
            D[a_IndexDecimal] = prob_state

        return X, P, R, D


# ------ representation of states  ----------
class BoyanRep:
    def __init__(self):
        self.NUM_ACTORS = 9
        self.Num_SingleState = 5
        # self.map = np.array([
        #     [1,    0,    0,    0   ],
        #     [0.75, 0.25, 0,    0   ],
        #     [0.5,  0.5,  0,    0   ],
        #     [0.25, 0.75, 0,    0   ],
        #     [0,    1,    0,    0   ],
        #     [0,    0.75, 0.25, 0   ],
        #     [0,    0.5,  0.5,  0   ],
        #     [0,    0.25, 0.75, 0   ],
        #     [0,    0,    1,    0   ],
        #     [0,    0,    0.75, 0.25],
        #     [0,    0,    0.5,  0.5 ],
        #     [0,    0,    0.25, 0.75],
        #     [0,    0,    0,    1   ],
        # ])
        # self.phimap = np.array([[[1, 0, 1, 0], [1, 0, 0.5, 0.5], [1, 0, 0, 1]],
        #                      [[0.5, 0.5, 1, 0], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0, 1]],
        #                      [[0, 1, 1, 0], [0, 1, 0.5, 0.5], [0, 1, 0, 1]]
        #                      ])

    def phimap(self, s):
        N = len(s)
        phis = []
        S_interval = self.Num_SingleState - 1
        for i in range(N):
            # for j in range(self.Num_SingleState):
            phis.extend([1 - s[i] * 1 / S_interval, 0 + s[i] * 1 / S_interval])
            # if s[i] == 0:
            #     phis.extend([1, 0])
            # if s[i] == 1:
            #     # phis.extend([0.5, 0.5])
            #     phis.extend([2 / 3, 1 / 3])
            # if s[i] == 2:
            #     phis.extend([1 / 3, 2 / 3])
            # if s[i] == 3:
            #     phis.extend([0, 1])
        return np.array(phis)

    def encode(self, s):
        return self.phimap(s)

    def features(self):
        return 2 * self.NUM_ACTORS

    def endFeature(self):
        return self.phimap([self.Num_SingleState - 1] * self.NUM_ACTORS)
        # return np.array([0, 1, 0, 1])
