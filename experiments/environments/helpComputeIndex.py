import numpy as np


def helpIndex(N):
    # collect = []
    def computeIndex(N):
        if N < 0:
            return [0]
        collect = []
        for i in range(1, 3):
            for a in computeIndex(N - 1):
                collect.append(i * (3 ** N) + a)
        return collect

    return computeIndex(N)


def ternaryIndex(N):
    if -1 < N < 3:
        return [0] * 9 + [N]
    # Indexternary = np.zeros(6)
    Indexternary = [0] * 10
    i = 0
    while N >= 3:
        Indexternary[9 - i] = N % 3
        N = N // 3
        i += 1
    Indexternary[9 - i] = N % 3
    return Indexternary


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


# def possible_Actions(a, b):
#     if a == [] and b == []:
#         return [[]]
#     collect_Actions = []
#     for subaction in possible_Actions(a[1:], b[1:]):
#         if a[0] >= 0:
#             collect_Actions.append([a[0]] + subaction)
#
#         if b[0] > 0:
#             collect_Actions.append([b[0]] + subaction)
#     return collect_Actions

# def helpIndex(N, Num_SingleState):
#     collect = []
#
#     def possible_Actions(a, b):
#         if a == [] and b == []:
#             return [[]]
#         collect_Actions = []
#         for subaction in possible_Actions(a[1:], b[1:]):
#             if a[0] >= 0:
#                 collect_Actions.append([a[0]] + subaction)
#
#             if b[0] > 0:
#                 collect_Actions.append([b[0]] + subaction)
#         return collect_Actions
#
#     def computeIndex(Current_Point):
#         if list(Current_Point) in collect:
#             return
#
#         # ----- if each element is less= than Num_SingleState - 1 ----- #
#         if np.multiply.reduce(Current_Point <= Num_SingleState - 1):
#             collect.append(list(Current_Point))
#
#             # if Start_point == [Num_SingleState - 1] * N:
#             #     return
#
#             # -- if some a[i] = 0 after forloop, then computeIndex(newstartpoint) return (None) - #
#             # # --- a indicates move to the next or not; b indicates skip or not --- #
#             a = [0] * N
#             b = [0] * N
#             for i in range(N):
#                 if Current_Point[i] < Num_SingleState - 1:
#                     # --- move one step --- #
#                     a[i] = 1
#                 if Current_Point[i] < Num_SingleState - 2:
#                     # --- move two steps --- #
#                     b[i] = 2
#             Possible_Actions = possible_Actions(a, b)
#             for action in Possible_Actions:
#                 computeIndex(Current_Point + np.array(action))
#                 # computeIndex([a[0], a[1]], N)
#                 # computeIndex([a[0], b[1]], N)
#                 # computeIndex([b[0], a[1]], N)
#                 # computeIndex([b[0], b[1]], N)
#         return
#
#     # --- for joint states of N agents --- #
#     Start_point = np.array([0, 0, 0])
#     computeIndex(Start_point)
#     return collect

def helpIndex(N, Num_SingleState):
    collect = []

    def possible_Actions(Current_Point):
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
        a = [0] * N
        b = [0] * N
        for i in range(N):
            if Current_Point[i] < Num_SingleState - 1:
                # --- move one step --- #
                a[i] = 1
            if Current_Point[i] < Num_SingleState - 2:
                # --- move two steps --- #
                b[i] = 2
        Possible_Actions = possible_Actions_help(a, b)
        return Possible_Actions

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

def naryIndex(N, N_SingleStates):
    NUM_ACTORS = 3
    if -1 < N < N_SingleStates:
        return [0] * (NUM_ACTORS - 1) + [N]
    # Indexternary = np.zeros(6)
    Indexternary = [0] * NUM_ACTORS
    i = 0
    while N >= N_SingleStates:
        Indexternary[(NUM_ACTORS - 1) - i] = N % N_SingleStates
        N = N // N_SingleStates
        i += 1
    Indexternary[(NUM_ACTORS - 1) - i] = N % N_SingleStates
    return Indexternary

def possible_Actions(Current_Point):
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
    N = len(Current_Point)
    Num_SingleState = 4
    a = [0] * N
    b = [0] * N
    for i in range(N):
        if Current_Point[i] < Num_SingleState - 1:
            # --- move one step --- #
            a[i] = 1
        if Current_Point[i] < Num_SingleState - 2:
            # --- move two steps --- #
            b[i] = 2
    Possible_Actions = possible_Actions_help(a, b)
    return Possible_Actions

def naryToDecimal(L, N_SingleStates):
    N = len(L)
    ans = 0
    for i in range(N):
        ans += (N_SingleStates ** i) * L[N - 1 - i]
    return ans

def averageRewards(Num_SingleState, InterStates):
    M = Num_SingleState ** 2 - 1
    R = np.zeros(M + 1)
    # R[364] = -24
    # R[0] = -3 * self.NUM_ACTORS
    reward_Transition = [-3] * (Num_SingleState - 2) + [-2] + [0]
    for a in InterStates:
        reward_a = 0
        for i in range(len(a)):
            # --- the most advanced state other than the state of agent i --- #
            Index_MostAdvanced = np.max(a[:i] + a[i+1:])
            # ---  reward for transition from state a[i] --- #
            reward_a += reward_Transition[a[i]] * (Index_MostAdvanced + 1)
        a_IndexDecimal = naryToDecimal(a, Num_SingleState)
        R[a_IndexDecimal] = reward_a
    return R

if __name__ == "__main__":
    # a = helpIndex(5)
    # b = ternaryIndex(1002)
    # adjointMat = np.array([[1.0, 1.0, 0, 0, 1.0, 0, 0, 0],
    #                                [1.0, 1.0, 1.0, 0, 0, 1.0, 0, 0],
    #                                [0, 1.0, 1.0, 1.0, 0, 0, 1.0, 0],
    #                                [0, 0, 1.0, 1.0, 0, 0, 0, 1.0],
    #                                [1.0, 0, 0, 0, 1.0, 1.0, 0, 0],
    #                                [0, 1.0, 0, 0, 1.0, 1.0, 1.0, 0],
    #                                [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0],
    #                                [0, 0, 0, 1.0, 0, 0, 1.0, 1.0]])
    # a = connectionMatrix(adjointMat)
    # a = [0, 0, 5]
    # b = [1, 1, 1]
    # a = [0, 0, 5]
    # b = [1, 1, 5]
    # a = [1, 1, 0]
    # b = [2, 0, 0]
    # a = [5, 5, 5]
    # b = [5, 5, 5]
    # c = possible_Actions(a, b)
    # c = helpIndex(2, 4)
    # c = helpIndex(3, 4)
    # c = possible_Actions([1, 2])
    # c = possible_Actions([1, 2, 2])
    # c = possible_Actions([2, 2, 3])
    # c = possible_Actions([3, 3, 3])
    # c = naryIndex(12, 4)
    # c = naryToDecimal([1, 2, 3, 3, 3], 4)
    # c = naryToDecimal([0, 0, 1, 0, 2], 4)
    c = helpIndex(2, 5)
    a = averageRewards(5, c)
    # a = [-3 * 2, -100, -100, -100, -4 * 2, -6, -100, -6, 0]
    print(a)
