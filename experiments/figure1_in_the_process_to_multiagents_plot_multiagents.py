import numpy as np

# source found at: https://github.com/andnp/RlGlue
from RlGlue.rl_glue import RlGlue
from utils.Collector import Collector
from utils.policies import actionArrayToPolicy, matrixToPolicy
from utils.rl_glue import RlGlueCompatWrapper
from utils.errors import buildRMSPBE, buildJ_mu

from environments.Boyan import Boyan, BoyanRep
from environments.Baird import Baird, BairdRep

from agents.MGOPAC import MGOPAC
from agents.EMOPAC import EMOPAC

import matplotlib.pyplot as plt
from utils.plotting import plot, plot1

# ---------------------------------
# Set up parameters for experiment
# ---------------------------------

RUNS = 5
EPISODES = 30000
marker_step = 2000
LEARNERS = [MGOPAC, EMOPAC]
# LEARNERS = [MGOPAC]
# LEARNERS = [EMOPAC]
# ---------- single-agent state size --------- #
state_size = 5
# --------- single-agent action size --------- #
action_size = 2
NUM_ACTORS = 9
THETA_DIM = 18
# np.random.normal(size=(state_size, action_size), dtype=np.float64), dtype=np.float64)
# ------ target policy parameters q_sa ------#
np.random.seed(2025)
# ------- select q_sa from Gaussian distribution; 12 is the dimension of theta -------- #
target_para_random = - np.random.normal(loc=0.0, scale=1.0, size=(state_size, action_size, THETA_DIM))

STEPSIZE_MGOPAC = 1.2e-1
STEPSIZE_EMOPAC = 0.8e-1

PROBLEMS = [
    # Boyan's chain
    {
        'env': Boyan,
        # --- representations ---
        'representation': BoyanRep,
        # go LEFT 40% of the time; target policy ???; actually the tabular policy
        # 'target': matrixToPolicy([[.5, .5]] * 10 + [[1., 0.]] * 2),
        'target_para': target_para_random,
        # take each action equally; behavior policy
        # 'behavior': matrixToPolicy([[.5, .5]] * 10 + [[1., 0.]] * 2),
        # ----- for each agent in multiagent settings; [1, 0] means move one step w.p. 1
        # [0, 1] means skip w.p. 1 ------ #
        # ------ the first argument is for single agent -------- #
        'behavior': matrixToPolicy([[0.5, 0.5]] * (state_size - 2) + [[1, 0]] * 2, NUM_ACTORS),
        'gamma': 1.0,
        # number of total steps; 5000
        'steps': 2000,
        # hardcode stepsizes found from parameter study
        'stepsizes': {
            'TD': 0.0625,
            # 'MGOPAC': 0.0625,
            # 'MGOPAC': 0.0625,
            'EMOPAC': STEPSIZE_EMOPAC,
            'MGOPAC': STEPSIZE_MGOPAC,
            'TDC': 0.5,
        }
    },
    # # Baird's Counter-example domain
    # {
    #     'env': Baird,
    #     'representation': BairdRep,
    #     # go LEFT 40% of the time; actually at each state uses the same principle
    #     # here we can define a policy controlled by theta
    #     'target': actionArrayToPolicy([0., 1.]),
    #     # take each action equally
    #     'behavior': actionArrayToPolicy([6/7, 1/7]),
    #     'starting_condition': np.array([1, 1, 1, 1, 1, 1, 1, 10]),
    #     'gamma': 0.99,
    #     'steps': 20000,
    #     # hardcode stepsizes found from parameter study
    #     'stepsizes': {
    #         'TD': 0.00390625,
    #         'MGOPAC': 0.015625,
    #         'TDC': 0.00390625,
    #     }
    # },
]

COLORS = {
    'TDC': 'pink',
    'MGOPAC': 'red',
    'EMOPAC': 'green',
}

# -----------------------------------
# Collect the data for the experiment
# -----------------------------------

# a convenience object to store data collected during runs
collector = Collector()
# lmdas = [0, 0.2, 0.5]
lmdas = [0, 0.2, 0.5]


fig = plt.figure(1)
ax = fig.gca()

Colors = {'MGOPAC': ['red', 'magenta', 'green', 'blue', 'orange', 'grey', 'purple', 'pink', 'brown', 'cyan',
                     'bisque', 'olive', 'lightcoral'],
          'EMOPAC': ['cyan', 'blue', 'orange', 'brown', 'pink', 'bisque', 'lightcoral', 'olive']}  # cyan is obvious
markers = ['o', 'v', '^', '<', '>', '*', '1', '2', '3', '4', '5', '6', '8']
markers_on = [i for i in range(1, EPISODES, marker_step)]
Marker_idx = 0
# Colors_mag = {'MGOPAC': ['green', 'blue', 'red', 'orange', 'grey', 'purple'],
#               'EMOPAC': ['pink', 'brown', 'cyan', 'bisque', 'olive', 'lightcoral']}

for Learner in LEARNERS:
    Color_Index = 0
    for lmda in lmdas:
        for run in range(RUNS):
            for problem in PROBLEMS:
                # for reproducibility, set the random seed for each run
                # also reset the seed for each learner, so we guarantee each sees the same data
                np.random.seed(run)

                # build a new instance of the environment each time
                # just to be sure we don't bleed one learner into the next; choose an environment
                Env = problem['env']
                env = Env()

                # target = problem['target']
                target_para = problem['target_para']
                behavior = problem['behavior']

                Rep = problem['representation']
                rep = Rep()
                # --------- learners mean algorithms --------- #
                print(run, Env.__name__, Rep.__name__, Learner.__name__)

                # build the X, P, R, and D matrices for computing RMSPBE
                # X, P, R, D = env.getXPRD(target_para, rep)
                # ------------------ RMSPBE -----------------#
                # RMSPBE = buildRMSPBE(X, P, R, D, problem['gamma'])
                # ------------ average return ----------- #
                J_MU = buildJ_mu(1, rep.endFeature())

                # build a new instance of the learning algorithm; learner with two arguments features and params
                learner = Learner(rep.features(), {
                    'gamma': problem['gamma'],
                    'alpha': problem['stepsizes'][Learner.__name__],
                    # 'alpha_theta': 0.0625,
                    'alpha_theta': problem['stepsizes'][Learner.__name__],
                    'beta': 1,
                }, lmda)

                # #--- build an "agent" which selects actions according to the behavior policy
                # and tries to estimate according to the target policy; agent includes the learner ---#
                # ------------- target is used to compute the importance ratio -------------#
                # ---------- NUM_ACTORS; each actor has an identical behavior policy ----------- #
                # agent = [RlGlueCompatWrapper(learner, behavior, target_para, rep.encode) for _ in range(NUM_ACTORS)]
                agent = RlGlueCompatWrapper(learner, behavior, target_para, rep.encode, NUM_ACTORS)

                # for Baird's counter-example, set the initial value function manually
                if problem.get('starting_condition') is not None:
                    learner.w = problem['starting_condition'].copy()

                # build the experiment runner
                # ---------- ties together the agent and environment ---------- #
                # and allows executing the agent-environment interface from Sutton-Barto
                glue = RlGlue(agent, env)

                # ---------------------------#
                #    run on control tasks
                # ---------------------------#
                glue.start()
                episode = 0
                timestep = 0
                for episode in range(EPISODES):
                    timestep += 1
                    glue.num_steps = 0
                    glue.total_reward = 0
                    glue.runEpisode(timestep, max_steps=1000)
                    # --- compute off policy expected value function j_mu ---
                    w = learner.getWeights()
                    # ----- record the j_mu for each agent !!!----- #
                    # j_mu = []
                    # for wi in w:
                    #     j_mu.append(J_MU(wi))
                    # -------- record the mean j_mu!! ------- #
                    w1 = np.mean(w, 0)
                    j_mu = J_MU(w1)
                    # # print('step:', step, 'rmspbe:', rmspbe, '\n')

                    print(Learner.__name__, run, episode, glue.num_steps)
                    # -------- here collect total rewards -------- #
                    # collector.collect(Learner.__name__, glue.total_reward)
                    collector.collect(Learner.__name__, j_mu)

                # collector.reset()

                # tell the data collector we're done collecting data for this env/learner/rep combination
                collector.reset()
        # draw the results in the plots
        name = Learner.__name__
        data = collector.getStats(name)
        plot(ax, data, label=name + ' $\lambda=$' + str(lmda), color=Colors[name][Color_Index],
             marker=markers[Marker_idx], markers_on=markers_on)
        # for i in range(min(NUM_ACTORS, 4)):
        #     # --- select agents separated --- #
        #     # plot(ax, [data[0][:, 3 * i], data[1][:, 3 * i], data[2]], label=name, color=Colors[i])
        #     # + ' $\lambda=$' + str(lmda)
        #     plot1(ax, [data[0][:, i], data[1][:, i], data[2]], label=name + str(2 * i + 1),
        #           color=Colors_mag[name][i])
        Color_Index += 1
        Marker_idx += 1

plt.xlabel("Episodes", fontsize=15)
plt.ylabel("Averaged Return", fontsize=15)
plt.grid()
# ax.set_xlim()
# ax.set_ylim([-40, 0])
# plt.axis([x, x1, y, y1])
# ax2.legend(loc='lower center')
ax.legend(loc='best', fontsize=15)
fig.savefig('mean.pdf')
plt.show()

# # --------------------------------
# # Plotting for actor critic
# # --------------------------------
# import matplotlib.pyplot as plt
# from utils.plotting import plot
#
# ax = plt.gca()
#
# # ----- plot for mean w ------ #
# for Learner in LEARNERS:
#     name = Learner.__name__
#     data = collector.getStats(name)
#     plot(ax, data, label=name, color=COLORS[name])
#     # plot(ax, data, label='MGOPAC', color=COLORS[name])
# plt.xlabel("Episodes")
# plt.ylabel("Averaged Return")
# plt.legend()
# plt.show()

# # ----- plot for all agents ----- #
# for Learner in LEARNERS:
#     name = Learner.__name__
#     # --- data[0] is of shape (NUM_of_episodes, NUM_ACTORS) --- #
#     # --- data: [(NUM_of_episodes, NUM_ACTORS), (NUM_of_episodes, NUM_ACTORS), 3]
#     data = collector.getStats(name)
#     # ------ plot for mean j_mu ------ #
#     Colors = {'MGOPAC': ['green', 'blue', 'red', 'orange', 'grey', 'purple'],
#               'EMOPAC': ['pink', 'brown', 'cyan', 'bisque', 'olive', 'lightcoral']}
#     for i in range(min(NUM_ACTORS, 4)):
#         # --- select agents separated --- #
#         # plot(ax, [data[0][:, 3 * i], data[1][:, 3 * i], data[2]], label=name, color=Colors[i])
#         plot(ax, [data[0][:, i], data[1][:, i], data[2]], label=name + str(2 * i + 1), color=Colors[name][i])
# plt.xlabel("Episodes")
# plt.ylabel("Averaged Return")
# ax.set_xlim()
# ax.set_ylim([-40, 0])
# plt.axis([x, x1, y, y1])
# plt.legend()
# # plt.savefig('general_cases_' + str(NUM_ACTORS) + 'actors_' + str(state_size) + 'states.pdf')
# plt.show()

