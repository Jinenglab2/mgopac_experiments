from copy import deepcopy as dcp
import numpy as np


class RlGlue:
    def __init__(self, agent, env):
        self.environment = env
        self.agent = agent
        # self.NUM_ACTORS = self.agent.NUM_ACTORS
        # self.step = step

        self.last_action = None
        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def start(self):
        # ----- initialize/reset the step process; s is of dimenion NUM_ACTORS ----- #
        s = self.environment.start()
        # ----- the same as s here (obs does NOT change automatically with s) ----- #
        # obs = self.observationChannel(s)
        obs = dcp(s)
        self.last_action = self.agent.start(obs)

        return (obs, self.last_action)

    def step(self, timestep):
        # ----- term = True means each agent attains termination state ----- #
        (rewards, s, term) = self.environment.step(self.last_action)
        # obs = self.observationChannel(s)
        obs = dcp(s)
        # reward = reward1 + reward2
        reward_sum = np.sum(rewards)

        self.total_reward += reward_sum

        if term:
            self.num_episodes += 1
            self.agent.end(rewards, timestep % 40)
            roat = (rewards, obs, None, term)
        else:
            self.num_steps += 1
            # --- such that timestep is not too small large --- #
            # timestep % 1000; timestep
            self.last_action = self.agent.step(rewards, obs, timestep % 200)
            roat = (rewards, obs, self.last_action, term)

        self.recordTrajectory(roat[1], roat[2], roat[0], roat[3])
        return roat

    def runEpisode(self, timestep=1, max_steps=0):
        is_terminal = False
        # timestep = 0
        self.start()

        while (not is_terminal) and ((max_steps == 0) or (self.num_steps < max_steps)):
            # -------- here the step means the environment and agent steps -------#
            rl_step_result = self.step(timestep)
            is_terminal = rl_step_result[3]
            # timestep += 1

        return is_terminal

    def observationChannel(self, s):
        return s

    def recordTrajectory(self, s, a, r, t):
        pass
