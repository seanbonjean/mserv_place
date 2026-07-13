"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, available_actions=None):
        self.check_state_exist(observation)
        if available_actions is None:
            available_actions = list(self.actions)
        else:
            available_actions = list(available_actions)

        if not available_actions:
            raise ValueError("No available action can be selected.")
        if any(action not in self.actions for action in available_actions):
            raise ValueError("Available actions contain unknown action ids.")

        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, available_actions]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(available_actions)
        return action

    def learn(self, s, a, r, s_, next_available_actions=None):
        self.check_state_exist(s)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # next state is not terminal
            self.check_state_exist(s_)
            if next_available_actions is None:
                next_available_actions = list(self.actions)
            else:
                next_available_actions = list(next_available_actions)
            if not next_available_actions:
                raise ValueError(
                    "Non-terminal state must have at least one available action."
                )
            if any(
                action not in self.actions
                for action in next_available_actions
            ):
                raise ValueError(
                    "Next available actions contain unknown action ids."
                )
            q_target = r + self.gamma * self.q_table.loc[
                s_, next_available_actions
            ].max()
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table._append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
