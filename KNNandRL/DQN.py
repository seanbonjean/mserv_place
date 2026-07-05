"""
Reusable DQN agent.

The original file was the Morvan Zhou CartPole demo.  This version keeps the
same small-network DQN style, but makes the state/action dimensions
configurable so it can be imported by graph-partitioning experiments.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
HIDDEN_DIM = 50


class Net(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=HIDDEN_DIM):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_dim, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.out(x)


class DQN(object):
    def __init__(
        self,
        n_states,
        n_actions,
        learning_rate=LR,
        reward_decay=GAMMA,
        e_greedy=EPSILON,
        target_replace_iter=TARGET_REPLACE_ITER,
        memory_capacity=MEMORY_CAPACITY,
        batch_size=BATCH_SIZE,
        hidden_dim=HIDDEN_DIM,
        mask_invalid_actions=False,
        device=None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.mask_invalid_actions = mask_invalid_actions
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.eval_net = Net(n_states, n_actions, hidden_dim).to(self.device)
        self.target_net = Net(n_states, n_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2), dtype=np.float32)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=learning_rate)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x, available_actions=None):
        state = np.asarray(x, dtype=np.float32)
        if state.shape[0] != self.n_states:
            raise ValueError(
                f"Expected state length {self.n_states}, got {state.shape[0]}"
            )

        available_actions = self._normalize_available_actions(
            available_actions, state
        )

        if np.random.uniform() < self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                actions_value = self.eval_net(state_tensor).cpu().numpy().ravel()
            if available_actions is not None:
                masked_value = np.full(self.n_actions, -np.inf, dtype=np.float32)
                masked_value[available_actions] = actions_value[available_actions]
                action = int(np.argmax(masked_value))
            else:
                action = int(np.argmax(actions_value))
        else:
            if available_actions is None:
                action = int(np.random.randint(0, self.n_actions))
            else:
                action = int(np.random.choice(available_actions))
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_)).astype(np.float32)
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.memory_counter < self.batch_size:
            return None

        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        memory_size = min(self.memory_counter, self.memory_capacity)
        sample_index = np.random.choice(memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(self.device)
        b_a = torch.LongTensor(
            b_memory[:, self.n_states:self.n_states + 1].astype(int)
        ).to(self.device)
        b_r = torch.FloatTensor(
            b_memory[:, self.n_states + 1:self.n_states + 2]
        ).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(self.device)

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        if self.mask_invalid_actions and self.n_states == self.n_actions:
            next_action_mask = b_s_ > 0
            q_next = q_next.masked_fill(~next_action_mask, -float("inf"))
            q_next_max = q_next.max(1)[0]
            q_next_max = torch.where(
                torch.isfinite(q_next_max),
                q_next_max,
                torch.zeros_like(q_next_max),
            )
        else:
            q_next_max = q_next.max(1)[0]
        q_target = b_r + self.gamma * q_next_max.view(self.batch_size, 1)

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _normalize_available_actions(self, available_actions, state):
        if available_actions is None and self.mask_invalid_actions:
            if self.n_states != self.n_actions:
                return None
            available_actions = np.where(state > 0)[0]
        elif available_actions is not None:
            available_actions = np.asarray(list(available_actions), dtype=np.int64)

        if available_actions is None:
            return None
        if len(available_actions) == 0:
            raise ValueError("No available action can be selected.")
        if np.any(available_actions < 0) or np.any(available_actions >= self.n_actions):
            raise ValueError("Available actions contain out-of-range action ids.")
        return available_actions


def _run_cartpole_demo():
    import gym

    env = gym.make("CartPole-v0")
    env = env.unwrapped
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]
    dqn = DQN(n_states=n_states, n_actions=n_actions)

    print("\nCollecting experience...")
    for i_episode in range(400):
        reset_result = env.reset()
        s = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        ep_r = 0
        while True:
            env.render()
            a = dqn.choose_action(s)
            step_result = env.step(a)
            if len(step_result) == 5:
                s_, r, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                s_, r, done, _ = step_result

            x, _, theta, _ = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (
                (env.theta_threshold_radians - abs(theta))
                / env.theta_threshold_radians
                - 0.5
            )
            r = r1 + r2
            dqn.store_transition(s, a, r, s_)
            ep_r += r
            loss = dqn.learn()
            if done:
                if loss is not None:
                    print("Ep: ", i_episode, "| Ep_r: ", round(ep_r, 2))
                break
            s = s_


if __name__ == "__main__":
    _run_cartpole_demo()
