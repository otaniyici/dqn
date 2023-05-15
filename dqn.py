# Code for https://www.youtube.com/watch?v=NP8pXZdU-5U

from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random
import matplotlib.pyplot as plt

# Discount for compute temporal difference
GAMMA = 0.99
# How many transitions to sample from replay buffer
BATCH_SIZE = 32
# Max number of transitions to be stored
BUFFER_SIZE = 50_000
# How many transition before start computing gradients and training
MIN_REPLAY_SIZE = 1000
# Epsilon configurations
## Initial value
EPSILON_START = 1.0
## Final value
EPSILON_END = 0.02
## How many steps to go from EPSILON_START to EPSILON_END
EPSILON_DECAY = 100_000
# Number of steps to update the target network
TARGET_UPDATE_FREQ = 1000

class Network(nn.Module):
    def __init__(self, env):
        super().__init__()

        # if observation_space.shape = [8,1], the number of in_features is 8x1
        in_features = int(np.prod(env.observation_space.shape))

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, env.action_space.n)  # action_space.n is the number of possible actions
        )

    def forward(self, x):
        return self.net(x)

    # Will compute next action
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).cuda()
        q_values = self(obs_t.unsqueeze(0)).cuda()

        # Get next action
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action


class DQN:
    def __init__(self, double):
        self.step_arr = []
        self.reward_arr = []
        self.double = double

    def train(self):
        env = gym.make('CartPole-v1', render_mode="rgb_array")
        # env = gym.make('LunarLander-v2')

        replay_buffer = deque(maxlen=BUFFER_SIZE)
        reward_buffer = deque([0.0], maxlen=100)
        epsilon_buffer = deque([1.0], maxlen=100)

        episode_reward = 0.0

        online_network = Network(env).cuda()
        target_network = Network(env).cuda()

        # Make target network to be initialised the same as online
        target_network.load_state_dict(online_network.state_dict())

        optimizer = torch.optim.Adam(online_network.parameters(), lr=5e-4)

        # Initialise replay buffer
        obs, _ = env.reset()
        for _ in range(MIN_REPLAY_SIZE):
            action = env.action_space.sample()
            new_obs, reward, terminated, truncated, _ = env.step(action)
            transition = (obs, action, reward, terminated or truncated, new_obs)
            replay_buffer.append(transition)
            obs = new_obs

            if terminated or truncated:
                env.reset()

        # Restart the environment
        env.reset()
        # Training loop
        for step in range(150000):
            epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = online_network.act(obs)

            new_obs, reward, terminated, truncated, _ = env.step(action)
            transition = (obs, action, reward, terminated or truncated, new_obs)
            replay_buffer.append(transition)
            obs = new_obs

            episode_reward += reward

            if terminated or truncated:
                env.reset()
                reward_buffer.append(episode_reward)
                epsilon_buffer.append(epsilon)
                episode_reward = 0.0

            # Start gradient step
            transitions = random.sample(replay_buffer, BATCH_SIZE)

            observations = np.asarray([t[0] for t in transitions], dtype=np.float32)
            actions = np.asarray([t[1] for t in transitions], dtype=np.int64)
            rewards = np.asarray([t[2] for t in transitions], dtype=np.float32)
            dones = np.asarray([t[3] for t in transitions], dtype=np.float32)
            new_observations = np.asarray([t[4] for t in transitions], dtype=np.float32)

            observations_t = torch.as_tensor(observations, dtype=torch.float32).cuda()
            actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).cuda()
            rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).cuda()
            dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).cuda()
            new_observations_t = torch.as_tensor(new_observations, dtype=torch.float32).cuda()

            # Compute targets
            with torch.no_grad():
                if self.double:
                    targets_online_q_values = online_network(new_observations_t)
                    targets_online_q_values_indices = targets_online_q_values.argmax(dim=1, keepdim=True)

                    target_q_values = target_network(new_observations_t).cuda()
                    online_selected_target_q_values = torch.gather(input=target_q_values, dim=1,
                                                                   index=targets_online_q_values_indices)

                    targets = rewards_t + GAMMA * (1 - dones_t) * online_selected_target_q_values
                else:
                    target_q_values = target_network(new_observations_t).cuda()
                    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0].cuda()

                    targets = rewards_t + GAMMA * (1 - dones_t) * max_target_q_values

            # Compute loss
            q_values = online_network(observations_t)
            action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)
            loss = nn.functional.smooth_l1_loss(action_q_values, targets)

            # Gradient Descent
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network
            if step % TARGET_UPDATE_FREQ == 0:
                target_network.load_state_dict(online_network.state_dict())

            if step % 1000 == 0:
                print('')
                print(f'{step=}')
                avg_score = np.mean(reward_buffer)
                print(f'avg reward={avg_score}')
                epsilon = np.mean(epsilon_buffer)
                print(f'epsilon={epsilon}')
                self.step_arr.append(step)
                self.reward_arr.append(avg_score)


dqn = DQN(False)
ddqn = DQN(True)
dqn.train()
ddqn.train()


plt.plot(dqn.step_arr, dqn.reward_arr, 'r--', ddqn.step_arr, ddqn.reward_arr, 'b--')
plt.ylabel("reward")
plt.xlabel("episode")
plt.legend(["dqn", "ddqn"])
plt.show()
