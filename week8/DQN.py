import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

"""
深度 Q 网络(Deep Q-Network, DQN(是将深度学习引入到强化学习中的一种方法,
旨在解决传统 Q-learning 难以处理高维状态空间的问题。DQN 通过使用深度神经网络来逼近 Q 函数,
从而能够处理具有高维度和连续状态空间的复杂任务。DQN 的核心思想是使用一个神经网络来估计动作价值函数 Q(s, a)。
为了减弱训练样本的相关性,经验回放让每个s-a-r-new_state的四元组存储起来,每次用小批量训练
用两个Q网络来训练,当前Q网络每次训练后更新,目标Q网络每隔一段时间更新一次。减少目标值的波动,提高训练稳定性
当前Q网络用来选择动作,目标Q网络用于计算目标Q值。
"""

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

def train_DQN():
    num_episodes = 500
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 500
    lr = 0.001
    batch_size = 64
    memory_size = 10000
    target_update = 10

    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    memory = deque(maxlen=memory_size)

    def select_action(state, epsilon):
        if random.random() < epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                return policy_net(state).argmax().item()
    
    def optimize_model():
        if len(memory) < batch_size:
            return
        transitions = random.sample(memory, batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        # policy_net(state_batch).shape : (64, 2)
        q_values = policy_net(state_batch).gather(1, action_batch)
        # q_values.shape : (64, 1)
        next_q_values = target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (gamma * next_q_values)
        # print(q_values.shape, expected_q_values.shape)
        loss = nn.functional.mse_loss(q_values, expected_q_values.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    epsilon = epsilon_start
    for episode in range(num_episodes):
        state = env.reset()[0]
        state = torch.tensor([state], dtype=torch.float32).to(device)
        for t in range(1000):
            action = select_action(state, epsilon)
            next_state, reward, done, _, _ = env.step(action)
            reward = torch.tensor([reward], dtype=torch.float32).to(device)
            next_state = torch.tensor([next_state], dtype=torch.float32).to(device)
            memory.append(Transition(state, torch.tensor([[action]], dtype=torch.int64).to(device), reward, next_state))
            state = next_state
            optimize_model()
            if done:
                break
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        epsilon = max(epsilon_end, epsilon * np.exp(-1.0 / epsilon_decay))
    
    print("训练完成!")


device = "cuda" if torch.cuda.is_available() else "cpu"
train_DQN()