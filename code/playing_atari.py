import gym
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn

env = gym.make("BreakoutNoFrameskip-v4")  # 游戏环境
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))  # 设置图片放缩
env = gym.wrappers.GrayScaleObservation(env)  # 设置图片为灰度图
env = gym.wrappers.FrameStack(env, 4)  # 4帧图片堆叠在一起作为一个观测
env = MaxAndSkipEnv(env, skip=4)  # 跳帧，一个动作维持4帧


class DQN(nn.Cell):
    def __init__(self, nb_actions):
        super().__init__()
        self.network = nn.SequentialCell(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4, pad_mode='valid'),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, pad_mode='valid'),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dense(in_channels=2592, out_channels=256),
            nn.ReLU(),
            nn.Dense(in_channels=256, out_channels=nb_actions),
        )

    def construct(self, x):
        return self.network(x / 255.)


class ReplayBuffer():
    def __init__(self, replay_memory_size):
        self.replay_memory_size = replay_memory_size
        self.obs = []
        self.next_obs = []
        self.action = []
        self.reward = []
        self.done = []
        self.pos = 0
        self.full = False

    def add(self, obs, next_obs, action, reward, done):
        if done:
            done = 1
        else:
            done = 0
        if self.full:
            self.obs[self.pos] = obs
            self.next_obs[self.pos] = next_obs
            self.action[self.pos] = [action]
            self.reward[self.pos] = [reward]
            self.done[self.pos] = [done]
        else:
            self.obs.append(obs)
            self.next_obs.append(next_obs)
            self.action.append([action])
            self.reward.append([reward])
            self.done.append([done])
        self.pos += 1
        if self.pos == self.replay_memory_size:
            self.pos = 0
            self.full = True

    def sample(self, sample_num):
        random_index = np.random.choice(min(self.replay_memory_size, len(self.obs)), sample_num, replace=False)
        temp_obs = np.array([self.obs[i] for i in random_index])
        temp_next_obs = np.array([self.next_obs[i] for i in random_index])
        temp_action = np.array([self.action[i] for i in random_index])
        temp_reward = np.array([self.reward[i] for i in random_index])
        temp_done = np.array([self.done[i] for i in random_index])
        return Tensor(temp_obs, ms.float32), Tensor(temp_next_obs, ms.float32), Tensor(temp_action, ms.int32), Tensor(temp_reward, ms.float32), Tensor(temp_done, ms.float32)


q_network = DQN(nb_actions=env.action_space.n)  # 网络实例化
# 尝试加载模型
try:
    param_dict = ms.load_checkpoint("../model/model.ckpt")
    param_not_load, _ = ms.load_param_into_net(q_network, param_dict)
    if len(param_not_load) == 0:
        print('\n', "All parameters are loaded.", end=' ')
    else:
        print('\n', "Some parameters are not loaded:", param_not_load, end=' ')
except:
    print('\n', "No model loaded. Initialize Training.", end=' ')
optimizer = nn.Adam(params=q_network.trainable_params(), learning_rate=1.25e-4)  # 优化器
loss_fn = nn.HuberLoss()  # 损失函数


# 损失值计算函数
def forward_fn(observations, actions, y):
    current_q_value = q_network(observations).gather_elements(dim=1, index=actions).squeeze()  # 把经验对中这个动作对应的q_value给提取出来
    loss = loss_fn(current_q_value, y)
    return loss


# 损失梯度函数
grad_fn = ms.ops.value_and_grad(forward_fn, None, optimizer.parameters)  # 参考:https://www.mindspore.cn/tutorials/zh-CN/r2.1/beginner/autograd.html


# 训练一步的函数
def train_step(observations, actions, y):
    loss, grads = grad_fn(observations, actions, y)
    optimizer(grads)
    return loss


def Deep_Q_Learning(env, replay_memory_size=100_000, nb_epochs=40000_000, update_frequency=4, batch_size=32,
                    discount_factor=0.99, replay_start_size=5000, initial_exploration=1, final_exploration=0.01,
                    exploration_steps=100_000):

    # Initialize replay memory D to capacity N
    # 初始化回放池
    rb = ReplayBuffer(replay_memory_size)
    epoch = np.array(0)  # 轮次
    smoothed_rewards = np.array([0])

    try:
        File = np.load('../model/checkpoint.npz')
        epoch = File['epoch']
        smoothed_rewards = File['smoothed_rewards']
        print("Continue training !")
    except:
        print("No checkpoint. Initialize training.")

    rewards = []
    q_network.set_train()
    progress_bar = tqdm(total=nb_epochs-epoch)  # tqdm: 用于显示进度条

    while epoch <= nb_epochs:

        dead = False
        total_rewards = 0

        # Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)、
        obs = env.reset()

        for _ in range(random.randint(1, 30)):  # Noop and fire to reset environment: 等待并且开火用于重置环境
            obs, _, _, info = env.step(1)

        while not dead:
            current_life = info['lives']  # 剩余生命值

            epsilon = max((final_exploration - initial_exploration) / exploration_steps * epoch + initial_exploration,
                          final_exploration)
            if random.random() < epsilon:  # With probability ε select a random action a
                action = np.array(env.action_space.sample())
            else:  # Otherwise select a = max_a Q∗(φ(st), a; θ)
                temp_input = Tensor(obs, ms.float32).unsqueeze(0)
                q_values = q_network(temp_input)
                action = q_values.argmax(axis=1).item().asnumpy()
                # action = np.array(env.action_space.sample())

            # Execute action a in emulator and observe reward rt and image xt+1
            next_obs, reward, dead, info = env.step(action)

            done = True if (info['lives'] < current_life) else False

            # Set st+1 = st, at, xt+1 and preprocess φt+1 = φ(st+1)
            real_next_obs = next_obs.copy()

            total_rewards += reward
            reward = np.sign(reward)  # Reward clipping

            # Store transition (φt, at, rt, φt+1) in D
            rb.add(obs, real_next_obs, action, reward, done)

            obs = next_obs

            if len(rb.obs) > replay_start_size and epoch % update_frequency == 0:  # 训练
                data_obs, data_next_obs, data_action, data_reward, data_done = rb.sample(batch_size)
                # if data_reward.flatten().sum() != 0:
                #     print("reward!=0, reward=", data_reward.flatten().sum())

                # 这一部分不用求梯度，所以写在forward_fn和train_step函数之外
                max_q_value = q_network(data_next_obs).max(1)
                y = data_reward.flatten() + discount_factor * max_q_value * (1 - data_done.flatten())

                loss = train_step(data_obs, data_action, y)
                # print(loss)

            epoch += 1

            if (epoch % 1_000 == 0) and epoch > 0 and len(rewards) != 0:
                smoothed_rewards = np.append(smoothed_rewards, np.mean(rewards))
                rewards = []
                plt.plot(smoothed_rewards)
                plt.title("Average Reward on Breakout")
                plt.xlabel("Training Epochs")
                plt.ylabel("Average Reward per Episode")
                plt.savefig('../Imgs/average_reward_on_breakout.png')
                plt.close()
            if (epoch % 10000 == 0) and epoch > 0:
                np.savez('../model/checkpoint.npz', smoothed_rewards=smoothed_rewards, epoch=epoch)
                ms.save_checkpoint(q_network, "../model/model.ckpt")

            progress_bar.update(1)
        rewards.append(total_rewards)

Deep_Q_Learning(env)
env.close()
