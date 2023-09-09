import gym
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
from matplotlib import animation

env = gym.make("BreakoutNoFrameskip-v4")  # 游戏环境
env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.ResizeObservation(env, (84, 84))  # 设置图片放缩
env = gym.wrappers.GrayScaleObservation(env)  # 设置图片为灰度图
env = gym.wrappers.FrameStack(env, 4)  # 4帧图片堆叠在一起作为一个观测
env = MaxAndSkipEnv(env, skip=4)  # 跳帧，一个动作维持4帧

gif_env = gym.make("BreakoutNoFrameskip-v4", rendor_mode='human')
gif_env = gym.wrappers.RecordEpisodeStatistics(gif_env)
gif_env = gym.wrappers.ResizeObservation(gif_env, (84, 84))  # 设置图片放缩
gif_env = gym.wrappers.FrameStack(gif_env, 4)  # 4帧图片堆叠在一起作为一个观测
gif_env = MaxAndSkipEnv(gif_env, skip=4)  # 跳帧，一个动作维持4帧


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


q_network = DQN(nb_actions=env.action_space.n)  # 网络实例化
# 尝试加载模型
try:
    param_dict = ms.load_checkpoint("../model/model.ckpt")
    param_not_load, _ = ms.load_param_into_net(q_network, param_dict)
    if len(param_not_load) == 0:
        print("All parameters are loaded.")
    else:
        print("Some parameters are not loaded:", param_not_load)
except:
    print("No model loaded. Initialize Training.")

q_network.set_train(False)


def Playing_Atari():
    dead = False
    total_rewards = 0
    frames = []
    # Initialise sequence s1 = {x1} and preprocessed sequenced φ1 = φ(s1)、
    obs = env.reset()
    gif_obs = gif_env.reset()
    # for frame in gif_obs:
    #     frames.append(frame)
    frames.append(gif_obs[0])

    for _ in range(random.randint(1, 40)):  # Noop and fire to reset environment: 等待并且开火用于重置环境
        obs, _, _, info = env.step(1)
        gif_obs, _, _, _ = gif_env.step(1)
        frames.append(gif_obs[0])

    step = 0
    while not dead:

        temp_input = Tensor(obs, ms.float32).unsqueeze(0)
        q_values = q_network(temp_input)
        action = q_values.argmax(axis=1).item().asnumpy()
        # Execute action a in emulator and observe reward rt and image xt+1
        next_obs, reward, dead, info = env.step(action)
        gif_obs, _, _, _ = gif_env.step(action)
        frames.append(gif_obs[0])

        total_rewards += reward

        obs = next_obs

        step += 1
        print(step, info)
    return frames


def display_frames_as_gif(frames):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=0.1)
    anim.save('/home/kingham/文档/CSLEARN/Python/MIndSpore/PlayAtariMindSpore/Imgs/breakout_result.gif', writer='pillow', fps=10)


frames = Playing_Atari()
env.close()
gif_env.close()
display_frames_as_gif(frames)

