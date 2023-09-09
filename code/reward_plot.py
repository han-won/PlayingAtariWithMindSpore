import numpy as np
import matplotlib.pyplot as plt


File = np.load('../model/checkpoint.npz')
smoothed_rewards = File['smoothed_rewards']
average_rewards = [np.mean(smoothed_rewards[i:i+50]) for i in range(0, smoothed_rewards.shape[0], 50)]

plt.plot(average_rewards)
plt.title("Average Reward on Breakout")
plt.xlabel("Training Epochs")
plt.ylabel("Average Reward per Episode")
plt.show()
