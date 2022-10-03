import numpy as np
import matplotlib.pyplot as plt

from skrl.utils import postprocessing


labels = []
rewards = []

# load the Tensorboard files and iterate over them (tag: "Reward / Total reward (mean)")
tensorboard_iterator = postprocessing.TensorboardFileIterator("runs/*/events.out.tfevents.*",
                                                              tags=["Reward / Total reward (mean)"])
for dirname, data in tensorboard_iterator:
    rewards.append(data["Reward / Total reward (mean)"])
    labels.append(dirname)

# convert to numpy arrays and compute mean and std
rewards = np.array(rewards)
mean = np.mean(rewards[:,:,1], axis=0)
std = np.std(rewards[:,:,1], axis=0)

# creae two subplots (one for each reward and one for the mean)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# plot the rewards for each experiment
for reward, label in zip(rewards, labels):
    ax[0].plot(reward[:,0], reward[:,1], label=label)

ax[0].set_title("Total reward (for each experiment)")
ax[0].set_xlabel("Timesteps")
ax[0].set_ylabel("Reward")
ax[0].grid(True)
ax[0].legend()

# plot the mean and std (across experiments)
ax[1].fill_between(rewards[0,:,0], mean - std, mean + std, alpha=0.5, label="std")
ax[1].plot(rewards[0,:,0], mean, label="mean")

ax[1].set_title("Total reward (mean and std of all experiments)")
ax[1].set_xlabel("Timesteps")
ax[1].set_ylabel("Reward")
ax[1].grid(True)
ax[1].legend()

# show and save the figure
plt.show()
plt.savefig("total_reward.png")
