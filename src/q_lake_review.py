import sys

sys.path.insert(0, "src/")

from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from q_learning import q_lake_run

gamma = 0.9
map_name = "Small"
is_slippery = True
n_iter = 5000000
alpha_decay = 0.999999
epsilon_decay = 0.999999
info = q_lake_run(map_name, gamma, n_iter, alpha_decay, epsilon_decay, is_slippery=is_slippery)
info
info[-1]
pprint(info[-1])


def get_reward_threshold(info, reward_threshold, x="Episode"):
    df = pd.melt(pd.DataFrame(info), x)
    y = "episode_reward"
    dfp = df[df["variable"] == y].copy()
    dfp["moving_avg"] = dfp["value"].rolling(20).mean()
    dfp["percent_chg"] = dfp["moving_avg"].pct_change().rolling(10).mean()
    threshold_episode = dfp[dfp["percent_chg"] < reward_threshold].iloc[0]["Episode"]
    return threshold_episode


def get_threshold(info, threshold, threshold_column, y, x="Episode"):
    df = pd.melt(pd.DataFrame(info), x)
    dfp = df[df["variable"] == y].copy()
    dfp["moving_avg"] = dfp["value"].rolling(20).mean()
    dfp["percent_chg"] = dfp["moving_avg"].pct_change().rolling(10).mean()
    threshold_episode = dfp[dfp[threshold_column] < threshold].iloc[0]["Episode"]
    return threshold_episode


x = "Episode"
error_threshold = 0.001
df = pd.melt(pd.DataFrame(info), x)
y = "Error"
dfp = df[df["variable"] == y].copy()
dfp["moving_avg"] = dfp["value"].rolling(20).mean()
dfp["percent_chg"] = dfp["moving_avg"].pct_change().rolling(10).mean()
dfp.tail(50)
dfp.head(50)


ax = dfp.plot(x="Episode", y=["moving_avg", "value"])


threshold_episode = dfp[dfp["moving_avg"] < error_threshold].iloc[0]["Episode"]

x = "Episode"
reward_threshold = 0.001
df = pd.melt(pd.DataFrame(info), x)
y = "episode_reward"
dfp = df[df["variable"] == y].copy()
dfp["moving_avg"] = dfp["value"].rolling(20).mean()
dfp["percent_chg"] = dfp["moving_avg"].pct_change().rolling(10).mean()

reward_threshold_episode = dfp[dfp["percent_chg"] < reward_threshold].iloc[0]["Episode"]
reward_threshold_episode
test2 = get_threshold(info, 0.001, "percent_chg", "episode_reward")
test2

ax = dfp.plot(x="Episode", y=["percent_chg", "value"], secondary_y=["value"], ylabel="Reward")
ax.set_ylabel("Percent Change")
plt.axvline(x=threshold_episode, color="r", linestyle="-", label="Threshold")
plt.text(threshold_episode + 5000, 0.5, "Convergence Threshold", color="r")
plt.show()

x = "Episode"
error_threshold = 0.001
df = pd.melt(pd.DataFrame(info), x)
y = "Error"
dfp = df[df["variable"] == y].copy()
dfp["moving_avg"] = dfp["value"].rolling(20).mean()
dfp["percent_chg"] = dfp["moving_avg"].pct_change().rolling(10).mean()
dfp.tail(50)
dfp.head(50)


ax = dfp.plot(x="Episode", y=["moving_avg", "value"])


error_threshold_episode = dfp[dfp["moving_avg"] < error_threshold].iloc[0]["Episode"]
error_threshold_episode

test = get_threshold(info, 0.001, "moving_avg", "Error")
test

plt.axvline(x=threshold_episode, color="r", linestyle="-", label="Threshold")
plt.text(threshold_episode + 5000, 0.5, "Convergence Threshold", color="r")
plt.show()


gamma = 0.9
map_name = "Medium"
is_slippery = True
n_iter = 5000000
alpha_decay = 0.999999
epsilon_decay = 0.999999
info = q_lake_run(map_name, gamma, n_iter, alpha_decay, epsilon_decay, is_slippery=is_slippery)

gamma = 0.9
map_name = "Large"
is_slippery = True
n_iter = 10000000
alpha_decay = 0.999999
epsilon_decay = 0.999999
info = q_lake_run(map_name, gamma, n_iter, alpha_decay, epsilon_decay, is_slippery=is_slippery)
