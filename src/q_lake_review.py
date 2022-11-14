import sys

sys.path.insert(0, "src/")

from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from q_learning import chart_lines, get_threshold, q_lake_run
from vi_pi_functions import lake_location

gamma = 0.9
map_name = "Medium"
is_slippery = True
n_iter = 5000000
alpha_decay = 0.999999
epsilon_decay = 0.999999
info = q_lake_run(map_name, gamma, n_iter, alpha_decay, epsilon_decay, is_slippery=is_slippery)
info
info[-1]
pprint(info[-1])

title_settings = f"(Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay} Map:{map_name})"

error_threshold_episode = get_threshold(info, 0.001, "moving_avg", "Error", "Episode")
chart_lines(
    info,
    ["Error"],
    title_settings,
    "TEST Q Lake Error",
    lake_location,
    review_frequency=100,
    x="Episode",
    threshold=error_threshold_episode,
)

reward_threshold_episode = get_threshold(
    info, 0.01, "percent_chg", "episode_reward", "Episode", value_window=50, pct_window=30
)
reward_threshold_episode
chart_lines(
    info,
    ["episode_reward"],
    title_settings,
    "TEST Q Lake Reward",
    lake_location,
    review_frequency=100,
    x="Episode",
    threshold=reward_threshold_episode,
)

threshold_column = "moving_avg"
x = "Error"
y = "Episode"
threshold = 0.01
error_threshold_episode = get_threshold(info, threshold, threshold_column, x, y)

df = pd.melt(pd.DataFrame(info), y)
dfp = df[df["variable"] == x].copy()
dfp["moving_avg"] = dfp["value"].rolling(20).mean()
dfp["percent_chg"] = dfp["moving_avg"].pct_change().rolling(10).mean()
dfp
threshold_episode = dfp[dfp[threshold_column] < threshold].iloc[0][y]


threshold_column = "percent_chg"
x = "episode_reward"
y = "Episode"
threshold = 0.01
threshold_episode = get_threshold(info, threshold, threshold_column, x, y)
threshold_episode

df = pd.melt(pd.DataFrame(info), y)
dfp = df[df["variable"] == x].copy()
dfp["moving_avg"] = dfp["value"].rolling(20).mean()
dfp["percent_chg"] = dfp["moving_avg"].pct_change().rolling(10).mean()
dfp
threshold_episode = dfp[dfp[threshold_column] < threshold].iloc[0][y]
threshold_episode


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
test2 = get_threshold(info, 0.001, "percent_chg", "Episode", "episode_reward")
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
