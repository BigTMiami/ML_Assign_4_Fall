import sys

sys.path.insert(0, "src/")

from math import sqrt
from pprint import pprint

import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from gym.envs.toy_text.frozen_lake import generate_random_map
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from chart_util import save_to_file
from maps import maps
from q_learning import (
    chart_lake_frequencies,
    chart_lines,
    get_terminal_states,
    lake_location,
)

###############################
# LAKE
###############################
gamma = 0.9
map_name = "Small"
lake_map = maps[map_name]
is_slippery = True
P, R = example.openai("FrozenLake-v1", desc=lake_map, is_slippery=is_slippery)

terminal_states = get_terminal_states(lake_map)

n_iter = 5000000
alpha_decay = 0.999999
epsilon_decay = 0.999999

title_settings = (
    f"(Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay})"
)

ql = mdp.QLearningEpisodic(
    P,
    R,
    gamma,
    terminal_states,
    n_iter=n_iter,
    alpha_decay=alpha_decay,
    epsilon_decay=epsilon_decay,
    episode_stat_frequency=1000,
)
ql_info, episode_stats = ql.run()
len(ql_info)
len(episode_stats)
pprint(ql_info[-1])
pprint(episode_stats[-1])
pprint(episode_stats[-3:-1])

chart_lines(
    episode_stats,
    ["episode_reward", "Max V"],
    title_settings,
    "Test",
    lake_location,
    review_frequency=100,
    x="Episode",
)

chart_lines(
    episode_stats,
    ["Error"],
    title_settings,
    "Test Error",
    lake_location,
    review_frequency=100,
    x="Episode",
)


episode = episode_stats[-2]["Episode"]
freq_title_settings = f"Episode: {episode} (Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay})"
chart_lake_frequencies(episode_stats, episode, freq_title_settings)

df = pd.melt(pd.DataFrame(episode_stats), "Episode")
# df[(df["variable"] == "S_Freq") & (df["Episode"]  == episode)]["value"]
# np.array(df[(df["variable"] == "S_Freq") & (df["Episode"]  == episode)]["value"])[0]
frequency_curr = np.array(df[(df["variable"] == "S_Freq") & (df["Episode"] == episode)]["value"])[
    0
]
frequency_prev = np.array(
    df[(df["variable"] == "S_Freq") & (df["Episode"] == episode - 1000)]["value"]
)[0]
frequency = frequency_curr - frequency_prev
freq_sum = frequency.sum()
freq_actions = frequency.sum(axis=0) / freq_sum
freq_states = frequency.sum(axis=1) / freq_sum
freq_states = np.reshape(freq_states, (4, 4))

ax = sns.heatmap(
    freq_states,
    cmap="flare",
    cbar_kws={"format": "%.0f%%", "label": "Frequency Percent"},
)
plt.show()

chart_lake_frequencies(episode_stats, 100, "test")

df = pd.melt(pd.DataFrame(ql_info), "Iteration")
df = df[df["Iteration"] % 1000 == 0]
df[df["variable"] == "State"]
df
f = np.array(df[(df["variable"] == "S_Freq") & (df.Iteration == 100000)]["value"])
f
sf = f[0].sum(axis=1) / f[0].sum()

P[:, 15]

sf = np.reshape(sf, (4, 4))

ax = sns.heatmap(
    sf * 100,
    cmap="flare",
    cbar_kws={"format": "%.0f%%", "label": "Frequency Percent"},
)
plt.show()


title = f"(epsilon_decay:{epsilon_decay} alpha_decay:{alpha_decay})"

chart_lake_frequencies(ql_all, "TEST")

chart_lines(
    ql_all,
    ["Epsilon", "Alpha"],
    title,
    "Q Epsilon, Alpha",
    lake_location,
    iter_review_frequency=10000,
)

chart_lines(
    ql_all,
    ["Max V", "V[0]"],
    title,
    "Q Max V, V[0]",
    lake_location,
    iter_review_frequency=10000,
)

chart_lines(
    ql_all,
    ["running_reward"],
    title,
    "Q Running Reward",
    lake_location,
    iter_review_frequency=10000,
)
