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
from q_learning import chart_lines, lake_location


def get_terminal_states(lake_map):
    states = "".join(lake_map)
    return [i for i, s in enumerate(states) if s in ["H", "G"]]


def chart_lake_frequencies(info, episode, title, location=lake_location):
    df = pd.melt(pd.DataFrame(info), "Iteration")
    frequency = np.array(df[(df["variable"] == "S_Freq") & (df["Episode"] == episode)]["value"])
    freq_sum = frequency.sum()
    freq_actions = frequency.sum(axis=0) / freq_sum
    freq_states = frequency.sum(axis=1) / freq_sum
    freq_states = np.reshape(freq_states, (4, 4))

    ax = sns.heatmap(
        freq_states[-1],
        cmap="flare",
        cbar_kws={"format": "%.0f%%", "label": "Frequency Percent"},
    )
    plt.show()
    return

    # ax.set_ylabel("Iteration (10000)")
    ax.set_xlabel("State")
    ax.set_title(title)
    suptitle = "Lake Frequencies States"
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)


def chart_lake_frequencies_old(info, title, location=lake_location):
    df = pd.melt(pd.DataFrame(info), "Iteration")
    df = df[df["Iteration"] % 10000 == 0 & ("S_Freq")]

    frequencies = []
    epsilons = []
    for i in range(df["Iteration"].min(), df["Iteration"].max(), 10000):
        f = np.array(df[(df["variable"] == "S_Freq") & (df.Iteration == i)]["value"])
        frequencies.append(f.sum(axis=0))
        e = np.array(df[(df["variable"] == "Epsilon") & (df.Iteration == i)]["value"])
        epsilons.append(e.mean())
    frequencies = np.array(frequencies)
    epsilons = np.array(epsilons)

    wcs = []
    sfs = []
    for i in range(1, len(frequencies)):
        freq = frequencies[i] - frequencies[i - 1]
        freq_sum = freq.sum()
        wait_cut = freq.sum(axis=0) / freq_sum
        wcs.append(wait_cut)
        sf = freq.sum(axis=1) / freq_sum
        sfs.append(sf)
        # print(f"{i:2}: {epsilons[i]:0.3f} || {wait_cut[0]:0.3f} {wait_cut[1]:0.3f} || {sf[0]:0.4f} {sf[1]:0.4f} {sf[2]:0.4f} {sf[3]:0.4f} {sf[4]:0.4f} {sf[5]:0.4f} {sf[6]:0.4f} ")

    sfs = np.array(sfs) * 100
    sfs = np.reshape(sfs, (4, 4))

    ax = sns.heatmap(
        sfs[-1],
        cmap="flare",
        cbar_kws={"format": "%.0f%%", "label": "Frequency Percent"},
    )
    # ax.set_ylabel("Iteration (10000)")
    ax.set_xlabel("State")
    ax.set_title(title)
    suptitle = "Lake Frequencies States"
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)


###############################
# LAKE
###############################
gamma = 0.9
map_name = "Small"
lake_map = maps[map_name]
is_slippery = True
P, R = example.openai("FrozenLake-v1", desc=lake_map, is_slippery=is_slippery)
title_settings = f"(Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, {map_name} Map)"

terminal_states = get_terminal_states(lake_map)


n_iter = 1000000
alpha_decay = 0.999999
epsilon_decay = 0.999999


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
pprint(episode_stats[-2])


episode = 65000
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
