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

lake_location = "results/lake"
forest_location = "results/forest"
forest_actions = 2


def chart_lines(info, lines, title, suptitle, location, iter_review_frequency=100):
    df = pd.melt(pd.DataFrame(info), "Iteration")
    df = df[df["Iteration"] % iter_review_frequency == 0]
    df = df[df.variable.isin(lines)]

    fig, ax1 = plt.subplots()
    sns.lineplot(data=df, x="Iteration", y="value", hue="variable")

    # ax1.set_xlabel("Iterations")
    ax1.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)


###############################
# Forest
###############################
gamma = 0.9
r1 = 4
r2 = 2
p = 0.1
S = 7
P, R = example.forest(S=S, p=p, r1=r1, r2=r2)


percent_reach = 0.5 * 0.9
curr_state = 1
for i in range(1, 7):
    curr_state = curr_state * percent_reach
    print(f"{i}:{curr_state * 100:8.4f}")


percent_reach = 0.9 * 0.9
curr_state = 1
for i in range(1, 7):
    curr_state = curr_state * percent_reach
    print(f"{i}:{curr_state * 100:8.4f}")


n_iter = 1000000
alpha_decay = 0.99999
alpha_min = 0.001
epsilon_decay = 0.99999

ql_all = []
for i in range(3):
    print(f"{i} Iteration")
    ql = mdp.QLearning(
        P,
        R,
        gamma,
        start_from_begining=True,
        n_iter=n_iter,
        alpha_decay=alpha_decay,
        alpha_min=alpha_min,
        epsilon_decay=epsilon_decay,
    )
    ql_info = ql.run()
    ql_all += ql_info

chart_lines(
    ql_all,
    ["Epsilon", "Alpha"],
    f"epsilon_decay:{epsilon_decay} alpha_decay:{alpha_decay}",
    "Q Learning Decays",
    forest_location,
    iter_review_frequency=10000,
)

chart_lines(
    ql_all,
    ["Max V", "V[0]"],
    f"epsilon_decay:{epsilon_decay} alpha_decay:{alpha_decay}",
    "Q Learning Values",
    forest_location,
    iter_review_frequency=10000,
)

chart_lines(
    ql_all,
    ["running_reward"],
    f"epsilon_decay:{epsilon_decay} alpha_decay:{alpha_decay}",
    "Q Learning Running Reward 1000 Iterations",
    forest_location,
    iter_review_frequency=1000,
)


def chart_forest_frequencies(ql_info, title, location=forest_location):
    df = pd.melt(pd.DataFrame(ql_info), "Iteration")
    df = df[df["Iteration"] % 10000 == 0]

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

    wcs = np.array(wcs) * 100

    ax = sns.heatmap(
        wcs,
        cmap="flare",
        xticklabels=["Wait", "Cut"],
        cbar_kws={"format": "%.0f%%", "label": "Frequency Percent"},
    )
    ax.set_ylabel("Iteration (10000)")
    ax.set_xlabel("Action")
    ax.set_title(title)
    suptitle = "Forest Frequencies Actions"
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)

    sfs = np.array(sfs) * 100

    ax = sns.heatmap(
        sfs,
        cmap="flare",
        xticklabels=[0, 1, 2, 3, 4, 5, 6],
        cbar_kws={"format": "%.0f%%", "label": "Frequency Percent"},
    )
    ax.set_ylabel("Iteration (10000)")
    ax.set_xlabel("State")
    ax.set_title(title)
    suptitle = "Forest Frequencies States"
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)


title = f"(epsilon_decay:{epsilon_decay} alpha_decay:{alpha_decay})"
chart_forest_frequencies(ql_all, title)

df = pd.DataFrame(wcs, columns=["Wait", "Cut"]).melt()

ax = sns.violinplot(df, x=df.index, hue="variable")
plt.show()

a = np.array(wcs)

df = pd.DataFrame(wcs, columns=["Wait", "Cut"]).melt()
ax = sns.barplot(df, y="value", x=df.index, hue="variable")
plt.show()

a = np.array(wcs)
ax = sns.barplot(y=wcs[:, 0])
plt.show()
