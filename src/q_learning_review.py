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


def chart_lines(info, lines, title, suptitle, location):
    df = pd.melt(pd.DataFrame(info), "Iteration")
    df = df[df.variable.isin(lines)]

    fig, ax1 = plt.subplots()
    sns.lineplot(data=df, x="Iteration", y="value", hue="variable")

    # ax1.set_xlabel("Iterations")
    ax1.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)


def chart_lines_old(info, lines, title, suptitle, location):
    df = pd.DataFrame(info)

    fig, ax1 = plt.subplots()
    ax = [ax1]
    for line in lines:
        sns.lineplot(df[line], label=line, ax=ax[-1])
        # ax1.legend(loc="upper right")
        ax.append(ax[0].twinx())
    ax1.set_xlabel("Iterations")
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


n_iter = 10000
alpha_decay = 0.9999
alpha_min = 0.001
epsilon_decay = 0.999999

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
len(ql_info)
pprint(ql_info[1000])

ql.S_freq

chart_lines(
    ql_info,
    ["Epsilon", "Alpha"],
    f"epsilon_decay:{epsilon_decay}",
    "Q Learning Decays",
    forest_location,
)

chart_lines(
    ql_info,
    ["Max V", "Mean V", "V[0]"],
    f"epsilon_decay:{epsilon_decay}",
    "Q Learning Values",
    forest_location,
)
