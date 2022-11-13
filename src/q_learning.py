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


def chart_lines(info, lines, title, suptitle, location, review_frequency=100, x="Iteration"):
    df = pd.melt(pd.DataFrame(info), x)
    df = df[df[x] % review_frequency == 0]
    df = df[df.variable.isin(lines)]

    fig, ax1 = plt.subplots()
    sns.lineplot(data=df, x=x, y="value", hue="variable")

    # ax1.set_xlabel("Iterations")
    ax1.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)


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


def reachable_forest_percentages(
    epsilon_values=[0.1, 0.5, 1.0], p=0.1, S=7, location=forest_location
):
    wait_policy = True
    wait_state_policy_percentage = 1.0 if wait_policy else 0.0
    er = len(epsilon_values)
    reachable_percents = np.ones((er, S))
    for i, epsilon in enumerate(epsilon_values):
        wsap = (epsilon * 0.5) + ((1 - epsilon) * wait_state_policy_percentage)
        ptnt = wsap * (1 - p)
        for s in range(1, S):
            reachable_percents[i][s] = reachable_percents[i][s - 1] * ptnt

    fig, [ax1, ax2] = plt.subplots(2, figsize=(7, 4))
    ax1 = sns.heatmap(
        reachable_percents * 100,
        cmap="flare",
        annot=True,
        fmt=".5f",
        # norm=LogNorm(vmin=0.0000001,vmax=100),
        vmin=0.0000001,
        vmax=100,
        # xticklabels=[0, 1, 2, 3, 4, 5, 6],
        yticklabels=epsilon_values,
        cbar=False,
        ax=ax1,
    )
    ax1.set_xticks([])
    ax1.set_ylabel("Epsilon")
    title = f"Policy {'Wait' if wait_policy else 'Cut'}"
    ax1.set_title(title)

    wait_policy = not wait_policy
    wait_state_policy_percentage = 1.0 if wait_policy else 0.0
    er = len(epsilon_values)
    reachable_percents = np.ones((er, S))
    for i, epsilon in enumerate(epsilon_values):
        wsap = (epsilon * 0.5) + ((1 - epsilon) * wait_state_policy_percentage)
        ptnt = wsap * (1 - p)
        for s in range(1, S):
            reachable_percents[i][s] = reachable_percents[i][s - 1] * ptnt

    ax2 = sns.heatmap(
        reachable_percents * 100,
        cmap="flare",
        annot=True,
        fmt=".5f",
        # norm=LogNorm(vmin=0.0000001,vmax=100),
        vmin=0.0000001,
        vmax=100,
        xticklabels=[0, 1, 2, 3, 4, 5, 6],
        yticklabels=epsilon_values,
        cbar=False,
        ax=ax2,
    )
    ax2.set_ylabel("Epsilon")
    ax2.set_xlabel("State")
    title = f"Policy {'Wait' if wait_policy else 'Cut'}"
    ax2.set_title(title)

    suptitle = "Forest States Reachability Percentages"
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle, location)


def forest_q(
    gamma=0.9,
    r1=4,
    r2=2,
    p=0.1,
    S=7,
    n_iter=1000000,
    alpha_decay=0.99,
    epsilon_decay=0.99,
    repeat_count=5,
):

    P, R = example.forest(S=S, p=p, r1=r1, r2=r2)

    ql_all = []
    for i in range(repeat_count):
        print(f"{i} Iteration")
        ql = mdp.QLearning(
            P,
            R,
            gamma,
            start_from_begining=True,
            n_iter=n_iter,
            alpha_decay=alpha_decay,
            epsilon_decay=epsilon_decay,
        )
        ql_info = ql.run()
        ql_all += ql_info

    title = f"(epsilon_decay:{epsilon_decay} alpha_decay:{alpha_decay})"

    chart_lines(
        ql_all,
        ["Epsilon", "Alpha"],
        title,
        "Q Epsilon, Alpha",
        forest_location,
        iter_review_frequency=10000,
    )

    chart_lines(
        ql_all,
        ["Max V", "V[0]"],
        title,
        "Q Max V, V[0]",
        forest_location,
        iter_review_frequency=10000,
    )

    chart_lines(
        ql_all,
        ["running_reward"],
        title,
        "Q Running Reward",
        forest_location,
        iter_review_frequency=1000,
    )

    chart_forest_frequencies(ql_all, title)


def get_terminal_states(lake_map):
    states = "".join(lake_map)
    return [i for i, s in enumerate(states) if s in ["H", "G"]]


def chart_lake_frequencies(
    episode_stats, episode, title, suptitle="Lake State Visit Frequencies", location=lake_location
):
    df = pd.melt(pd.DataFrame(episode_stats), "Episode")
    frequency = np.array(df[(df["variable"] == "S_Freq") & (df["Episode"] == episode)]["value"])[0]
    freq_sum = frequency.sum()
    freq_actions = frequency.sum(axis=0) / freq_sum
    freq_states = frequency.sum(axis=1) / freq_sum
    freq_states = np.reshape(freq_states, (4, 4))

    ax = sns.heatmap(
        freq_states * 100,
        cmap="flare",
        cbar_kws={"format": "%.0f%%", "label": "Frequency Percent"},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)
