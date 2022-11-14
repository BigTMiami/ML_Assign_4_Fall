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
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from chart_util import save_json_to_file, save_to_file
from maps import maps
from vi_pi_functions import lake_plot_policy_and_value

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
    freq_states = frequency.sum(axis=1) / freq_sum
    policy = np.array(df[(df["variable"] == "Policy") & (df["Episode"] == episode)]["value"])[0]

    lake_plot_policy_and_value(
        policy,
        freq_states,
        title,
        suptitle,
        red_direction=False,
        location=lake_location,
        show_policy=True,
        value_label="Visit Frequency (Percent)",
        save_json=False,
    )


def q_lake_run(
    map_name,
    gamma,
    n_iter,
    alpha_decay,
    epsilon_decay,
    is_slippery=True,
    episode_stat_frequency=1000,
):
    map_used = maps[map_name]
    lake_map = maps[map_name]
    P, R = example.openai("FrozenLake-v1", desc=lake_map, is_slippery=is_slippery)

    terminal_states = get_terminal_states(lake_map)
    title_settings = f"(Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay} Map:{map_name})"

    ql = mdp.QLearningEpisodic(
        P,
        R,
        gamma,
        terminal_states,
        n_iter=n_iter,
        alpha_decay=alpha_decay,
        epsilon_decay=epsilon_decay,
        episode_stat_frequency=episode_stat_frequency,
    )

    ql_info, episode_stats = ql.run()

    chart_lines(
        episode_stats,
        ["episode_reward", "Max V"],
        title_settings,
        "Q Lake Reward and Max V",
        lake_location,
        review_frequency=100,
        x="Episode",
    )

    chart_lines(
        episode_stats,
        ["Error"],
        title_settings,
        "Q Lake Error",
        lake_location,
        review_frequency=100,
        x="Episode",
    )

    chart_lines(
        episode_stats,
        ["Iterations per Episode"],
        title_settings,
        "Q Lake Iterations per Episode",
        lake_location,
        review_frequency=100,
        x="Episode",
    )

    suptitle = f"Q Lake State Visits - {map_name} Map"
    episode = episode_stats[-2]["Episode"]
    freq_title_settings = f"Episode: {episode} (Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay})"
    chart_lake_frequencies(episode_stats, episode, freq_title_settings, suptitle=suptitle)

    episode = 2000
    freq_title_settings = f"Episode: {episode} (Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay})"
    chart_lake_frequencies(episode_stats, episode, freq_title_settings, suptitle=suptitle)

    final_stats = episode_stats[-2]
    for item, value in final_stats.items():
        if isinstance(value, np.ndarray):
            final_stats[item] = value.tolist()
    final_stats["alpha_decay"] = alpha_decay
    final_stats["epsilon_decay"] = epsilon_decay
    final_stats["map_name"] = map_name
    final_stats["is_slippery"] = is_slippery
    save_json_to_file(final_stats, "Q Lake Best " + title_settings, lake_location)

    return final_stats
