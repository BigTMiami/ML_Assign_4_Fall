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


def get_Q(P, R, V, actions, states, gamma):
    Q = np.empty((actions, states))
    for aa in range(actions):
        Q[aa] = R[aa] + gamma * P[aa].dot(V)
    return Q


def get_size(a):
    return int(sqrt(len(a)))


def string_policy(policy):
    tree = ""
    ax = "X"
    return [tree if i == 0 else ax for i in policy]


def df_string_policy(df):
    switch = {0: "", 1: "X"}
    return df.replace(switch)


def df_from_info(info, columns):
    info_dict = {}
    for item in info:
        iteration = item["Iteration"]
        for column in columns:
            info_dict[iteration] = item[column]
    df = pd.DataFrame.from_dict(info_dict, orient="index")
    return df


def value_from_dict(info_array):
    value_dict = {}
    for item in info_array:
        iteration = item["Iteration"]

        value = item["Value"]
        value_dict[iteration] = value
    df = pd.DataFrame.from_dict(value_dict, orient="index")
    return df


def reward_error_from_dict(info_array):
    out_dict = {}
    for item in info_array:
        iteration = item["Iteration"]
        out_dict[iteration] = {}
        out_dict[iteration]["reward"] = item["reward"]

    df = pd.DataFrame.from_dict(out_dict, orient="index")
    return df


def chart_policy_vs_value(policy, value, title, suptitle):
    dfV = pd.DataFrame.from_dict(value).T
    dfP = pd.DataFrame.from_dict(policy).T
    ax = sns.heatmap(
        dfV,
        annot=dfP,
        fmt="",
        cbar_kws={"label": "Value"},
        annot_kws={"weight": "bold", "color": "red"},
    )
    ax.set_yticklabels([f"{x:.0%}" for x in dfV.index], va="center", rotation=0)
    ax.set_ylabel("Chance of Forest Fire")
    ax.set_xlabel("Years")
    ax.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, forest_location)


def chart_reward_vs_error(info, title, suptitle, location):
    df = pd.DataFrame(info)

    fig, ax1 = plt.subplots()
    sns.lineplot(df["Reward"], label="Reward", color="g", ax=ax1)
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    sns.lineplot(df["Error"], label="Error", color="r", ax=ax2)
    ax2.legend(loc="upper left")
    ax1.set_xlabel("Iterations")
    ax1.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)


def chart_change_vs_iteration(info, suptitle=None):
    # dfv = value_from_dict(info)
    dfv = df_from_info(info, ["Value"])
    dfvp = dfv.pct_change().fillna(0)
    dfp = df_from_info(info, ["Policy"])
    dfp = df_string_policy(dfp)
    ax = sns.heatmap(
        dfvp,
        annot=dfp,
        fmt="",
        norm=LogNorm(),
        cmap="BuGn",
        cbar_kws={"label": "Pct Change"},
        annot_kws={"weight": "bold", "color": "red"},
    )
    ax.set_yticklabels([int(a) for a in ax.get_yticks()], rotation=0)
    ax.set_ylabel("Iterations")
    ax.set_xlabel("Years")
    title = "Value Change vs Iteration"
    ax.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, forest_location)


def chart_value_vs_iteration(info, suptitle=None):
    # dfv = value_from_dict(info)
    dfv = df_from_info(info, ["Value"])
    dfp = df_from_info(info, ["Policy"])
    dfp = df_string_policy(dfp)
    ax = sns.heatmap(
        dfv,
        annot=dfp,
        fmt="",
        cmap="BuGn",
        cbar_kws={"label": "Value"},
        annot_kws={"weight": "bold", "color": "red"},
        # linewidths=0.02,
        # linecolor='black',
    )
    ax.set_yticklabels([int(a) for a in ax.get_yticks()], rotation=0)
    ax.set_ylabel("Iteration")
    ax.set_xlabel("Year")
    title = "Value vs Iteration"
    ax.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, forest_location)


def percent_fire_review(
    fire_percents=[0.01, 0.05, 0.1, 0.14, 0.18, 0.2, 0.4, 0.8],
    S=10,
    gamma=0.9,
    wait_reward=4,
    cut_reward=2,
):
    title = f"Policy Iteration (Gamma:{gamma}, Wait Reward:{wait_reward})"
    suptitle = "Chance of Forest Fire Review"
    p_V = {}
    p_P = {}
    for p in fire_percents:
        P, R = example.forest(S=S, p=p, r1=wait_reward, r2=cut_reward)
        vi = mdp.PolicyIteration(P, R, gamma)
        vi.setVerbose()
        info = vi.run()
        p_V[p] = vi.V
        p_P[p] = string_policy(vi.policy)
    chart_policy_vs_value(p_P, p_V, title, suptitle)
    return info


def vi_run(S=10, forest_fire_percent=0.1, gamma=0.9, wait_reward=4, cut_reward=2):
    P, R = example.forest(S=S, p=forest_fire_percent, r1=wait_reward, r2=cut_reward)
    pi = mdp.ValueIteration(P, R, gamma=gamma)
    info = pi.run()
    chart_reward_vs_error(info)
    chart_change_vs_iteration(info)
    chart_value_vs_iteration(info)
    return info


def pi_run(S=10, forest_fire_percent=0.1, gamma=0.9, wait_reward=4, cut_reward=2):
    P, R = example.forest(S=S, p=forest_fire_percent, r1=wait_reward, r2=cut_reward)
    pi = mdp.PolicyIteration(P, R, gamma=gamma)
    info = pi.run()
    chart_reward_vs_error(info)
    chart_change_vs_iteration(info)
    chart_value_vs_iteration(info)
    return info


def compare_vi_pi(
    suptitle="Forest MDP Value Iteration vs Policy Iteration",
    S_max=10,
    forest_fire_percent=0.1,
    gamma=0.9,
    gamma_range=None,
    gamma_S=10,
    wait_reward=4,
    cut_reward=2,
    location=forest_location,
):
    review = {}
    for S in range(2, S_max):
        review[S] = {}
        P, R = example.forest(S=S, p=forest_fire_percent, r1=wait_reward, r2=cut_reward)
        pi = mdp.PolicyIteration(P, R, gamma=gamma)
        pi.run()
        review[S]["pi_time"] = pi.time
        review[S]["pi_iter"] = pi.iter

        vi = mdp.ValueIteration(P, R, gamma=gamma)
        vi.run()
        review[S]["vi_time"] = vi.time
        review[S]["vi_iter"] = vi.iter

    df = pd.DataFrame.from_dict(review, orient="index")

    ax = sns.lineplot(df["pi_time"], label="Policy Iteration", color="g")
    sns.lineplot(df["vi_time"], label="Value Iteration", color="y")
    ax.set_xlabel("# of States")
    ax.set_ylabel("Time")
    title = f"# of States vs Time (Gamma={gamma})"
    ax.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)

    ax = sns.lineplot(df["pi_iter"], label="Policy Iteration", color="g")
    sns.lineplot(df["vi_iter"], label="Value Iteration", color="y")
    ax.set_xlabel("# of States")
    ax.set_ylabel("Iterations")
    title = f"# States vs Iterations (Gamma={gamma})"
    ax.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)

    if gamma_range is not None:
        review = {}
        for gamma in gamma_range:
            review[gamma] = {}
            P, R = example.forest(S=gamma_S, p=forest_fire_percent, r1=wait_reward, r2=cut_reward)
            pi = mdp.PolicyIteration(P, R, gamma=gamma)
            pi.run()
            review[gamma]["pi_time"] = pi.time
            review[gamma]["pi_iter"] = pi.iter

            vi = mdp.ValueIteration(P, R, gamma=gamma)
            vi.run()
            review[gamma]["vi_time"] = vi.time
            review[gamma]["vi_iter"] = vi.iter

        df = pd.DataFrame.from_dict(review, orient="index")

        ax = sns.lineplot(df["pi_time"], label="Policy Iteration", color="g")
        sns.lineplot(df["vi_time"], label="Value Iteration", color="y")
        ax.set_xlabel("Gamma (log)")
        plt.xscale("log")
        ax.set_ylabel("Time")
        title = f"Gamma vs Time (# of States={gamma_S})"
        ax.set_title(title)
        plt.suptitle(suptitle)
        save_to_file(plt, suptitle + " " + title, location)

        ax = sns.lineplot(df["pi_iter"], label="Policy Iteration", color="g")
        sns.lineplot(df["vi_iter"], label="Value Iteration", color="y")
        ax.set_xlabel("Gamma (log)")
        plt.xscale("log")
        ax.set_ylabel("Iterations")
        title = f"Gamma vs Iterations (# of States={gamma_S})"
        ax.set_title(title)
        plt.suptitle(suptitle)
        save_to_file(plt, suptitle + " " + title, location)

    return


def map_to_array(map, size, as_int=True):
    map_array = []
    for row in map:
        for letter in row:
            map_array.append(letter)
    size = get_size(map_array)
    map_array = np.reshape(map_array, (size, size))

    if as_int:
        map_array[map_array == "F"] = 0
        map_array[map_array == "S"] = 0
        map_array[map_array == "G"] = 1
        map_array[map_array == "H"] = 2
        map_array = map_array.astype("int")

    return map_array


def lake_policy_as_string(policy):
    size = get_size(policy)
    pp = np.reshape(policy, (size, size))
    pp = pp.astype("object")
    pp[pp == 0] = "<"
    pp[pp == 1] = "v"
    pp[pp == 2] = ">"
    pp[pp == 3] = "^"
    return pp


def lake_plot_policy(
    policy,
    map_used,
    title_settings,
    suptitle="Lake Plot Policy",
    red_direction=False,
    location=lake_location,
):
    direction_color = "red" if red_direction else "orange"
    pp = lake_policy_as_string(policy)
    map_array = map_to_array(map_used, 8)
    cp = sns.color_palette(["white", "green", "blue"])
    ax = sns.heatmap(
        map_array,
        annot=pp,
        fmt="",
        cbar=False,
        cmap=cp,
        linewidth=0.5,
        linecolor="black",
        annot_kws={"fontsize": 12, "weight": "bold", "color": direction_color},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)


def lake_plot_policy_and_value(
    policy,
    value,
    title,
    map_used,
    suptitle="Lake Value vs Policy",
    red_direction=False,
    location=lake_location,
    ax=None,
    cbar=True,
    vmin=None,
    vmax=None,
    show_policy=True,
    cbar_ax=None,
):
    save_chart = ax is None
    if save_chart:
        fig, ax = plt.subplots()
    direction_color = "red" if red_direction else "orange"

    size = get_size(value)
    value = np.reshape(value, (size, size))
    pp = (
        lake_policy_as_string(policy) if show_policy else np.full((size, size), "", dtype="object")
    )

    ax = sns.heatmap(
        value,
        annot=pp,
        fmt="",
        cbar=cbar,
        norm=LogNorm(),
        cmap="BuGn",
        linewidth=0.5,
        linecolor="black",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Value"},
        annot_kws={"fontsize": 12, "weight": "bold", "color": direction_color},
        ax=ax,
        cbar_ax=cbar_ax,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    if save_chart:
        plt.suptitle(suptitle)
        save_to_file(plt, suptitle + " " + title, location)


def compare_two_policies(policy_1, policy_2, title, suptitle, map_used):
    mask = policy_1 == policy_2
    m = np.array(mask)
    p = policy_2.astype("object")
    p[m] = ""
    lake_plot_policy(
        p, title, map_used, suptitle=suptitle, red_direction=True, location=lake_location
    )


def compare_two_policies_and_values(
    policy_1, policy_2, value_1, value_2, title, suptitle, map_used
):
    mask = policy_1 == policy_2
    m = np.array(mask)
    p = policy_2.astype("object")
    p[m] = ""
    value = value_2 - value_1
    # Need to make sure all values are positive for log value
    value = value - np.min(value)
    lake_plot_policy_and_value(
        p, value, title, map_used, suptitle=suptitle, red_direction=True, location=lake_location
    )


def compare_policy_iterations(info, suptitle, map_used, max_iteration=None):
    max_iteration = max_iteration if max_iteration is not None else len(info)
    prev_policy = info[0]["Policy"]
    for i in range(1, max_iteration):
        curr_policy = info[i]["Policy"]
        title = f"{i-1} vs {i}"
        compare_two_policies(prev_policy, curr_policy, title, suptitle, map_used)
        prev_policy = curr_policy


def value_max_min(info):
    vmin = 100000
    vmax = 0
    for info_iter in info:
        vmin = min(vmin, np.min(info_iter["Value"]))
        vmax = max(vmax, np.max(info_iter["Value"]))
    return vmin, vmax


def plot_policy_value_iterations(
    info, suptitle, map_used, iters_to_use=None, seperate_charts=True
):
    iters_to_use = range(1, len(info)) if iters_to_use is None else iters_to_use

    if not seperate_charts:
        fig, ax = plt.subplots(1, len(iters_to_use), figsize=(3 * len(iters_to_use), 4))
        curr_ax = 0
        last_ax = len(iters_to_use) - 1
        vmin, vmax = value_max_min(pi_info)

    for i in iters_to_use:
        iteration = info[i]["Iteration"]
        policy = info[i]["Policy"]
        value = info[i]["Value"]
        title = f"Iteration {iteration}"
        if seperate_charts:
            lake_plot_policy_and_value(
                policy,
                value,
                title,
                map_used,
                suptitle=suptitle,
                vmin=None,
                vmax=None,
                cbar=True,
                ax=None,
            )
        else:
            lake_plot_policy_and_value(
                policy,
                value,
                title,
                map_used,
                suptitle=suptitle,
                vmin=vmin,
                vmax=vmax,
                cbar=(curr_ax == last_ax),
                ax=ax[curr_ax],
            )
            curr_ax += 1
    if not seperate_charts:
        plt.suptitle(suptitle)
        title = f"Iteration {iters_to_use[0]} to {iters_to_use[-1]} "
        save_to_file(plt, suptitle + " " + title, lake_location)


def plot_gamma_iterations(gamma_values, suptitle, map_name, is_slippery, model_type):
    map_used = maps[map_name]

    P, R = example.openai("FrozenLake-v1", desc=map_used, is_slippery=is_slippery)
    title_settings = f"({'Is' if is_slippery else 'Not'} Slippery, {map_name} Map)"

    gamma_info = []
    for gamma in gamma_values:
        if model_type == "vi":
            model = mdp.ValueIteration(P, R, gamma)
        elif model_type == "pi":
            model = mdp.PolicyIteration(P, R, gamma, max_iter=100)
        else:
            raise Exception(f"Unsupported model type of {model_type}")
        info = model.run()
        gamma_info.append(info[-1])

    fig, ax = plt.subplots(1, len(gamma_values), figsize=(3 * len(gamma_values), 4))
    curr_ax = 0
    last_ax = len(gamma_values) - 1
    vmin, vmax = value_max_min(gamma_info)

    for gamma, info in zip(gamma_values, gamma_info):
        policy = info["Policy"]
        value = info["Value"]
        title = f"Gamma {gamma}"
        cbar = curr_ax == last_ax
        cbar_ax = None
        lake_plot_policy_and_value(
            policy,
            value,
            title,
            map_used,
            suptitle=suptitle,
            vmin=vmin,
            vmax=vmax,
            cbar=cbar,
            ax=ax[curr_ax],
            show_policy=False,
            cbar_ax=cbar_ax,
        )
        curr_ax += 1

    plt.suptitle(suptitle + " " + title_settings)
    title = f"gamma {gamma_values[0]} to {gamma_values[-1]} "
    save_to_file(plt, suptitle + " " + title, lake_location)


###############################
# Forest
###############################
gamma = 0.9
r1 = 4
r2 = 2
p = 0.1
S = 7
P, R = example.forest(S=S, p=p, r1=r1, r2=r2)

pi = mdp.PolicyIteration(P, R, gamma)
pi_info = pi.run()
len(pi_info)

vi = mdp.ValueIteration(P, R, gamma)
vi_info = vi.run()
len(vi_info)

# pprint(info)
chart_value_vs_iteration(pi_info, suptitle=f"Policy Iteration (Gamma:{gamma})")
chart_value_vs_iteration(vi_info, suptitle=f"Value Iteration (Gamma:{gamma})")

chart_change_vs_iteration(pi_info, suptitle=f"Policy Iteration (Gamma:{gamma})")
chart_change_vs_iteration(vi_info, suptitle=f"Value Iteration (Gamma:{gamma})")

chart_reward_vs_error(
    pi_info, "Reward vs Error", f"Policy Iteration (Gamma:{gamma})", forest_location
)
chart_reward_vs_error(
    vi_info, "Reward vs Error", f"Value Iteration (Gamma:{gamma})", forest_location
)

percent_fire_review()
percent_fire_review(gamma=0.95)
percent_fire_review(wait_reward=8)

compare_vi_pi(S_max=500, gamma_range=[0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999])

###############################
# LAKE
###############################
gamma = 0.99
map_name = "Large"
map_used = maps[map_name]
is_slippery = True
P, R = example.openai("FrozenLake-v1", desc=map_used, is_slippery=is_slippery)
title_settings = f"(Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, {map_name} Map)"
title_settings


vi = mdp.ValueIteration(P, R, gamma)
vi_info = vi.run()
len(vi_info)

pi = mdp.PolicyIteration(P, R, gamma, max_iter=100)
pi_info = pi.run()
len(pi_info)
for i in range(50):
    print(f"{i:2}: {pi_info[i]['Error']}")

lake_plot_policy(vi.policy, "VI Test", map_used)
lake_plot_policy_and_value(vi.policy, vi.V, "VI Test", map_used, show_policy=False)

lake_plot_policy(pi.policy, "PI Test", map_used)
lake_plot_policy_and_value(pi.policy, pi.V, "PI Test", map_used, show_policy=False)

compare_two_policies(
    pi_info[-1]["Policy"],
    vi_info[-1]["Policy"],
    "PI vs VI",
    "Policy Differences",
    map_used,
)

compare_two_policies_and_values(
    pi_info[-1]["Policy"],
    vi_info[-1]["Policy"],
    pi_info[-1]["Value"],
    vi_info[-1]["Value"],
    "PI vs VI",
    "Policy and Value Differences",
    map_used,
)

compare_policy_iterations(vi_info, "Value Iteration Policy Comparison", map_used)
compare_policy_iterations(
    pi_info, "Policy Iteration Policy Comparison", map_used, max_iteration=12
)

plot_policy_value_iterations(
    pi_info, "Policy Iteration", map_used, seperate_charts=False, iters_to_use=[8, 9, 10, 11]
)
plot_policy_value_iterations(
    vi_info, "Value Iteration", map_used, seperate_charts=False, iters_to_use=[0, 3, 6, 9, 13]
)

chart_reward_vs_error(vi_info, "Lake Reward and Error", "Value Iteration", location=lake_location)
chart_reward_vs_error(pi_info, "Lake Reward and Error", "Policy Iteration", location=lake_location)

plot_gamma_iterations(
    [0.1, 0.5, 0.9, 0.99], "Value Iteration Gamma Comparison", "Large", False, "vi"
)
