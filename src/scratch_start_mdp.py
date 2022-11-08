import sys

sys.path.insert(0, "src/")

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

lake_location = "results/lake"
forest_location = "results/forest"


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


def chart_policy_vs_value(policy, value, title=None):
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
    plt.show()


def chart_reward_vs_error(info):
    df = pd.DataFrame(info)

    fig, ax1 = plt.subplots()
    sns.lineplot(df["Reward"], label="Reward", color="g", ax=ax1)
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    sns.lineplot(df["Error"], label="Error", color="r", ax=ax2)
    ax2.legend(loc="upper left")
    ax1.set_xlabel("Iterations")
    plt.show()


def chart_change_vs_iteration(info, title=None):
    # dfv = value_from_dict(info)
    dfv = df_from_info(info, ["Value"])
    dfvp = dfv.pct_change().fillna(0)
    print(dfvp)
    dfp = df_from_info(info, ["Policy"])
    dfp = df_string_policy(dfp)
    print(dfp)
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
    ax.set_title("Change vs Iteration")
    plt.show()


def chart_value_vs_iteration(info, title=None):
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
    )
    ax.set_yticklabels([int(a) for a in ax.get_yticks()], rotation=0)
    ax.set_ylabel("Iterations")
    ax.set_xlabel("Years")
    ax.set_title("Value vs Iteration")
    plt.show()


def percent_fire_review(S=10, gamma=0.9, wait_reward=4, cut_reward=2):
    title = f"Gamma:{gamma}"
    p_V = {}
    p_P = {}
    for i in range(1, 10):
        p = round(i * 0.1, 1)
        P, R = example.forest(S=S, p=p, r1=wait_reward, r2=cut_reward)
        vi = mdp.PolicyIteration(P, R, gamma)
        vi.setVerbose()
        info = vi.run()
        p_V[p] = vi.V
        p_P[p] = string_policy(vi.policy)
    chart_policy_vs_value(p_P, p_V, title=title)
    return info


# info = percent_fire_review(S=5, gamma=0.9)

# chart_reward_vs_error(info)
# chart_change_vs_iteration(info)
# chart_value_vs_iteration(info)


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
    S_max=10, forest_fire_percent=0.1, gamma=0.9, gamma_range=None, wait_reward=4, cut_reward=2
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
    ax.set_xlabel("States")
    ax.set_ylabel("Time")
    plt.show()

    ax = sns.lineplot(df["pi_iter"], label="Policy Iteration", color="g")
    sns.lineplot(df["vi_iter"], label="Value Iteration", color="y")
    ax.set_xlabel("States")
    ax.set_ylabel("Iterations")
    plt.show()

    if gamma_range is not None:
        S = 10
        review = {}
        for gamma in gamma_range:
            print(gamma)
            review[gamma] = {}
            P, R = example.forest(S=S, p=forest_fire_percent, r1=wait_reward, r2=cut_reward)
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
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Time")
        plt.show()

        ax = sns.lineplot(df["pi_iter"], label="Policy Iteration", color="g")
        sns.lineplot(df["vi_iter"], label="Value Iteration", color="y")
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Iterations")
        plt.show()

    return df


def map_to_array(map, size, as_int=True):
    map_array = []
    for row in map:
        for letter in row:
            map_array.append(letter)
    map_array = np.reshape(map_array, (size, size))

    if as_int:
        map_array[map_array == "F"] = 0
        map_array[map_array == "S"] = 0
        map_array[map_array == "G"] = 1
        map_array[map_array == "H"] = 2
        map_array = map_array.astype("int")

    return map_array


def lake_policy_as_string(policy):
    # policy = np.round(np.reshape(policy, (8, 8)), 4)
    policy = np.reshape(policy, (8, 8))
    pp = policy.astype("object")
    pp[pp == 0] = "<"
    pp[pp == 1] = "v"
    pp[pp == 2] = ">"
    pp[pp == 3] = "^"
    return pp


def lake_plot_policy(
    policy, title, suptitle="Lake Plot Policy", red_direction=False, location=lake_location
):
    direction_color = "red" if red_direction else "orange"
    pp = lake_policy_as_string(policy)
    map_array = map_to_array(lake_map, 8)
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
    suptitle="Lake Value vs Policy",
    red_direction=False,
    location=lake_location,
):
    direction_color = "red" if red_direction else "orange"
    pp = lake_policy_as_string(policy)

    value = np.reshape(value, (8, 8))

    cp = sns.color_palette(["white", "green", "blue"])
    ax = sns.heatmap(
        value,
        annot=pp,
        fmt="",
        cbar=True,
        norm=LogNorm(),
        cmap="BuGn",
        linewidth=0.5,
        linecolor="black",
        cbar_kws={"label": "Value"},
        annot_kws={"fontsize": 12, "weight": "bold", "color": direction_color},
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    suptitle = "Lake Value vs Policy"
    plt.suptitle(suptitle)
    save_to_file(plt, suptitle + " " + title, location)


def compare_two_policies(policy_1, policy_2, title, suptitle):
    mask = policy_1 == policy_2
    m = np.array(mask)
    m.sum()
    p = policy_2.astype("object")
    p[m] = ""
    ps = lake_policy_as_string(p)
    lake_plot_policy(ps, title, suptitle=suptitle, red_direction=True, location=lake_location)


def compare_two_policies_and_values(policy_1, policy_2, value_1, value_2, title, suptitle):
    mask = policy_1 == policy_2
    m = np.array(mask)
    m.sum()
    p = policy_2.astype("object")
    p[m] = ""
    ps = lake_policy_as_string(p)
    value = value_2 - value_1
    lake_plot_policy_and_value(
        ps, value, title, suptitle=suptitle, red_direction=True, location=lake_location
    )


def compare_policy_iterations(info, suptitle):
    prev_policy = info[0]["Policy"]
    for i in range(1, len(info)):
        curr_policy = info[i]["Policy"]
        title = f"{i-1} vs {i}"
        compare_two_policies(prev_policy, curr_policy, title, suptitle)
        prev_policy = curr_policy


def plot_policy_value_iterations(info, suptitle):
    for i in range(1, len(info)):
        iteration = info[i]["Iteration"]
        policy = info[i]["Policy"]
        value = info[i]["Value"]
        title = f"Iteration {iteration}"
        lake_plot_policy_and_value(policy, value, title, suptitle=suptitle)


lake_plot_policy(vi.policy, "TEST")


df = compare_vi_pi(S_max=250, gamma_range=[0.1, 0.5, 0.9, 0.99])

info = vi_run(S=5, gamma=0.9)
pprint(info)
info = pi_run(S=5, gamma=0.9)
pprint(info)


###################################
gamma = 0.9
r1 = 4
r2 = 2
p = 0.1
P, R = example.forest(S=S, p=p, r1=r1, r2=r2)
pi = mdp.PolicyIteration(P, R, gamma)
info = pi.run()
# pprint(info)
df = value_from_dict(info)
# df.diff().max(axis=1)
print(f"PI time:{pi.time:.5f} iter:{pi.iter}")

###############################
# LAKE
###############################
gamma = 0.9
lake_map = generate_random_map(size=8)

P, R = example.openai("FrozenLake-v1", desc=lake_map, is_slippery=True)
vi = mdp.ValueIteration(P, R, gamma)
vi_info = vi.run()
len(vi_info)

pi = mdp.PolicyIteration(P, R, gamma, max_iter=10)
pi_info = pi.run()
len(pi_info)


lake_plot_policy(vi.policy, "VI Test")
lake_plot_policy_and_value(vi.policy, vi.V, "VI Test")

lake_plot_policy(pi.policy, "PI Test")
lake_plot_policy_and_value(pi.policy, pi.V, "PI Test")

compare_two_policies_and_values(
    pi_info[-1]["Policy"],
    vi_info[-1]["Policy"],
    pi_info[-1]["Value"],
    vi_info[-1]["Value"],
    "PI vs VI",
    "Policy and Value Differences",
)

compare_policy_iterations(vi_info, "Value Iteration Policy Comparison")
compare_policy_iterations(pi_info, "Policy Iteration Policy Comparison")

plot_policy_value_iterations(pi_info, "Policy Iteration")
plot_policy_value_iterations(vi_info, "Value Iteration")

compare_two_policies(
    pi_info[-1]["Policy"],
    vi_info[-1]["Policy"],
    "Policy Iteration vs Value Iteration",
    "Policy Differences",
)
compare_two_policies(
    vi_info[-1]["Policy"],
    pi_info[-1]["Policy"],
    "Value Iteration vs Policy Iteration",
    "Policy Differences",
)


lake_plot_policy_and_value(vi.policy, vi.V, "Test")

vi.policy
vi.V
