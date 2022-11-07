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


def string_policy(policy):
    tree = ""
    ax = "X"
    return [tree if i == 0 else ax for i in policy]

def df_string_policy(df):
    switch = {0:"",1:"X"}
    return df.replace(switch)


def df_from_info(info,columns):
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

def chart_policy_vs_value(policy, value,title=None):
    dfV = pd.DataFrame.from_dict(value).T
    dfP = pd.DataFrame.from_dict(policy).T
    ax = sns.heatmap(dfV, annot=dfP, fmt="", cbar_kws={"label": "Value"})
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

def chart_change_vs_iteration(info,title=None):
    #dfv = value_from_dict(info)
    dfv = df_from_info(info, ["Value"])
    dfvp = dfv.pct_change()
    dfp = df_from_info(info, ["Policy"])
    dfp = df_string_policy(dfp)
    # First row NAN
    #dfvp = dfvp.drop([1])
    # ax = sns.heatmap(dfvp, annot=True, fmt=".1f", norm=LogNorm(), cbar_kws={"label": "Pct Change"})
    # ticks = ax.set_yticklabels([f"{x:.0%}" for x in dfV.index], va="center", rotation=0)
    ax = sns.heatmap(dfvp, annot=dfp, fmt="", norm=LogNorm(), cmap="BuGn", cbar_kws={"label": "Pct Change"})
    ax.set_yticklabels([int(a) for a in ax.get_yticks()], rotation = 0)
    ax.set_ylabel("Iterations")
    ax.set_xlabel("Years")
    ax.set_title(title)
    plt.show()

def chart_value_vs_iteration(info,title=None):
    #dfv = value_from_dict(info)
    dfv = df_from_info(info, ["Value"])
    dfp = df_from_info(info, ["Policy"])
    dfp = df_string_policy(dfp)
    ax = sns.heatmap(dfv, annot=dfp, fmt="", cmap="BuGn", cbar_kws={"label": "Value"})
    ax.set_yticklabels([int(a) for a in ax.get_yticks()], rotation = 0)
    ax.set_ylabel("Iterations")
    ax.set_xlabel("Years")
    ax.set_title(title)
    plt.show()


def percent_fire_review(S=10,gamma = 0.9,wait_reward=4,cut_reward=2 ):
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
    chart_policy_vs_value(p_P, p_V,title=title)
    return info

info = percent_fire_review(S=5,gamma=.9)

chart_reward_vs_error(info)
chart_change_vs_iteration(info)
chart_value_vs_iteration(info)

def vi_run(S=10,forest_fire_percent=0.1, gamma = 0.9,wait_reward=4,cut_reward=2 ):
    P, R = example.forest(S=S, p=forest_fire_percent, r1=wait_reward, r2=cut_reward)
    pi = mdp.ValueIteration(P, R, gamma=gamma)
    info = pi.run()
    chart_reward_vs_error(info)
    chart_change_vs_iteration(info)
    chart_value_vs_iteration(info)
    return info

def pi_run(S=10,forest_fire_percent=0.1, gamma = 0.9,wait_reward=4,cut_reward=2 ):
    P, R = example.forest(S=S, p=forest_fire_percent, r1=wait_reward, r2=cut_reward)
    pi = mdp.PolicyIteration(P, R, gamma=gamma)
    info = pi.run()
    chart_reward_vs_error(info)
    chart_change_vs_iteration(info)
    chart_value_vs_iteration(info)
    return info

def compare_vi_pi(S_max=10,forest_fire_percent=0.1, gamma = 0.9,gamma_range=None, wait_reward=4,cut_reward=2 ):
    review={}
    for S in range(2,S_max):
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

    df = pd.DataFrame.from_dict(review, orient='index')

    ax = sns.lineplot(df["pi_time"],label="Policy Iteration", color="g")
    sns.lineplot(df["vi_time"],label="Value Iteration", color="y")
    ax.set_xlabel("States")
    ax.set_ylabel("Time") 
    plt.show()

    ax = sns.lineplot(df["pi_iter"],label="Policy Iteration", color="g")
    sns.lineplot(df["vi_iter"],label="Value Iteration", color="y")
    ax.set_xlabel("States")
    ax.set_ylabel("Iterations")
    plt.show()

    if gamma_range is not None:
        S = 10
        review={}
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

        df = pd.DataFrame.from_dict(review, orient='index')

        ax = sns.lineplot(df["pi_time"],label="Policy Iteration", color="g")
        sns.lineplot(df["vi_time"],label="Value Iteration", color="y")
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Time")
        plt.show()

        ax = sns.lineplot(df["pi_iter"],label="Policy Iteration", color="g")
        sns.lineplot(df["vi_iter"],label="Value Iteration", color="y")
        ax.set_xlabel("Gamma")
        ax.set_ylabel("Iterations")
        plt.show()        

    return df

df = compare_vi_pi(S_max=250,gamma_range=[.1,.5,.9,.99])

info = vi_run(S=5,gamma=.9)
pprint(info)
info = pi_run(S=5,gamma=.9)
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


#######################################
lake_map = generate_random_map(size=8)
lake_map

P, R = example.openai("FrozenLake-v1", desc=lake_map)
vi = mdp.ValueIteration(P, R, gamma)
info = vi.run()
df = value_from_dict(info)
print(f"VI time:{vi.time:.5f} iter:{vi.iter}")
v = np.round(np.reshape(vi.V, (8, 8)), 4)
info

ax = sns.heatmap(v, cbar_kws={"label": "Value"})
plt.show()


vi.policy
policy = np.round(np.reshape(vi.policy, (8, 8)), 4)
pp = policy.astype("object")
pp
pp[pp==0] = "<"
pp[pp==1] = "v"
pp[pp==2] = ">"
pp[pp==3] = "^"

map_array = map_to_array(lake_map, 8)
map_array.dtype
map_array[map_array=='F'] = 0
map_array[map_array=='S'] = 0
map_array[map_array=='G'] = 1
map_array[map_array=='H'] = 2
map_array = map_array.astype('int')

zeros = np.zeros((8,8))

cp = sns.color_palette(['white','green','blue'])
ax = sns.heatmap(map_array, annot=pp, fmt="", cbar = False, cmap=cp, linewidth=.5, linecolor='black')
plt.xticks([])
plt.yticks([])
plt.show()


def map_to_array(map, size):
    map_array = []
    for row in map:
        for letter in row:
            map_array.append(letter)
    map_array = np.reshape(map_array, (size, size))
    return map_array


def plot_policy_directions(policy, old_policy, map):
    map_policy = 
    ax = sns.heatmap(zeros, annot=map_array, fmt="", cbar_kws={"label": "Value"})
    plt.show()


def plot_policy_directions_old(policy, old_policy, map):
    size = policy.shape[0]
    fig, ax = plt.subplots()
    for start_y, row in enumerate(policy):
        use_y = size - start_y - 1
        for start_x, value in enumerate(row):
            if map[start_y][start_x] == "F":
                if value == 0:
                    # left
                    x = start_x + 0.75
                    dx = -0.4
                    y = use_y + 0.5
                    dy = 0
                elif value == 2:
                    # right
                    x = start_x + 0.25
                    dx = 0.4
                    y = use_y + 0.5
                    dy = 0
                elif value == 1:
                    # down
                    x = start_x + 0.5
                    dx = 0
                    y = use_y + 0.75
                    dy = -0.4
                elif value == 2:
                    # up
                    x = start_x + 0.5
                    dx = 0
                    y = use_y + 0.25
                    dy = 0.4
                old_value = old_policy[start_y][start_x]
                facecolor = "red" if value != old_value else "green"
                plt.arrow(x=x, y=y, dx=dx, dy=dy, width=0.06, facecolor=facecolor)
            elif map[start_y][start_x] == "H":
                ax.add_patch(
                    plt.Circle((start_x + 0.5, use_y + 0.5), 0.4, fill=True, facecolor="blue")
                )
    size = len(policy)
    plt.xlim([0, size])
    plt.ylim([0, size])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.grid()
    plt.show()
    plt.close()


plot_policy_directions(policy, policy, map_array)

for i in range(8):
    row = ""
    for j in range(8):
        row += map_array[i][j]
    print(row)
