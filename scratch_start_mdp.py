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


S = 10
p = 0.3
gamma = 0.9
P, R = example.forest(S=S, p=p)
vi = mdp.ValueIteration(P, R, gamma)
# vi.setVerbose()
info = vi.run()
print(vi.V)
pprint(info)
df = pd.DataFrame.from_dict(info)
df.columns
vi.policy

max_iter = 1000
S = 10
gamma = 0.9
r1 = 4
r2 = 2
title = f"Gamma:{gamma}"
p_V = {}
p_P = {}
for i in range(1, 2):
    p = round(i * 0.1, 1)
    P, R = example.forest(S=S, p=p, r1=r1, r2=r2)
    vi = mdp.ValueIteration(P, R, gamma, max_iter=max_iter)
    vi.setVerbose()
    print(f"MAX ITER:{vi.max_iter}")
    info = vi.run()
    p_V[p] = vi.V
    p_P[p] = string_policy(vi.policy)

dfV = pd.DataFrame.from_dict(p_V).T
dfP = pd.DataFrame.from_dict(p_P).T

ax = sns.heatmap(dfV, annot=dfP, fmt="", cbar_kws={"label": "Value"})
ticks = ax.set_yticklabels([f"{x:.0%}" for x in dfV.index], va="center", rotation=0)
ax.set_ylabel("Chance of Forest Fire")
ax.set_xlabel("Years")
ax.set_title(title)
plt.show()

info
df = pd.DataFrame(info)

fig, ax1 = plt.subplots()
sns.lineplot(df["Reward"], label="Reward", color="g", ax=ax1)
ax1.legend(loc="upper right")
ax2 = ax1.twinx()
sns.lineplot(df["Error"], label="Error", color="r", ax=ax2)
ax2.legend(loc="upper left")
plt.show()
df["Error"]

dfv = value_from_dict(info)
dfv
dfvp = dfv.pct_change()
# First row NAN
dfvp = dfvp.drop([1])
# ax = sns.heatmap(dfvp, annot=True, fmt=".1f", norm=LogNorm(), cbar_kws={"label": "Pct Change"})
# ticks = ax.set_yticklabels([f"{x:.0%}" for x in dfV.index], va="center", rotation=0)
ax = sns.heatmap(dfvp, norm=LogNorm(), cbar_kws={"label": "Pct Change"})
ax.set_ylabel("Chance of Forest Fire")
ax.set_xlabel("Years")
ax.set_title(title)
plt.show()

pprint(info)

gamma = 0.9
r1 = 4
r2 = 2
p = 0.1
S = 100
for gamma in range(1, 10):
    P, R = example.forest(S=S, p=p, r1=r1, r2=r2)
    vi = mdp.ValueIteration(P, R, 0.9 + gamma * 0.01)
    print(f"MAX ITER:{vi.max_iter}")
    info = vi.run()
    print(f"ITER:{vi.iter}")


S = 100

gamma = 0.9
r1 = 4
r2 = 2
p = 0.1
epsilon = 0.01
P, R = example.forest(S=S, p=p, r1=r1, r2=r2)
vi = mdp.ValueIteration(P, R, gamma, epsilon=epsilon)
info = vi.run()
df = value_from_dict(info)
# df.diff().max(axis=1)
print(f"VI time:{vi.time:.5f} iter:{vi.iter}")


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
vi.policy
p = np.round(np.reshape(vi.policy, (8, 8)), 4)
ax = sns.heatmap(p, cbar_kws={"label": "Value"})
plt.show()
