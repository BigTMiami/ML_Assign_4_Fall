from pprint import pprint

import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def string_policy(policy):
    tree = ""
    ax = "X"
    return [tree if i == 0 else ax for i in policy]


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

S = 5
gamma = 0.90
r1 = 10
r2 = 2
title = f"Gamma:{gamma}"
p_V = {}
p_P = {}
for i in range(1, 10):
    p = round(i * 0.1, 1)
    P, R = example.forest(S=S, p=p, r1=r1, r2=r2)
    vi = mdp.ValueIteration(P, R, gamma)
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
R
