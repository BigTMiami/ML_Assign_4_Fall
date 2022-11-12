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
from q_learning import chart_forest_frequencies, chart_lines, forest_location, forest_q

###############################
# Forest
###############################

forest_q()
forest_q(epsilon_decay=0.99999)
forest_q(epsilon_decay=0.999999)
forest_q(epsilon_decay=0.999998)
forest_q(epsilon_decay=0.999998, alpha_decay=0.99999)
forest_q(epsilon_decay=0.99999, alpha_decay=0.99999)


gamma = 0.9
r1 = 4
r2 = 2
p = 0.1
S = 7
P, R = example.forest(S=S, p=p, r1=r1, r2=r2)


n_iter = 1000000
alpha_decay = 0.99999
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
    iter_review_frequency=10000,
)

chart_forest_frequencies(ql_all, title)
