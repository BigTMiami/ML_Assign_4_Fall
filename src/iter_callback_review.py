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


class QL_Forest_Runner:
    def __init__(self, gamma=0.9, r1=4, r2=2, p=0.1, S=7):
        self.gamma = gamma
        self.r1 = r1
        self.r2 = r2
        self.p = p
        self.S = S
        self.P, self.R = example.forest(S=S, p=p, r1=r1, r2=r2)
        self.model = None
        self.i = 0

    def check_things(self, s, a, s_new):
        # print(f"s:{s} a:{a} s_new:{s_new}")
        self.i += 1
        if self.i % 1000 == 0:
            print(f"{self.i:5}: ql.epsilon:{self.model.epsilon}")

    def run(self, n_iter=10000, alpha_decay=0.9999, alpha_min=0.001, epsilon_decay=0.999998):
        self.model = mdp.QLearning(
            self.P,
            self.R,
            self.gamma,
            start_from_begining=True,
            n_iter=n_iter,
            alpha_decay=alpha_decay,
            alpha_min=alpha_min,
            epsilon_decay=epsilon_decay,
            iter_callback=self.check_things,
        )
        self.model.run()


ql = QL_Forest_Runner()

ql.run()
