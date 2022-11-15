import sys

sys.path.insert(0, "src/")

from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from q_learning import chart_lines, get_threshold, q_lake_run
from vi_pi_functions import lake_location

gamma = 0.9
map_name = "Large"
is_slippery = True
n_iter = 20000000
alpha_decay = 0.99999995
epsilon_decay = 0.99999995
alpha = 0.4
info = q_lake_run(
    map_name, gamma, n_iter, alpha_decay, epsilon_decay, alpha=alpha, is_slippery=is_slippery
)
