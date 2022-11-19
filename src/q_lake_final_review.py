import sys

sys.path.insert(0, "src/")

from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from q_learning import chart_lines, get_threshold, q_lake_run
from vi_pi_functions import lake_location

gamma = 0.9
map_name = "Small"
is_slippery = True
n_iter = 3000000
alpha_decay = 0.999999
epsilon_decay = 0.999999
alpha = 0.1
info = q_lake_run(
    map_name, gamma, n_iter, alpha_decay, epsilon_decay, alpha=alpha, is_slippery=is_slippery
)


gamma = 0.9
map_name = "Medium"
is_slippery = True
n_iter = 5000000
alpha_decay = 0.999999
epsilon_decay = 0.999999
alpha = 0.1
info = q_lake_run(
    map_name, gamma, n_iter, alpha_decay, epsilon_decay, alpha=alpha, is_slippery=is_slippery
)


# First - not great
gamma = 0.9
map_name = "Large"
is_slippery = True
n_iter = 20000000
alpha_decay = 0.99999995
epsilon_decay = 0.99999995
alpha = 0.2
info = q_lake_run(
    map_name, gamma, n_iter, alpha_decay, epsilon_decay, alpha=alpha, is_slippery=is_slippery
)

# First - not great
gamma = 0.9
map_name = "Large"
is_slippery = True
n_iter = 20000000
alpha_decay = 0.99999995
epsilon_decay = 0.99999995
alpha = 0.1
info = q_lake_run(
    map_name, gamma, n_iter, alpha_decay, epsilon_decay, alpha=alpha, is_slippery=is_slippery
)

# Better, lower initial alpha
gamma = 0.9
map_name = "Large"
is_slippery = True
n_iter = 20000000
alpha_decay = 0.9999995
epsilon_decay = 0.9999995
alpha = 0.1
info = q_lake_run(
    map_name, gamma, n_iter, alpha_decay, epsilon_decay, alpha=alpha, is_slippery=is_slippery
)


# Optimized set for Large
gamma = 0.9
map_name = "Large"
is_slippery = True
n_iter = 20000000
alpha_decay = 0.9999995
epsilon_decay = 0.9999995
alpha = 0.5
info = q_lake_run(
    map_name, gamma, n_iter, alpha_decay, epsilon_decay, alpha=alpha, is_slippery=is_slippery
)


# Slightly less aggressive early learning for Large
gamma = 0.9
map_name = "Large"
is_slippery = True
n_iter = 20000000
alpha_decay = 0.9999995
epsilon_decay = 0.9999995
alpha = 0.3
info = q_lake_run(
    map_name, gamma, n_iter, alpha_decay, epsilon_decay, alpha=alpha, is_slippery=is_slippery
)
