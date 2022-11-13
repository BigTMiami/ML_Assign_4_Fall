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
from q_learning import (
    chart_lake_frequencies,
    chart_lines,
    get_terminal_states,
    lake_location,
)

###############################
# LAKE
###############################
gamma = 0.9
map_name = "Medium"
lake_map = maps[map_name]
is_slippery = True
P, R = example.openai("FrozenLake-v1", desc=lake_map, is_slippery=is_slippery)

terminal_states = get_terminal_states(lake_map)

n_iter = 5000000
alpha_decay = 0.999999
epsilon_decay = 0.999999

title_settings = (
    f"(Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay})"
)

ql = mdp.QLearningEpisodic(
    P,
    R,
    gamma,
    terminal_states,
    n_iter=n_iter,
    alpha_decay=alpha_decay,
    epsilon_decay=epsilon_decay,
    episode_stat_frequency=1000,
)
ql_info, episode_stats = ql.run()
len(ql_info)
len(episode_stats)
pprint(ql_info[-1])
pprint(episode_stats[1])
pprint(episode_stats[-1])
pprint(episode_stats[-3:-1])

chart_lines(
    episode_stats,
    ["episode_reward", "Max V"],
    title_settings,
    "Test",
    lake_location,
    review_frequency=100,
    x="Episode",
)

chart_lines(
    episode_stats,
    ["Error"],
    title_settings,
    "Test Error",
    lake_location,
    review_frequency=100,
    x="Episode",
)

chart_lines(
    episode_stats,
    ["Iterations per Episode"],
    title_settings,
    "Test Episode Iterations",
    lake_location,
    review_frequency=100,
    x="Episode",
)


episode = episode_stats[-2]["Episode"]
freq_title_settings = f"Episode: {episode} (Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay})"
chart_lake_frequencies(episode_stats, episode, freq_title_settings)

episode = 2000
freq_title_settings = f"Episode: {episode} (Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay})"
chart_lake_frequencies(episode_stats, episode, freq_title_settings)
