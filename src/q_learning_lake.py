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

from chart_util import save_json_to_file, save_to_file
from maps import maps
from q_learning import (
    chart_lake_frequencies,
    chart_lines,
    get_terminal_states,
    lake_location,
)
from vi_pi_functions import lake_plot_policy_and_value

###############################
# LAKE
###############################
gamma = 0.9
map_name = "Small"
lake_map = maps[map_name]
is_slippery = True
P, R = example.openai("FrozenLake-v1", desc=lake_map, is_slippery=is_slippery)

terminal_states = get_terminal_states(lake_map)

n_iter = 2500000
alpha_decay = 0.999999
epsilon_decay = 0.999999

title_settings = f"(Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay} Map:{map_name})"

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

info = episode_stats
threshold = 0.01
threshold_column = "percent_chg"
x = "episode_reward"
y = "Episode"
value_window = 30
pct_window = 10

df = pd.melt(pd.DataFrame(info), y)
dfp = df[df["variable"] == x].copy()
dfp["moving_avg"] = dfp["value"].rolling(value_window).mean()
dfp["percent_chg"] = dfp["moving_avg"].pct_change().rolling(pct_window).mean()
min_column_value = dfp[threshold_column].dropna().min()
if min_column_value < threshold:
    threshold_episode = dfp[dfp[threshold_column] < threshold].iloc[0][y]
threshold_episode
dfp[dfp[threshold_column] < threshold].iloc[0]

found_index = None
for i, value in enumerate(episode_stats):
    if value["Episode"] == threshold_episode:
        found_index = i
        break
found_index

pprint(episode_stats[found_index])

final_stats = episode_stats[-2]

for item, value in final_stats.items():
    if isinstance(value, np.ndarray):
        final_stats[item] = value.tolist()
final_stats["alpha_decay"] = alpha_decay
final_stats["epsilon_decay"] = epsilon_decay
final_stats["map_name"] = map_name
final_stats["is_slippery"] = is_slippery
save_json_to_file(final_stats, "Q Lake Test", lake_location)


chart_lines(
    episode_stats,
    ["episode_reward", "Max V"],
    title_settings,
    "Lake Q Reward and Max V",
    lake_location,
    review_frequency=100,
    x="Episode",
)

chart_lines(
    episode_stats,
    ["Error"],
    title_settings,
    "Lake Q Error",
    lake_location,
    review_frequency=100,
    x="Episode",
)

chart_lines(
    episode_stats,
    ["Iterations per Episode"],
    title_settings,
    "Lake Q Iterations per Episode",
    lake_location,
    review_frequency=100,
    x="Episode",
)


suptitle = f"Lake State Visits - {map_name} Map"
episode = episode_stats[-2]["Episode"]
freq_title_settings = f"Episode: {episode} (Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay})"
chart_lake_frequencies(episode_stats, episode, freq_title_settings, suptitle=suptitle)

episode = 2000
freq_title_settings = f"Episode: {episode} (Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Decay:{epsilon_decay})"
chart_lake_frequencies(episode_stats, episode, freq_title_settings, suptitle=suptitle)
