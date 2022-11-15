import sys

sys.path.insert(0, "src/")

from pprint import pprint

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from q_learning import chart_forest_frequencies, chart_lines, forest_q, get_threshold
from vi_pi_functions import forest_location

###############################
# Forest
###############################

info = forest_q(epsilon_decay=0.99999, alpha_decay=0.99999, repeat_count=2)
# info = forest_q(epsilon_decay=0.99999, repeat_count=2)

title = "TESTING"
chart_forest_frequencies(info, title)


threshold = 0.0004
threshold_column = "percent_chg"
x = "running_reward"
y = "Iteration"
value_window = 1000
pct_window = 1000


reward_threshold_iteration = get_threshold(
    info, threshold, threshold_column, x, y, value_window=value_window, pct_window=pct_window
)
reward_threshold_iteration

for index, value in enumerate(info):
    if value["Iteration"] == reward_threshold_iteration:
        break
save_index = index
index
pprint(info[0])
pprint(info[save_index + 100])
len(info) / 2
save_stats = info[save_index]

for item, value in save_stats.items():
    if isinstance(value, np.ndarray):
        save_stats[item] = value.tolist()
save_stats
if reward_threshold_iteration is not None:
    save_type = "Threshold"
    save_index = None
    for index, value in enumerate(info):
        if value["Iteration"] == reward_threshold_iteration:
            break
    save_index = index
else:
    save_type = "Final"
    save_index = -2

save_stats = ql_all[0][save_index]
for item, value in save_stats.items():
    if isinstance(value, np.ndarray):
        save_stats[item] = value.tolist()


df = pd.melt(pd.DataFrame(info), y)
# dfp = df[df["variable"] == x].copy()
dfp = df[df["variable"] == x].copy().groupby(y)["value"].mean().to_frame().reset_index()
dfp["moving_avg"] = dfp["value"].rolling(value_window).mean()
dfp["percent_chg"] = dfp["moving_avg"].pct_change().rolling(pct_window).mean()
min_column_value = dfp[threshold_column].dropna().min()
min_column_value = 0
min_column_value < threshold
threshold_episode = dfp[dfp[threshold_column] < threshold].iloc[0][y]
threshold_episode

dfp.plot(x="Iteration", y=["value", "moving_avg"])
plt.show()
dfp.plot(x="Iteration", y=["percent_chg"])
plt.show()

pd.set_option("display.max_rows", 500)
dfp.head(500)

dfp.shape
dfp
gamma = 1
epsilon_decay = 1
alpha_decay = 1
title = f"TESTING CHART THRESHOLD"

chart_lines(
    info,
    ["Epsilon", "Alpha"],
    title,
    "Q Forest Epsilon, Alpha",
    forest_location,
    review_frequency=10000,
    threshold=reward_threshold,
)

chart_lines(
    info,
    ["running_reward"],
    title,
    "Q Forest Running Reward",
    forest_location,
    review_frequency=10000,
    threshold=reward_threshold,
)


df = pd.melt(pd.DataFrame(info), y)
dfp = df[df["variable"] == x].copy().groupby(y)["value"].mean().to_frame()
dfp[dfp["Iteration"] == 100]
dfp
dfd = dfp.groupby(y)["value"].mean()
dfd
dfp["moving_avg"] = dfp["value"].rolling(value_window).mean()
dfp["percent_chg"] = dfp["moving_avg"].pct_change().rolling(pct_window).mean()
min_column_value = dfp[threshold_column].dropna().min()
