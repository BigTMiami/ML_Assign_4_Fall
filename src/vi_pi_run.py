import sys

sys.path.insert(0, "src/")


import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp

from maps import maps
from q_learning import chart_lines
from vi_pi_functions import compare_two_policies_and_values, lake_plot_policy_and_value

###############################
# LAKE
###############################
gamma = 0.9
e_stop = 0.0001
map_name = "Medium"
map_used = maps[map_name]
is_slippery = True
P, R = example.openai("FrozenLake-v1", desc=map_used, is_slippery=is_slippery)
title_settings = (
    f"({map_name}, Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Stop:{e_stop})"
)
title_settings


vi = mdp.ValueIteration(P, R, gamma, epsilon=e_stop)
vi_info = vi.run()
len(vi_info)

pi = mdp.PolicyIteration(P, R, gamma, max_iter=100)
pi_info = pi.run()
len(pi_info)

lake_plot_policy_and_value(
    vi.policy, vi.V, title_settings, suptitle="Value Iteration Lake", show_policy=True
)

lake_plot_policy_and_value(
    pi.policy, pi.V, title_settings, suptitle="Policy Iteration Lake", show_policy=True
)

compare_two_policies_and_values(
    pi_info[-1]["Policy"],
    vi_info[-1]["Policy"],
    pi_info[-1]["Value"],
    vi_info[-1]["Value"],
    title_settings,
    "PI and VI Differences",
)

from pprint import pprint

pprint(pi_info[-1])
pprint(vi_info[-1])


gamma = 0.99
map_name = "Small"
map_used = maps[map_name]
is_slippery = True
P, R = example.openai("FrozenLake-v1", desc=map_used, is_slippery=is_slippery)
title_settings = f"({map_name}, Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery)"
title_settings

vi = mdp.ValueIteration(P, R, gamma)
vi_info = vi.run()
len(vi_info)

pi = mdp.PolicyIteration(P, R, gamma, max_iter=100)
pi_info = pi.run()
len(pi_info)

lake_plot_policy_and_value(
    vi.policy, vi.V, title_settings, suptitle="Value Iteration Lake", show_policy=True
)

lake_plot_policy_and_value(
    pi.policy, pi.V, title_settings, suptitle="Policy Iteration Lake", show_policy=True
)

compare_two_policies_and_values(
    pi_info[-1]["Policy"],
    vi_info[-1]["Policy"],
    pi_info[-1]["Value"],
    vi_info[-1]["Value"],
    title_settings,
    "PI and VI Differences",
)


compare_policy_iterations(vi_info, "Value Iteration Policy Comparison", map_used)
compare_policy_iterations(
    pi_info, "Policy Iteration Policy Comparison", map_used, max_iteration=12
)

plot_policy_value_iterations(
    pi_info, "Policy Iteration", map_used, seperate_charts=False, iters_to_use=[8, 9, 10, 11]
)
plot_policy_value_iterations(
    vi_info, "Value Iteration", map_used, seperate_charts=False, iters_to_use=[0, 3, 6, 9, 13]
)

chart_reward_vs_error(vi_info, "Lake Reward and Error", "Value Iteration", location=lake_location)
chart_reward_vs_error(pi_info, "Lake Reward and Error", "Policy Iteration", location=lake_location)

plot_gamma_iterations(
    [0.1, 0.5, 0.9, 0.99], "Value Iteration Gamma Comparison", "Large", False, "vi"
)
