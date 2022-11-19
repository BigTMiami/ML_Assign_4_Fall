import sys

sys.path.insert(0, "src/")


import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp

from chart_util import save_json_to_file
from maps import maps
from q_learning import chart_lines
from vi_pi_functions import (
    chart_change_vs_iteration,
    chart_reward_vs_error,
    chart_value_vs_iteration,
    compare_policy_iterations,
    compare_two_policies_and_values,
    compare_vi_pi,
    forest_location,
    lake_location,
    lake_plot_policy_and_value,
    percent_fire_review,
    plot_e_stop_values,
    plot_gamma_iterations,
    plot_policy_value_iterations,
)

###############################
# Forest
###############################
gamma = 0.9
r1 = 4
r2 = 2
p = 0.1
S = 7
P, R = example.forest(S=S, p=p, r1=r1, r2=r2)

pi = mdp.PolicyIteration(P, R, gamma)
pi_info = pi.run()


vi = mdp.ValueIteration(P, R, gamma, epsilon=0.001)
vi_info = vi.run()

save_json_to_file(pi_info[-1], "Forest PI.json", forest_location, clean_dict=True)
save_json_to_file(vi_info[-1], "Forest VI.json", forest_location, clean_dict=True)

chart_value_vs_iteration(pi_info, suptitle=f"Forest PI (Gamma:{gamma})")
chart_value_vs_iteration(vi_info, suptitle=f"Forest VI (Gamma:{gamma})")

chart_change_vs_iteration(pi_info, suptitle=f"Forest PI (Gamma:{gamma})")
chart_change_vs_iteration(vi_info, suptitle=f"Forest VI (Gamma:{gamma})")

chart_reward_vs_error(pi_info, "Reward vs Error", f"Forest PI (Gamma:{gamma})", forest_location)
chart_reward_vs_error(vi_info, "Reward vs Error", f"Forest VI (Gamma:{gamma})", forest_location)

percent_fire_review()
percent_fire_review(gamma=0.95)
percent_fire_review(wait_reward=8)

compare_vi_pi(S_max=500, gamma_range=[0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999])


###############################
# LAKE - MEDIUM
###############################
gamma = 0.9
e_stop = 0.0001
map_name = "Medium"
map_used = maps[map_name]
is_slippery = True
P, R = example.openai("FrozenLake-v1", desc=map_used, is_slippery=is_slippery)
title_settings = (
    f"{map_name}, Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Stop:{e_stop}"
)

vi = mdp.ValueIteration(P, R, gamma, epsilon=e_stop)
vi_info = vi.run()

pi = mdp.PolicyIteration(P, R, gamma, max_iter=100)
pi_info = pi.run()

save_json_to_file(pi_info[-1], f"Lake PI {title_settings}.json", lake_location, clean_dict=True)
save_json_to_file(vi_info[-1], f"Lake VI {title_settings}.json", lake_location, clean_dict=True)

lake_plot_policy_and_value(
    vi.policy, vi.V, title_settings, suptitle="Lake VI Final Policy and Value", show_policy=True
)

lake_plot_policy_and_value(
    pi.policy, pi.V, title_settings, suptitle="Lake PI Final Policy and Value", show_policy=True
)

compare_two_policies_and_values(
    pi_info[-1]["Policy"],
    vi_info[-1]["Policy"],
    pi_info[-1]["Value"],
    vi_info[-1]["Value"],
    title_settings,
    "Lake PI and VI Final Differences",
)

chart_reward_vs_error(vi_info, title_settings, "Lake VI Reward and Error", location=lake_location)
chart_reward_vs_error(pi_info, title_settings, "Lake PI Reward and Error", location=lake_location)

plot_gamma_iterations(
    [0.1, 0.5, 0.9, 0.99],
    "Lake VI Gamma Comparison",
    map_name,
    is_slippery,
    "vi",
    show_policy=True,
)

plot_e_stop_values(
    [0.01, 0.001, 0.0001], "Lake VI E Stop Comparison", gamma, map_name, is_slippery, True
)

compare_policy_iterations(vi_info, title_settings, "Lake VI Policy Comparison")
compare_policy_iterations(pi_info, title_settings, "Lake PI Policy Comparison", max_iteration=12)

plot_policy_value_iterations(
    pi_info,
    "Lake PI Policy Iterations",
    title_settings,
    seperate_charts=False,
    iters_to_use=[8, 9, 10, 11],
)
plot_policy_value_iterations(
    vi_info,
    "Lake VI Policy Iterations",
    title_settings,
    seperate_charts=False,
    iters_to_use=[0, 3, 6, 9, 13],
)

###############################
# LAKE - Large
###############################
gamma = 0.9
e_stop = 0.0001
map_name = "Large"
map_used = maps[map_name]
is_slippery = True
P, R = example.openai("FrozenLake-v1", desc=map_used, is_slippery=is_slippery)
title_settings = (
    f"{map_name}, Gamma:{gamma}, {'Is' if is_slippery else 'Not'} Slippery, E Stop:{e_stop}"
)

vi = mdp.ValueIteration(P, R, gamma, epsilon=e_stop)
vi_info = vi.run()

pi = mdp.PolicyIteration(P, R, gamma, max_iter=100)
pi_info = pi.run()

save_json_to_file(pi_info[-1], f"Lake PI {title_settings}.json", lake_location, clean_dict=True)
save_json_to_file(vi_info[-1], f"Lake VI {title_settings}.json", lake_location, clean_dict=True)

lake_plot_policy_and_value(
    vi.policy, vi.V, title_settings, suptitle="Lake VI Final Policy and Value", show_policy=True
)

lake_plot_policy_and_value(
    pi.policy, pi.V, title_settings, suptitle="Lake PI Final Policy and Value", show_policy=True
)

compare_two_policies_and_values(
    pi_info[-1]["Policy"],
    vi_info[-1]["Policy"],
    pi_info[-1]["Value"],
    vi_info[-1]["Value"],
    title_settings,
    "Lake PI and VI Final Differences",
)

chart_reward_vs_error(vi_info, title_settings, "Lake VI Reward and Error", location=lake_location)
chart_reward_vs_error(pi_info, title_settings, "Lake PI Reward and Error", location=lake_location)

plot_gamma_iterations(
    [0.1, 0.5, 0.9, 0.99],
    "Lake VI Gamma Comparison",
    map_name,
    is_slippery,
    "vi",
    show_policy=True,
)

plot_e_stop_values(
    [0.01, 0.001, 0.0001], "Lake VI E Stop Comparison", gamma, map_name, is_slippery, True
)

compare_policy_iterations(vi_info, title_settings, "Lake VI Policy Comparison")
compare_policy_iterations(pi_info, title_settings, "Lake PI Policy Comparison")

plot_policy_value_iterations(
    pi_info,
    "Lake PI Policy Iterations",
    title_settings,
    seperate_charts=False,
    iters_to_use=[1, 5, 9, 16],
)
plot_policy_value_iterations(
    vi_info,
    "Lake VI Policy Iterations",
    title_settings,
    seperate_charts=False,
    iters_to_use=[0, 18, 34, 52],
)
