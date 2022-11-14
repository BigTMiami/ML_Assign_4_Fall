import sys

sys.path.insert(0, "src/")

from q_learning import q_lake_run

gamma = 0.9
map_name = "Small"
is_slippery = True
n_iter = 5000000
alpha_decay = 0.999999
epsilon_decay = 0.999999
info = q_lake_run(map_name, gamma, n_iter, alpha_decay, epsilon_decay, is_slippery=is_slippery)

gamma = 0.9
map_name = "Medium"
is_slippery = True
n_iter = 5000000
alpha_decay = 0.999999
epsilon_decay = 0.999999
info = q_lake_run(map_name, gamma, n_iter, alpha_decay, epsilon_decay, is_slippery=is_slippery)

gamma = 0.9
map_name = "Large"
is_slippery = True
n_iter = 10000000
alpha_decay = 0.999999
epsilon_decay = 0.999999
info = q_lake_run(map_name, gamma, n_iter, alpha_decay, epsilon_decay, is_slippery=is_slippery)
