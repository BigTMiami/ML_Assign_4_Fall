import sys

sys.path.insert(0, "src/")

from q_learning import forest_q, reachable_forest_percentages

###############################
# Forest
###############################

ql = forest_q()
ql = forest_q(epsilon_decay=0.99999)
ql = forest_q(epsilon_decay=0.999999)
ql = forest_q(epsilon_decay=0.999998)
ql = forest_q(epsilon_decay=0.999998, alpha_decay=0.99999)
ql = forest_q(epsilon_decay=0.99999, alpha_decay=0.99999)

reachable_forest_percentages()
