import sys

sys.path.insert(0, "src/")

from q_learning import forest_q, reachable_forest_percentages

###############################
# Forest
###############################


ql = forest_q(epsilon_decay=0.999998)
