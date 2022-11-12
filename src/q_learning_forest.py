import sys

sys.path.insert(0, "src/")

from q_learning import forest_q, reachable_forest_percentages

###############################
# Forest
###############################

forest_q()
forest_q(epsilon_decay=0.99999)
forest_q(epsilon_decay=0.999999)
forest_q(epsilon_decay=0.999998)
forest_q(epsilon_decay=0.999998, alpha_decay=0.99999)
forest_q(epsilon_decay=0.99999, alpha_decay=0.99999)

reachable_forest_percentages()
