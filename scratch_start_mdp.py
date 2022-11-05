from pprint import pprint

import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp
import pandas as pd

S = 5
p = 0.3
gamma = 0.9
P, R = example.forest(S=S, p=p)
vi = mdp.ValueIteration(P, R, gamma)
# vi.setVerbose()
info = vi.run()
print(vi.V)
pprint(info)
df = pd.DataFrame.from_dict(info)
df.columns
vi.policy

for i in range(1, 10):
    P, R = example.forest(S=S, p=i * 0.1)
    vi = mdp.ValueIteration(P, R, gamma)
    info = vi.run()
    print(vi.V)
    print(vi.policy)
