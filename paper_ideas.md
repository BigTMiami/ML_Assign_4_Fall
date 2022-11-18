todo
check on nd for PI - if 0
change charts for forest
heatmaps with same colorbar

forest 
iteration vs # states pi vs vi
time vs # states pi vs vi

can we check V change to deterimine stable policy in vi


convergence uses max_iter limit calculation - what is this
max_iter governed by gamma, and epsilon.  max_iters is the necessary amount to get to it
how do we know if epsilon threshold for vi is optimal policy?
lower epsilon until policy doesn't change

show rate of change

iteration 

state size vs time

discount, max reward, epsilon / error vs max_iter

forest - essentially one best policy
lake - many maps may have several best policies



policy iteration doesn't converge, but ossiclates
try modified pi to allow convergence using error threshold?
really just check for super low error in PI

# Q learning
    
Modifiy to always start in state 0 to better model real life

Forest is challenging - each step initially the right thing to do is to cut

bar chart comparing time & iterations till convergence for vi,pi, q

try exponential decay for q learning on epsilon

create twiddle to find epsilon \ alpha maximizer - use call back

Lake
The model created stays stuck in the holes and G, instead of starting back at start.
Want to change P for transition back to Start

Created  QLearningEpisodic
    Change S_new selection to use choice
    Create Terminal States - Holes \ Goal 
    Check if Terminal State, no 
    Create episode over flag 

When checking on VI vs PI for medium lake to get ready to check against Q, VI and PI had very different results!!!  Epsilon is an absolute number defaulted to 0.01 and Lake has a max reward of only 1, so it allowed a lot of difference. Confirm policy difference between two is 0.01, adjust epsilon to check.

Q learning convergence \ stopping condition
If environment is really unknown, look for plataue in episidic return.  Could also look at v_max, maybe error?

Q learning is sensitive to
    state size
        Like Policy iteration, but much more(?), by time, iterations, episdoes
    to challenge of state space
        try with no holes, some holes, a lot of holes - medium easy, medium, medium hard
    

Add reward threshold to forest, along with error chart

Todo
    final vi \ pi 
        use new large Lake
    final comparison tables for vi, pi, Q
    Set up doc
    Try Large Hard (More Holes) 

    Need to finish lake vi pi
    check e-stop lake comparison for vi - pi

Show reward curves of VI for different epsilon - see if that is a good way to determine if a good epsilon is chosen. Doesn't work well - curves look the same.  Maybe V Max is not

Zeroed out Q was a mistake - should have created small random values to generate random actions, instead of all left.

