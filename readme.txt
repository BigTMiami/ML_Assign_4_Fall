Anthony Menninger
amenninger3
CS 7641 Machine Learning
Assignment 4

Infrastructure Setup
* Uses Python 3.8 (May work with 3.6 or 3.7) 
# CODE
    * Core code located at https://github.com/BigTMiami/ML_Assign_4_Fall.git.  To install:
        * git clone https://github.com/BigTMiami/ML_Assign_4_Fall.git
    * Forked MDP Toolbox at https://github.com/BigTMiami/hiivemdptoolbox.git.  This will be installed using the requirements.txt
# Python Setup
    * In ML_Assign_4_Fall directory is a requirements.txt.  This will load all the needed python libraries, including the forked MDP Toolbox.
    * Best practice is to use a virtual environment, such as virtualenv, to load libraries.
    * From the command line in the ML_Assign_4_Fall directory, install with command: 
        pip install -r requirements.txt
    * Commands shown use the python and pip commmands, but depending on your system you may use python3 or pip3
# Running the code
    * To generate all results in the paper, first setup the results directory structure:  
        * python src/setup_results.py
    * To generate the Value Iteration and Policy Iteration Results:
        * python src/vi_pi_final_run.py 
    * To generate the Q Learning Forest Results:
        * python src/q_forest_final_review.py      
    * To generate the Q Learning Lake Results:
        * python src/q_lake_final_review.py    
    * To generate some Final Comparison Charts:
        * python src/final_results.py