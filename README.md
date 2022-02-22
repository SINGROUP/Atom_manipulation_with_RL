# Atom_manipulation_with_RL

This repository contains codes used in the autonomous atom manipulation project. In this project, we use deep reinforcment learning algorithms including soft actor-critic, hindsight experience replay, and emphsize recent experience replay to automatize atom manipulation in the Createc scanning tunneling microscope system. 

The codes are implemented in python3 and the deep learning algorithms are implemented in pytorch. Each of the jupyter notebook in the repository is used to carry out one of the main task in the project.


## Examples
- single_atom_training.ipynb: collects atom manipulation data and train the deep reinforcement learning agent.
- single_atom_evaluation.ipynb: evaluate the performance of a deep RL agent. 
- baseline_evaluation.ipynb: evaluate the performance of a hard-coded atom manipulation routine.
- multiple_atoms_building.ipynb: used to build multiple-atom structures. The structure building process is divided into individual atom manipulation episodes through assignment and path planning algorithms.
