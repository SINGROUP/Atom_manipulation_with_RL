# Atom manipulation with reinforcement learning
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

This repository contains codes used in the autonomous atom manipulation project. In this project, we use deep reinforcment learning algorithms including soft actor-critic, hindsight experience replay, and emphsize recent experience replay to automatize atom manipulation in the Createc scanning tunneling microscope system. 

The codes are implemented in python3 and the deep learning algorithms are implemented in pytorch. 


## Usage

### Training reinforcement learning agent
Run 
`
single_atom_training.ipynb
`.
The notebook goes through the workflow of setting the hyperparameters, collecting atom manipulation data, and training the deep reinforcement learning agent.

### Evaluate a RL or baseline atom manipulation agent
Run `baseline_evaluation.ipynb`. The notebook can be used to evaluate the performance of a hard-coded atom manipulation routine or a trained RL agent on real-world atom manipulation experiments.

### Build a multiple-atom structure with a trained RL agent
Run `multiple_atoms_building.ipynb`. The notebook goes through the process to build multi-atom structures, including defining the design, dividing the building process into individual atom manipulation episodes through assignment and path planning algorithms, and running a trained RL agent.


## Installation
