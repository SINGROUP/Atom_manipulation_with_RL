# Atom_manipulation_with_RL

This repository contains codes used in the autonomous atom manipulation project. In this project, we use deep reinforcment learning algorithms including soft actor-critic, hindsight experience replay, and emphsize recent experience replay to automatize atom manipulation in the Createc scanning tunneling microscope system. 

The codes are implemented in python3 and the deep learning algorithms are implemented in pytorch. 


## Usage

### Training reinforcement learning agent

- single_atom_training.ipynb: collects atom manipulation data and train the deep reinforcement learning agent.

### Evaluate a RL or baseline atom manipulation agent
- baseline_evaluation.ipynb: evaluate the performance of a hard-coded atom manipulation routine.

### Build a multiple-atom structure with a trained RL agent
- multiple_atoms_building.ipynb: used to build multiple-atom structures. The structure building process is divided into individual atom manipulation episodes through assignment and path planning algorithms.


## Installation
