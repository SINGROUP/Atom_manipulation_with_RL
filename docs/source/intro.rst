Introduction
************

``AMRL`` is a high-level OO Python package which aims to provide a way to train reinforcement learning agents to learn correct parameters for atom manipulation using low-temperature STM.

The aim here was to develop a method which would allow users to easily find STM manipulation parameters for precise atom manipulation in substrate-adsorbate combinations where the parameters for manipulation are unknown.

The current implementation has been developed in Python 3 and tested on Ag and Co adatoms on Ag(111) using Createc STMAFM software version 4.4

More details about the Createc STM remote operation can be found at http://spm-wiki.createc.de/index.php?title=STMAFM_Remote_Operation

Motivation
**********

Atom manipulation with low-temperature STM has provided a technique capable of realizing atomically precise structure for research in condensed matter, but the parameters for manipulating atoms on surfaces may be unknown a priori and may depend largely on tip condition.

Limitations
***********

- Currently the Python package is only built to work for the Createc STMAFM software. Future work will implement wrapper classes for the AMRL.Environment.createc_control class in order to expand functionality to other STM electronics controllers.
