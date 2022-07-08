import numpy as np
class Episode_Memory:
    """
    Used to store episode information
    """
    def memory_init(self, episode: int) -> None:
        """Create a memory dict

        Parameters
        ----------

        episode : int
            episode index

        Returns
        -------

        None

        """
        self.episode = episode
        transitions_dict = {'state':[], 'action':[],
                            'next_state':[], 'reward':[],
                            'done':[], 'info':[]}
        self.memory = {'episode_start_info':{},'episode_start_info':{},
                       'transitions': transitions_dict}


    def update_memory_reset(self,
                            img_info: dict,
                            episode: int,
                            info: dict) -> None:
        """Create memory dict and store info related to reset step of environment

        Parameters
        ----------

        img_info : dict
            image and related information

        episode : int
            episode index

        info: dict

        Returns
        -------

        None : None
        """
        self.memory_init(episode)
        self.memory['episode_start_info'] = img_info
        self.memory['episode_start_info']['info'] = info

    def update_memory_step(self,
                           state: np.array,
                           action: np.array,
                           next_state: np.array,
                           reward: float,
                           done: bool,
                           info: dict) -> None:
        """Store information of a transition

        Parameters
        ----------

        state, action, next_state : array_like

        reward : float

        done : bool

        info : dict

        Returns
        -------

        None : None

        """
        self.memory['transitions']['state'].append(state)
        self.memory['transitions']['action'].append(action)
        self.memory['transitions']['next_state'].append(next_state)
        self.memory['transitions']['reward'].append(reward)
        self.memory['transitions']['done'].append(done)
        self.memory['transitions']['info'].append(info)

    def update_memory_done(self,
                           img_info: dict,
                           atom_absolute_nm: np.array,
                           atom_relative_nm: np.array) -> None:
        """
        Store information when an episode terminates

        Parameters
        ----------

        img_info : dict
            image and related information

        atom_absolute_nm, atom_relative_nm : array_like
            atom position (rel. to template position) in STM coordinates (nm)

        Returns
        -------

        None : None

        """
        self.memory['episode_end_info'] = img_info
        self.memory['episode_end_info']['atom_absolute_nm'] = atom_absolute_nm
        self.memory['episode_end_info']['atom_relative_nm'] = atom_relative_nm

    def save_memory(self, folder_name: str) -> None:
        """Store the memory dict as .npy file

        Parameters
        ----------

        folder_name : str

        Returns
        -------

        None : None
        """
        np.save('{}/{:0>5d}.npy'.format(folder_name, self.episode), self.memory)
