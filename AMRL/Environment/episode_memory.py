import numpy as np
class Episode_Memory:
    """
    Used to store episode information
    """
    def memory_init(self, episode):
        """
        Create a memory dict
        Arguments:
            episode (int): episode index
        """
        self.episode = episode
        self.memory = {'episode_start_info':{},'episode_start_info':{},
                       'transitions':{'state':[], 'action':[], 'next_state':[], 'reward':[], 'done':[], 'info':[]}}

        
    def update_memory_reset(self, img_info, episode, info):
        """
        Create a memory dict and store information related to the reset step of the environment
        Arguments:
            img_info (dict): image and related information
            episode (int): episode index
            info (dict)
        """
        self.memory_init(episode)
        self.memory['episode_start_info'] = img_info
        self.memory['episode_start_info']['info'] = info

    def update_memory_step(self, state, action, next_state, reward, done, info):
        """
        Store information of a transition
        Arguments:
            state, action, next_state (np.array)
            reward (float)
            done (bool)
            info (dict)
        """
        self.memory['transitions']['state'].append(state)
        self.memory['transitions']['action'].append(action)
        self.memory['transitions']['next_state'].append(next_state)
        self.memory['transitions']['reward'].append(reward)
        self.memory['transitions']['done'].append(done)
        self.memory['transitions']['info'].append(info)

    def update_memory_done(self, img_info, atom_absolute_nm, atom_relative_nm):
        """
        Store information when an episode terminates
        Arguments:
            img_info (dict): image and related information
            atom_absolute_nm, atom_relative_nm (np.array): the atom position (relative to the template position) in STM coordinates (nm)
        """
        self.memory['episode_end_info'] = img_info
        self.memory['episode_end_info']['atom_absolute_nm'] = atom_absolute_nm
        self.memory['episode_end_info']['atom_relative_nm'] = atom_relative_nm
    
    def save_memory(self, folder_name):
        """
        Store the memory dict as npy file
        Arguments:
            folder_name (str)
        """
        np.save('{}/{:0>5d}.npy'.format(folder_name, self.episode), self.memory)
