import numpy as np
class Episode_Memory:
    
    def memory_init(self, episode):
        self.episode = episode
        self.memory = {'episode_start_info':{},'episode_start_info':{},
                       'transitions':{'state':[], 'action':[], 'next_state':[], 'reward':[], 'done':[], 'info':[]}}

        
    def update_memory_reset(self, img_info, episode, info):
        self.memory_init(episode)
        self.memory['episode_start_info'] = img_info
        self.memory['episode_start_info']['info'] = info
    def update_memory_step(self, state, action, next_state, reward, done, info):

        self.memory['transitions']['state'].append(state)
        self.memory['transitions']['action'].append(action)
        self.memory['transitions']['next_state'].append(next_state)
        self.memory['transitions']['reward'].append(reward)
        self.memory['transitions']['done'].append(done)
        self.memory['transitions']['info'].append(info)

        
    def update_memory_done(self, img_info,atom_absolute_nm, atom_relative_nm):
        self.memory['episode_end_info'] = img_info
        self.memory['episode_end_info']['atom_absolute_nm'] = atom_absolute_nm
        self.memory['episode_end_info']['atom_relative_nm'] = atom_relative_nm
    
    def save_memory(self, folder_name):
        np.save('{}/{:0>5d}.npy'.format(folder_name, self.episode), self.memory)
