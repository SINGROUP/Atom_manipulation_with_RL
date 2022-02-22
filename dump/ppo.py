import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

epsilon = 1e-6
LOG_SIG_MAX = 20
LOG_SIG_MIN = -2

class MlpPolicy(nn.Module):
    def __init__(self, action_size=4, input_size=20, n = 64):
        super(MlpPolicy, self).__init__()
        self.action_size = action_size
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, n)
        self.fc2 = nn.Linear(n, n)
        self.fc4 = nn.Linear(self.input_size, n)
        self.fc5 = nn.Linear(n, n)
        self.fc3_mean = nn.Linear(n, self.action_size)
        self.fc3_v = nn.Linear(n, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.log_std = nn.parameter.Parameter(torch.zeros(action_size),requires_grad = True)
        self.apply(weights_init_)
        
    def pi(self, x, a_input=None):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        mean = self.fc3_mean(x)
        log_std = torch.clamp(self.log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        std = log_std.exp()
        dist = torch.distributions.normal.Normal(mean, std)
        a = dist.sample()
        if a_input==None:
            log_prob = dist.log_prob(a)
        else:
            log_prob = dist.log_prob(a_input)
        log_prob = torch.sum(log_prob, dim= -1, keepdim=True)
        entropy = dist.entropy()
        entropy = torch.sum(entropy, dim = -1, keepdim=True)
        return a, log_prob, entropy

    def v(self, x):
        x = self.tanh(self.fc4(x))
        x = self.tanh(self.fc5(x))
        x = self.fc3_v(x)
        return x
    
    def to(self, device):
        return super(MlpPolicy, self).to(device)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class PPO():
    def __init__(self, device, action_size=4, input_size=20, n = 64, gamma = 0.99, learning_rate = 3E-4, lmbda = 0.95, eps_clip = 0.2, v_coef = 0.5, entropy_coef = 0.02, 
                 memory_size = 2048, batch_size = 64):
        self.device = device
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.v_coef = v_coef
        self.entropy_coef = entropy_coef
        self.memory_size = memory_size
        self.batch_size = batch_size
         
        self.policy_network = MlpPolicy(action_size, input_size, n)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate, eps=1e-5)
        self.loss = 0
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': [],
            'advantage': [], 'td_target': torch.FloatTensor([])}
        self.init_episode_memory()
        
    def init_episode_memory(self):
        self.episode_memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': []}

    def add_memory(self, s, a, r, next_s, t, prob):
        self.episode_memory['state'].append(s)
        self.episode_memory['action'].append(a)
        self.episode_memory['reward'].append([r])
        self.episode_memory['next_state'].append(next_s)
        self.episode_memory['terminal'].append([1 - t])
        self.episode_memory['action_prob'].append(prob)
                 
    def finish_path(self):
        for k in self.episode_memory.keys():
            self.memory[k] += self.episode_memory[k]
            
        state = self.episode_memory['state']
        reward = self.episode_memory['reward']
        next_state = self.episode_memory['next_state']
        terminal = self.episode_memory['terminal']
        td_target = torch.FloatTensor(reward) + self.gamma*self.policy_network.v(torch.FloatTensor(next_state))*torch.FloatTensor(terminal)
        delta = td_target - self.policy_network.v(torch.FloatTensor(state))
        delta = delta.detach().numpy()
        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()
        values = self.policy_network.v(torch.FloatTensor(state))
        td_lambda = torch.FloatTensor(advantages)+values
        if self.memory['td_target'].shape == torch.Size([1, 0]):
            self.memory['td_target'] = td_lambda.data #td_target.data
        else:
            self.memory['td_target'] = torch.cat((self.memory['td_target'], td_lambda.data), dim=0)
        self.memory['advantage'] += advantages
        
        if len(self.memory['state']) > self.memory_size:
            for k in self.memory.keys():
                self.memory[k] = self.memory[k][-self.memory_size:]
            
        self.init_episode_memory()
        
    def update(self):
        length = len(self.memory['state'])        
        ind = np.random.permutation(length)
        for i in range(int(length/self.batch_size)):
            self.update_network(ind[i*self.batch_size:(i+1)*self.batch_size])
            

    def update_network(self, index):
        actions = torch.FloatTensor(self.memory['action'])[index].to(self.device)
        state_torch = torch.FloatTensor(self.memory['state'])[index].to(self.device)
        advantage_torch = normalize(torch.FloatTensor(self.memory['advantage']))[index].to(self.device)
        _, new_prob_a, entropy = self.policy_network.pi(state_torch, a_input = actions)
        
        old_prob_a = torch.FloatTensor(self.memory['action_prob'])[index].unsqueeze(1).to(self.device)
        
        ratio = torch.exp(new_prob_a-old_prob_a)
        surr1 = ratio*advantage_torch
        surr2 = torch.clamp(ratio, 1-self.eps_clip,1+self.eps_clip)*advantage_torch
        
        pred_v = self.policy_network.v(state_torch)
        v_loss = F.mse_loss(pred_v, self.memory['td_target'][index].to(self.device))

        self.loss = (-torch.min(surr1,surr2) - self.entropy_coef*entropy).mean()+self.v_coef*v_loss
        self.optimizer.zero_grad()
        self.loss.backward()
        
        '''total_norm = 0
        for p in self.policy_network.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)'''
        
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 0.5, norm_type=2.0)
        self.optimizer.step()
        
def normalize(array):
    return (array-array.mean())/(array.std()+1e-5)