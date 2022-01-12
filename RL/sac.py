import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from collections import deque
import copy

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen = capacity)
    def push(self, state, action, reward, next_state, mask):
        self.buffer.insert(0,(state, action, reward, next_state, mask))
    def sample(self, batch_size, c_k):
        N = len(self.buffer)
        if c_k>N:
            c_k = N
        indices = np.random.choice(c_k, batch_size)
        batch = [self.buffer[idx] for idx in indices]
        #batch = random.sample(self.buffer,batch_size)
        state, action, reward, next_state, mask = map(np.stack,zip(*batch))
        return state, action, reward, next_state, mask
    def __len__(self):
        return len(self.buffer)

class HerReplayMemory(ReplayMemory):
    def __init__(self, capacity, env):
        super(HerReplayMemory, self).__init__(capacity)
        self.env = env
        self.n_sampled_goal = 2
    def sample(self, batch_size, c_k):
        N = len(self.buffer)
        if c_k>N:
            c_k = N
        indices = np.random.choice(c_k, int(batch_size/self.n_sampled_goal))
        batch = []
        for idx in indices:
            batch.append(self.buffer[idx])
            state, action, reward, next_state, mask = self.buffer[idx]
            '''print('old state:', state, 'old next state:', next_state, 'old reward:', reward)
            i = copy.copy(idx)   
            while True:
                _,_,_,n,m = self.buffer[i]
                i-=1
                if not m:
                    new_goal = n[:2]
                    break'''
            #state[:2] = new_goal
            #next_state[:2] = new_goal
            #old_value = self.calculate_value(state)
            #new_value = self.calculate_value(next_state)
            #reward = self.env.default_reward + new_value - old_value
            #achieved_goal = next_state[2:]
            if isinstance(next_state, dict):
                final_idx = self.sample_goals(idx)
                _, _, _, final_next_state, _ = self.buffer[final_idx]
                new_next_state = copy.copy(next_state)
                new_state = copy.copy(state)
                new_next_state['desired_goal'] = final_next_state['achieved_goal']
                new_state['desired_goal'] = final_next_state['achieved_goal']
                achieved_goal = new_next_state['achieved_goal']
                desired_goal = new_next_state['desired_goal']
                new_reward = self.env.compute_reward(achieved_goal, desired_goal, None)
            batch.append((new_state, action, new_reward, new_next_state, mask))
        state, action, reward, next_state, mask = map(np.stack,zip(*batch))
        return state, action, reward, next_state, mask

    def sample_goals(self, idx):
        #get done state idx
        i = copy.copy(idx)   
        while True:
            _,_,_,_,m = self.buffer[i]
            if not m:
                return i
            i-=1

    def calculate_value(self, state):
        goal_nm = state[:2]*self.env.goal_nm
        atom_nm = state[2:]*self.env.goal_nm
        dist_destination = np.linalg.norm(atom_nm - goal_nm)
        a = atom_nm
        b = goal_nm
        cos_similarity_destination = np.inner(a,b)/(self.env.goal_nm*np.clip(np.linalg.norm(a), a_min=self.env.goal_nm, a_max=None))
        value = self.env.calculate_value(dist_destination, cos_similarity_destination)
        return value


    
def soft_update(target,source,tau):
    for target_param, param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)
        
def hard_update(target,source):
    for target_param, param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(param.data)
        
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self,num_inputs,num_actions,hidden_dim):
        super(QNetwork,self).__init__()
        self.fc1 = nn.Linear(num_inputs+num_actions,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)
        
        self.fc4 = nn.Linear(num_inputs+num_actions,hidden_dim)
        self.fc5 = nn.Linear(hidden_dim,hidden_dim)
        self.fc6 = nn.Linear(hidden_dim,1)
        self.apply(weights_init_)
        
    def forward(self,state,action):
        sa = torch.cat([state,action],1)
        x1 = F.relu(self.fc1(sa))
        x1 = F.relu(self.fc2(x1))
        x1 = self.fc3(x1)
        
        x2 = F.relu(self.fc4(sa))
        x2 = F.relu(self.fc5(x2))
        x2 = self.fc6(x2)
        return x1, x2
    
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6
class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy,self).__init__()
        
        self.fc1 = nn.Linear(num_inputs,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.mean = nn.Linear(hidden_dim,num_actions)
        self.log_std = nn.Linear(hidden_dim,num_actions)
        
        if action_space is None:
            self.action_scale = torch.tensor([1,1,1,1,1/3,0.25])
            self.action_bias = torch.tensor([0,0,0,0,2/3,0.75])
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        self.apply(weights_init_)
    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std
    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean,std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t*self.action_scale+self.action_bias                                                                             
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)
    
class sac_agent():
    def __init__(self, num_inputs, num_actions, action_space, device, hidden_size,lr,gamma, tau, alpha):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device
        
        self.critic = QNetwork(num_inputs, num_actions, hidden_size).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(),lr=lr)
        self.critic_target = QNetwork(num_inputs, num_actions, hidden_size).to(self.device)
        hard_update(self.critic_target,self.critic)
        
        self.policy = GaussianPolicy(num_inputs, num_actions, hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(),lr=lr)
        
        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True, device=self.device, dtype=torch.float32)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        self.target_entropy = -torch.prod(torch.Tensor([num_actions]).to(self.device)).item()        
    
    def select_action(self, state, eval=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, memory, batch_size, c_k, train_pi = True):
        states, actions, rewards, next_states, masks = memory.sample(batch_size,c_k)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_actions, next_state_log_pi, _ = self.policy.sample(next_states)
            q1_next_target, q2_next_target = self.critic_target(next_states,next_actions)
            min_q_next_target = torch.min(q1_next_target,q2_next_target)-self.alpha*next_state_log_pi
            next_q_value = rewards+masks*self.gamma*min_q_next_target
            
        q1, q2 = self.critic(states,actions)
        
        q_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        self.critic_optim.zero_grad()
        q_loss.backward()
        
        '''
        total_norm = 0
        for p in self.critic.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print(total_norm)'''
        
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5, norm_type=2.0)
        self.critic_optim.step()
        
        if train_pi:
            pi, log_pi,_ = self.policy.sample(states)
            q1_pi, q2_pi = self.critic(states, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            policy_loss = (self.alpha*log_pi-min_q_pi).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1, norm_type=2.0)
            self.policy_optim.step()
            
            alpha_loss = -(self.log_alpha*(log_pi+self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()
        
        soft_update(self.critic_target,self.critic,self.tau)

    def feature_extractor(state):
        if isinstance(state, np.ndarray):
            states = []
            for s in state:
                state_flat = np.concatenate((s['observation'], s['desired_goal']),-1)
                states.append(state_flat)
            return np.vstack(states)
        else:
            return np.concatenate((state['observation'], state['desired_goal']),-1)

