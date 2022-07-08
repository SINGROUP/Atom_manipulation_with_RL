import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.optim import Adam
from collections import deque, namedtuple
import copy

class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        """
        Parameters
        ----------
        capacity: int
            max length of buffer deque

        Returns
        -------
        None
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self,
             state: list,
             action: list,
             reward: list,
             next_state: list,
             mask: list):
        """
        Insert a new memory into the end of the ReplayMemory buffer

        Parameters
        ----------
        state, action, reward, next_state, mask: array_like
        Returns
        -------
        None
        """
        self.buffer.insert(0, (state, action, reward, next_state, mask))

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
    def __init__(self,
                 capacity: int,
                 env,
                 strategy: str='final'):
        """
        Initialize HerReplayMemory object

        Parameters
        ----------
        capacity: int
        env: AMRL.RealExpEnv
        strategy: str

        Returns
        -------
        None
        """
        super(HerReplayMemory, self).__init__(capacity)
        self.env = env
        self.n_sampled_goal = 2
        self.strategy = strategy

    def sample(self,
               batch_size: int,
               c_k: float) -> tuple:
        """
        Sample batch_size (state, action, reward, next_state, mask) # of memories
        from the HERReplayMemory, emphasizing the c_k most recent experiences
        to account for potential tip changes.

        Also implemented: hindsight experience replay, which treats
        memories in which the achieved goal was different than the intended goal
        as 'succesful' in order to speed up training.


        Parameters
        ----------
        batch_size: int
        c_k: int
            select from the c_k most recent memories

        Returns
        -------
        tuple
        """
        N = len(self.buffer)
        if c_k>N:
            c_k = N
        indices = np.random.choice(c_k, int(batch_size))
        batch = []
        for idx in indices:
            batch.append(self.buffer[idx])
            state, action, reward, next_state, mask = self.buffer[idx]
            #print('old state:', state, 'old next state:', next_state, 'old reward:', reward)

            final_idx = self.sample_goals(idx)
            for fi in final_idx:
                _, _, _, final_next_state, _ = self.buffer[fi]
                new_next_state = copy.copy(next_state)
                new_state = copy.copy(state)
                new_state[:2] = final_next_state[2:]
                new_next_state[:2] = final_next_state[2:]
                new_reward = self.env.compute_reward(new_state, new_next_state)
                m = (new_state, action, new_reward, new_next_state, mask)
                batch.append(m)
        print('No. of samples:', len(batch))
        state, action, reward, next_state, mask = map(np.stack,zip(*batch))
        return state, action, reward, next_state, mask

    def sample_goals(self, idx):
        """
        Sample memories in the same episode

        Parameters
        ----------
        idx: int

        Returns
        -------
        array_like
            list of final_idx HerReplayMemory buffer indices
        """
        #get done state idx
        i = copy.copy(idx)

        while True:
            _,_,_,_,m = self.buffer[i]
            if not m:
                break
            else:
                i-=1
        if self.strategy == 'final' or i == idx:
            return [i]
        elif self.strategy == 'future':
            iss = np.random.choice(np.arange(i, idx+1), min(idx-i+1, 3))
            return iss

    def calculate_value(self, state):
        """
        Deprecated
        """
        goal_nm = state[:2]*self.env.goal_nm
        atom_nm = state[2:]*self.env.goal_nm
        dist_destination = np.linalg.norm(atom_nm - goal_nm)
        a = atom_nm
        b = goal_nm
        cos_similarity_destination = np.inner(a,b)/(self.env.goal_nm*np.clip(np.linalg.norm(a), a_min=self.env.goal_nm, a_max=None))
        value = self.env.calculate_value(dist_destination, cos_similarity_destination)
        return value


def soft_update(target,source,tau):
    """


    Parameters
    ----------

    Returns
    -------
    """
    for target_param, param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(target_param.data*(1.0-tau)+param.data*tau)

def hard_update(target,source):
    """


    Parameters
    ----------

    Returns
    -------
    """
    for target_param, param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(param.data)

def weights_init_(m):
    """
    Initialize weights in torch.nn object

    Parameters
    ----------
    m: torch.nn.Linear

    Returns
    -------
    None
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        """
        Initialize the Q network.

        Parameters
        ----------
        num_inputs: int
        num_actions: int
        hidden_dim: int

        Returns
        -------
        None
        """
        super(QNetwork,self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.fc4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        self.apply(weights_init_)

    def forward(self, state, action):
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
    def __init__(self,
                 num_inputs: int,
                 num_actions: int,
                 hidden_dim: int,
                 action_space: namedtuple=None):
        """

        Parameters
        ----------
        num_inputs: int
        num_actions: int
        hidden_dim: int
        action_space: namedtuple

        Returns
        -------
        """
        super(GaussianPolicy,self).__init__()

        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)

        if action_space is None:
            self.action_scale = torch.tensor([1, 1, 1, 1, 1/3, 0.25])
            self.action_bias = torch.tensor([0, 0, 0, 0, 2/3, 0.75])
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
        self.apply(weights_init_)

    def forward(self, state: np.array):
        """

        Parameters
        ----------
        state: array_like

        Returns
        -------
        mean, log_std
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        """
        Parameters
        ----------
        state: array_like

        Returns
        -------
        action, log_prob, mean
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean,std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t*self.action_scale + self.action_bias
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
    def __init__(self,
                 num_inputs: int,
                 num_actions: int,
                 action_space: namedtuple,
                 device: torch.device,
                 hidden_size: int,
                 lr: float,
                 gamma: float,
                 tau: float,
                 alpha: float) -> None:
        """
        Initialize soft-actor critic agent for performing RL task and training

        Parameters
        ----------
        num_inputs: int
            number of input values to agent
        num_actions: int
            number of values output by agent
        action_space: namedtuple
            namedtuple called ACTION_SPACE with fields called 'high' and 'low'
            that are each torch tensors of length len(num_actions)
            which define the GaussianPolicy action_scale and action_bias
        device: torch.device
        hidden_size: int

        lr: float

        gamma: float

        tau: float

        alpha: float

        Returns
        -------
        None
        """

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device

        args = num_inputs, num_actions, hidden_size
        self.critic = QNetwork(*args).to(self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)
        self.critic_target = QNetwork(*args).to(self.device)
        hard_update(self.critic_target,self.critic)

        args = num_inputs, num_actions, hidden_size, action_space
        self.policy = GaussianPolicy(*args).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        kwargs = {'requires_grad':True,
                  'device':self.device,
                  'dtype':torch.float32}
        self.log_alpha = torch.tensor([np.log(alpha)], **kwargs)
        self.alpha = self.log_alpha.detach().exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr)
        arg = torch.Tensor([num_actions]).to(self.device)
        self.target_entropy = -torch.prod(arg).item()

    def select_action(self,
                      state: np.array,
                      eval:bool=False):
        """

        Parameters
        ----------
        state: array_like
            should be of length num_inputs

        Returns
        -------
        action: array_like
            should be of length num_actions

        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if eval == False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self,
                          memory: HerReplayMemory,
                          batch_size: int,
                          c_k: float,
                          train_pi: bool = True):
        """
        SAC agent training step

        Parameters
        ----------
        memory: HerReplayMemory or ReplayMemory object
        batch_size: int
            minibatch size, i.e. number of memories to sample per batch
        c_k: float
        train_pi: bool

        Returns
        -------
        """
        memories = memory.sample(batch_size, c_k)
        states, actions, rewards, next_states, masks = memories
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        masks = torch.FloatTensor(masks).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions, next_state_log_pi, _ = self.policy.sample(next_states)
            q1_next_target, q2_next_target = self.critic_target(next_states, next_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target)-self.alpha*next_state_log_pi
            next_q_value = rewards + masks*self.gamma*min_q_next_target

        q1, q2 = self.critic(states,actions)

        q_loss = F.mse_loss(q1, next_q_value) + F.mse_loss(q2, next_q_value)
        self.critic_optim.zero_grad()
        q_loss.backward()
        critic_norm = self.get_grad_norm(self.critic)
        '''
        total_norm = 0
        for p in self.critic.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        print(total_norm)'''

        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10, norm_type=2.0)
        self.critic_optim.step()

        if train_pi:
            pi, log_pi,_ = self.policy.sample(states)
            q1_pi, q2_pi = self.critic(states, pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            policy_loss = (self.alpha*log_pi-min_q_pi).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            policy_norm = self.get_grad_norm(self.policy)
            print('Training','critic norm:', critic_norm, 'policy norm:', policy_norm)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 2, norm_type=2.0)
            self.policy_optim.step()

            alpha_loss = -(self.log_alpha*(log_pi+self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        soft_update(self.critic_target,self.critic,self.tau)

    def get_grad_norm(self, net):
        """


        Parameters
        ----------
        net:

        Returns
        -------
        total_norm:
        """
        total_norm = 0
        for p in net.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)

        return total_norm
