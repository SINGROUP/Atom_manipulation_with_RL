import numpy as np
from matplotlib import pyplot as plt
from Environment.a_star import AStarPlanner

class Hexagonal_path_finder:
    def __init__(self, a, f):
        self.a = a
        self.f = f
        self.unit_vectors = np.vstack((self.a,self.f))
        self.ox, self.oy = self.create_o()
        self.motions_hex = np.array([[1, 0, 1],
                            [0, 1, 1],
                            [-1, 0, 1],
                            [0, -1, 1],
                            [-1, 1, 1],
                            [1, -1, 1]])
        self.motions_cart = np.matmul(self.motions_hex[:, :2], self.unit_vectors)
        
    def get_next_step(self, goal_cart, current_cart = None):
        if current_cart is None:
            current_cart = np.array([0,0])
        route_cart, next_step_cart = self.find_path(goal_cart)
        self.plot_next_step(route_cart, next_step_cart, current_cart, goal_cart)
        return next_step_cart
    
    def plot_next_step(self, route_cart, next_step_cart, current_cart, goal_cart):
        plt.rcParams.update({'font.size': 15})
        _, ax = plt.subplots(figsize=(6,6))
        route_cart += current_cart
        motion_cart = self.motions_cart + current_cart
        goal_cart += current_cart
        ax.plot(route_cart[:,0], route_cart[:,1], color = '#A566CC')
        ax.scatter(route_cart[:,0], route_cart[:,1], label = 'planned path', s = 60, color='#A566CC')
        ax.scatter(motion_cart[:,0], motion_cart[:,1], label='nearest neighbors', s = 60, color='#479DC2')
        ax.scatter(current_cart[0], current_cart[1], label='atom position', s = 60, color='#BB893E')
        ax.scatter(goal_cart[0], goal_cart[1], label = 'goal', s = 80, color='#2F848E', edgecolor='black',linewidth=3)
        ax.arrow(current_cart[0], current_cart[1], next_step_cart[0], next_step_cart[1], width=0.05, 
                 head_length = 0.05, length_includes_head = True, alpha=0.5, color='#A566CC')
        s = np.linalg.norm(goal_cart)+0.5
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_xlim([-s+current_cart[0],s+current_cart[0]])
        ax.set_ylim([-s+current_cart[1],s+current_cart[1]])
        ax.legend(frameon=False)
        
        
    def find_path(self, goal_cart):
        goal_hex, resid = self.decompose(goal_cart)
        print('Precision', resid)
        route_hex, droute_hex = self.find_path_hex(goal_hex)
        route_cart = np.matmul(route_hex, self.unit_vectors)
        droute_cart = np.matmul(droute_hex, self.unit_vectors)
        return route_cart, droute_cart[0,:]
        
    def find_path_hex(self, g_hex):
        sx, sy = 0, 0
        [gx, gy] = g_hex
        
        grid_size = 1.0  
        robot_radius = 0  
        a_star = AStarPlanner(self.ox, self.oy, grid_size, robot_radius)
        a_star.motion = self.motions_hex
        rx, ry = a_star.planning(sx, sy, gx, gy)
        r = np.flipud(np.vstack((rx, ry)).T)
        dr= np.diff(r, axis=0)
        return r, dr
    
    def decompose(self, goal_cart):
        goal_hex,_,_,_, = np.linalg.lstsq(self.unit_vectors.T, goal_cart, rcond=None)
        goal_hex = np.rint(goal_hex)
        x_ = np.matmul(goal_hex,self.unit_vectors)
        resid = np.linalg.norm(x_ - goal_cart)
        return goal_hex.astype(int), resid
        
    def create_o(self):
        ox, oy = [], []
        for i in range(-15, 15):
            ox.append(i)
            oy.append(-15.0)
        for i in range(-15, 15):
            ox.append(15.0)
            oy.append(i)
        for i in range(-15, 15):
            ox.append(i)
            oy.append(15.0)
        for i in range(-15, 15):
            ox.append(-15.0)
            oy.append(i)
        return ox, oy 