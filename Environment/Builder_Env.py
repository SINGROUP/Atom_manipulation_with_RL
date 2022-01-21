from Environment.Env_new import RealExpEnv
from Environment.get_atom_coordinate import get_atom_coordinate_nm, get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor
from scipy.spatial.distance import cdist as cdist

import numpy as np
import scipy.spatial as spatial
from scipy.optimize import linear_sum_assignment
import copy
from Environment.rrt import RRT
import importlib
import Environment.Env_new
importlib.reload(Environment.Env_new)
from Environment.Env_new import RealExpEnv
import Environment.get_atom_coordinate
importlib.reload(Environment.get_atom_coordinate)
from Environment.get_atom_coordinate import get_atom_coordinate_nm, get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor
from Environment.data_visualization import plot_atoms_and_design


def circle(x, y, r, p = 100):
    x_, y_ = [], []
    for i in range(p):
        x_.append(x+r*np.cos(2*i*np.pi/p))
        y_.append(y+r*np.sin(2*i*np.pi/p))
    return x_, y_ 

def assignment(start, goal):
    cost_matrix = cdist(np.array(start)[:,:2], np.array(goal)[:,:2])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind]
    total_cost = np.sum(cost)
    return np.array(start)[row_ind,:], np.array(goal)[col_ind,:], cost, total_cost, row_ind, col_ind

def align_design(atoms, design):
    assert atoms.shape == design.shape
    c_min = np.inf
    for i in range(atoms.shape[0]):
        for j in range(design.shape[0]):
            a = atoms[i,:]
            d = design[j,:]
            design_ = design+a-d
            a_index = np.delete(np.arange(atoms.shape[0]), i)
            d_index = np.delete(np.arange(design.shape[0]), j)
            a, d, _, c, _, _ = assignment(atoms[a_index,:], design_[d_index,:])
            if (c<c_min):
                c_min = c
                atoms_assigned, design_assigned = a, d
                anchor = atoms[i,:]
    return atoms_assigned, design_assigned, c_min, anchor

def get_atom_and_anchor(all_atom_absolute_nm, anchor_nm):
    new_anchor_nm, anchor_nm, _, _, row_ind, col_ind = assignment(all_atom_absolute_nm, anchor_nm)
    atoms_nm = np.delete(all_atom_absolute_nm, row_ind, axis=0)
    return atoms_nm, new_anchor_nm

def get_anchor(atom, anchors):
    if anchors.shape[0]==1:
        anchor= anchors[0,:]
    else:
        anchor, _, _, _, _, _ = assignment(anchors, atom.reshape((-1,2)))
    return anchor.flatten()

def align_deisgn_stitching(all_atom_absolute_nm, design_nm, align_design_params):
    anchor_atom_nm = align_design_params['atom_nm']
    anchor_design_nm = align_design_params['design_nm']
    
    assert anchor_design_nm.tolist() in design_nm.tolist()
    dist = cdist(all_atom_absolute_nm, anchor_atom_nm.reshape((-1,2)))
    atoms = np.delete(all_atom_absolute_nm, np.argmin(dist), axis=0)
    dist = cdist(design_nm, anchor_design_nm.reshape((-1,2)))
    designs = np.delete(design_nm, np.argmin(dist), axis=0)
    designs += (anchor_atom_nm - anchor_design_nm)
    return atoms, designs, anchor_atom_nm

class Structure_Builder(RealExpEnv):
    def __init__(self, step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, current_jump, im_size_nm, offset_nm,
                 pixel, scan_mV, max_len, safe_radius_nm = 1, speed = None, precision_lim = None):
        super(Structure_Builder, self).__init__(step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, None, current_jump,
                                                im_size_nm, offset_nm, None, pixel, None, scan_mV, max_len, None, random_scan_rate = 0)
        self.atom_absolute_nm_f = None
        self.atom_absolute_nm_b = None
        #self.large_DX_DDeltaX = float(self.createc_controller.stm.getparam('DX/DDeltaX'))
        self.large_offset_nm = offset_nm
        self.large_len_nm = im_size_nm
        self.safe_radius_nm = safe_radius_nm
        self.anchor_nm = None
        if speed is None:
            self.speed = self.createc_controller.get_speed()
        else:
            self.speed = speed
        if precision_lim is not None:
            self.precision_lim = precision_lim
        print('speed:', self.speed)
        
    def reset_large(self, design_nm, align_design_mode = 'auto', align_design_params = {'atom_nm':None, 'design_nm':None}):
        self.align_design_mode = align_design_mode
        self.num_atoms = design_nm.shape[0]
        self.all_atom_absolute_nm = self.scan_all_atoms(self.large_offset_nm, self.large_len_nm) 
        if self.align_design_mode == 'auto':
            self.atoms, self.designs, c_min, anchor = align_design(self.all_atom_absolute_nm, design_nm)
        elif self.align_design_mode =='manual':
            self.atoms, self.designs, anchor = align_deisgn_stitching(self.all_atom_absolute_nm, design_nm, align_design_params)
        self.init_anchor = anchor

        plot_atoms_and_design(self.large_img_info, self.atoms,self.designs, self.init_anchor)
        self.design_nm = np.concatenate((self.designs, anchor.reshape((-1,2))))
        self.large_img_info |= {'design': self.design_nm}
        
        self.anchors = [self.init_anchor]

        for i in range(self.atoms.shape[0]):
            self.atom_chosen, self.design_chosen, self.obstacle_list = self.match_atoms_designs(i)
            self.anchor_chosen = self.init_anchor
            self.next_destinatio_nm, self.paths = self.find_path()
            if (self.next_destinatio_nm is not None) and (self.paths is not None):
                break
        offset_nm, len_nm = self.get_offset_len()
        print('Use anchor:', self.use_anchor)
        if np.linalg.norm(self.atom_chosen - self.design_chosen) > 1.5*self.goal_nm:
            self.stop_lim = np.sqrt(3)*self.precision_lim
        else:
            self.stop_lim = self.precision_lim
        return self.atom_chosen, self.design_chosen, self.next_destinatio_nm, self.paths, self.anchor_chosen, offset_nm, len_nm
        
    def step_large(self, succeed, new_atom_position):
        self.all_atom_absolute_nm = self.scan_all_atoms(self.large_offset_nm, self.large_len_nm)
        self.large_img_info |= {'design': self.design_nm}
        self.atoms, new_anchor = get_atom_and_anchor(self.all_atom_absolute_nm, np.vstack(self.anchors))
        self.anchors = list(new_anchor)
        done = False
        if succeed and (np.linalg.norm(self.next_destinatio_nm - self.design_chosen)<0.01):
            done = self.update_after_success(new_atom_position)
        for i in range(self.atoms.shape[0]):
            self.atom_chosen, self.design_chosen, self.obstacle_list = self.match_atoms_designs(i)
            self.anchor_chosen = get_anchor(self.atom_chosen, np.vstack(self.anchors))
            self.next_destinatio_nm, self.paths = self.find_path()
            if (self.next_destinatio_nm is not None) and (self.paths is not None):
                break

        offset_nm, len_nm = self.get_offset_len()
        if np.linalg.norm(self.atom_chosen - self.design_chosen) > 1.5*self.goal_nm:
            self.stop_lim = np.sqrt(3)*self.precision_lim
        else:
            self.stop_lim = self.precision_lim
        return self.atom_chosen, self.design_chosen, self.next_destinatio_nm, self.paths, self.anchor_chosen, offset_nm, len_nm, done

    def get_offset_len(self):
        len_nm_0 = 2*max(np.max(np.abs(self.anchor_chosen - self.atom_chosen)), 2)+1
        len_nm_1 = 2*max(np.max(np.abs(self.next_destinatio_nm - self.atom_chosen)), 2)+2
        len_nm = max(len_nm_0, len_nm_1)
        if len_nm > self.large_len_nm:
            len_nm = len_nm_1
            self.use_anchor = False
        else:
            self.use_anchor = True
        offset_nm = self.atom_chosen +np.array([0,-0.5*len_nm])
        return offset_nm, len_nm

    def update_after_success(self, new_atom_position):
        i = np.argmin(cdist(self.all_atom_absolute_nm, new_atom_position.reshape((-1,2))))
        new_atom_position = self.all_atom_absolute_nm[i,:]
        print('update after success')
        print('atoms before:', self.atoms)
        print((self.atoms == new_atom_position).all(axis=1).nonzero())
        self.atoms = np.delete(self.atoms, (self.atoms == new_atom_position).all(axis=1).nonzero(), axis=0)
        print('atoms after:', self.atoms)
        print('designs before:', self.designs)
        self.designs = np.delete(self.designs, (self.designs == self.design_chosen).all(axis=1).nonzero(), axis=0)
        print('designs after:', self.designs)
        self.anchors.append(new_atom_position)
        return (self.atoms.shape[0] == 0) and (self.designs.shape == 0)

    def match_atoms_designs(self, i = 0):
        atoms, designs, costs, _, _, _ = assignment(self.atoms, self.designs)
        j = np.flip(np.argsort(costs))[i]

        atom_chosen = atoms[j,:]
        design_chosen = designs[j,:]
        obstacle_list = []
        for i in range(atoms.shape[0]):
            if i!=j:
                obstacle_list.append((atoms[i,0],atoms[i,1],self.safe_radius_nm))
        for a in self.anchors:
            obstacle_list.append((a[0], a[1], self.safe_radius_nm))
        return atom_chosen, design_chosen, obstacle_list
     
    def find_path(self, max_step = 2):
        print('start:',self.atom_chosen, 'goal',self.design_chosen)
        rrt = RRT(
            start=self.atom_chosen, goal=self.design_chosen, rand_area=[-2, 15],
            obstacle_list=self.obstacle_list, expand_dis= max_step, path_resolution=1)
        path_len = np.inf
        min_path = None
        for _ in range(20):
            path = rrt.planning(animation=False)
            if path is not None:
                if len(path)<path_len:
                    min_path = path
                    path_len = len(path)
            else:
                break

        if min_path is None:
            print('Cannot find path')
            return None, None
        return np.array(min_path[-2]), min_path   

    def reset(self, destination_nm, anchor_nm, offset_nm, len_nm, large_len_nm):
        self.len = 0
        self.scan_atom(anchor_nm, offset_nm, len_nm, large_len_nm)

        self.atom_start_absolute_nm = self.atom_absolute_nm
        print('anchor from small scan:', self.anchor_nm, 'anchor from large scan:', anchor_nm)
        if (self.anchor_nm is not None) and self.use_anchor:
            destination_nm_with_correction = destination_nm + self.anchor_nm - anchor_nm
        else:
            destination_nm_with_correction = destination_nm

        self.destination_absolute_nm, self.goal = self.get_destination(self.atom_start_absolute_nm, destination_nm_with_correction)
        info = {'start_absolute_nm':self.atom_start_absolute_nm, 'goal_absolute_nm':self.destination_absolute_nm,
                'start_absolute_nm_f':self.atom_absolute_nm_f, 'start_absolute_nm_b':self.atom_absolute_nm_b, 'img_info':self.img_info}
        self.dist_destination = np.linalg.norm(self.atom_absolute_nm - self.destination_absolute_nm)
        return np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm)), info
    
    def step(self, action):
        x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent = self.action_to_latman_input(action)
        current_series, d = self.step_latman(x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent)
        info = {'current_series':current_series, 'd': d, 'start_nm':  np.array([x_start_nm , y_start_nm]), 'end_nm':np.array([x_end_nm , y_end_nm])}
        done = False
        self.len+=1
        done = self.len == self.max_len
        if not done:
            jump = self.detect_current_jump(current_series)

        if done or jump:
            self.dist_destination, dist_start, dist_last = self.check_similarity()
            print('atom moves by:', dist_start)
            done = done or (dist_start > 1.5*self.goal_nm) or (self.dist_destination < self.stop_lim)
            self.atom_move_detector.push(current_series, dist_last)

        next_state = np.concatenate((self.goal, (self.atom_absolute_nm -self.atom_start_absolute_nm)/self.goal_nm))
        info |= {'atom_absolute_nm':self.atom_absolute_nm, 'atom_absolute_nm_f' : self.atom_absolute_nm_f, 'atom_absolute_nm_b' : self.atom_absolute_nm_b, 'img_info' : self.img_info}
        
        return next_state, None, done, info

    def scan_atom(self, anchor_nm = None, offset_nm = None, len_nm = None, large_len_nm = None):
        if offset_nm is not None:
            self.offset_nm = offset_nm 
        if len_nm is not None:
            self.len_nm = len_nm 
        if anchor_nm is not None:
            self.anchor_nm = anchor_nm 
        '''if large_len_nm is not None:
            small_DX_DDeltaX = int(self.large_DX_DDeltaX*len_nm/large_len_nm)
            self.createc_controller.stm.setparam('DX/DDeltaX', small_DX_DDeltaX)'''

        self.createc_controller.offset_nm = self.offset_nm
        self.createc_controller.im_size_nm = self.len_nm
        
        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image(speed= self.speed)
        if self.use_anchor:
            self.atom_absolute_nm_f, self.anchor_nm_f = get_atom_coordinate_nm_with_anchor(img_forward, offset_nm, len_nm, self.anchor_nm)
            self.atom_absolute_nm_b, self.anchor_nm_b = get_atom_coordinate_nm_with_anchor(img_backward, offset_nm, len_nm, self.anchor_nm)
            self.anchor_nm = 0.5*(self.anchor_nm_f+self.anchor_nm_b)
        else:
            self.atom_absolute_nm_f, _ = get_atom_coordinate_nm_with_anchor(img_forward, offset_nm, len_nm, None)
            self.atom_absolute_nm_b, _ = get_atom_coordinate_nm_with_anchor(img_backward, offset_nm, len_nm, None)

        self.atom_absolute_nm = 0.5*(self.atom_absolute_nm_f+self.atom_absolute_nm_b)
        self.img_info = {'img_forward':img_forward,'img_backward':img_backward, 'offset_nm':offset_nm, 'len_nm':len_nm,
                         'anchor': self.anchor_nm, 'atom_absolute_nm_f': self.atom_absolute_nm_f,
                         'atom_absolute_nm_b': self.atom_absolute_nm_b, 'atom_absolute_nm': self.atom_absolute_nm}
        return self.atom_absolute_nm, None


    def scan_all_atoms(self, offset_nm, len_nm):
        #self.createc_controller.stm.setparam('DX/DDeltaX', self.large_DX_DDeltaX)
        self.createc_controller.offset_nm = offset_nm
        self.createc_controller.im_size_nm = len_nm
        self.offset_nm = offset_nm
        self.len_nm = len_nm
        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image(speed= self.speed)
        all_atom_absolute_nm_f = get_all_atom_coordinate_nm(img_forward, offset_nm, len_nm)
        all_atom_absolute_nm_b = get_all_atom_coordinate_nm(img_backward, offset_nm, len_nm)

        all_atom_absolute_nm_f = np.array(sorted(all_atom_absolute_nm_f, key = lambda x: (x[0], x[1])))
        all_atom_absolute_nm_b = np.array(sorted(all_atom_absolute_nm_b, key = lambda x: (x[0], x[1])))

        self.all_atom_absolute_nm_f = all_atom_absolute_nm_f
        self.all_atom_absolute_nm_b = all_atom_absolute_nm_b

        if len(all_atom_absolute_nm_b)!=len(all_atom_absolute_nm_f):
            print('length of list of atoms found in b and f different')

        all_atom_absolute_nm = 0.5*(all_atom_absolute_nm_f+all_atom_absolute_nm_b)
        self.all_atom_absolute_nm = all_atom_absolute_nm
        self.large_img_info = {'img_forward':img_forward,'img_backward':img_backward, 'offset_nm':offset_nm, 'len_nm':len_nm,
                                'all_atom_absolute_nm':all_atom_absolute_nm, 'all_atom_absolute_nm_f':all_atom_absolute_nm_f,
                                'all_atom_absolute_nm_b':all_atom_absolute_nm_b}
        return all_atom_absolute_nm

    def get_destination(self, atom_start_absolute_nm, destination_absolute_nm):
        angle = np.arctan2((destination_absolute_nm-atom_start_absolute_nm)[1],(destination_absolute_nm-atom_start_absolute_nm)[0])
        goal_nm = min(self.goal_nm, np.linalg.norm(destination_absolute_nm-atom_start_absolute_nm))
        dr = goal_nm*np.array([np.cos(angle),np.sin(angle)])
        destination_absolute_nm = atom_start_absolute_nm + dr
        return destination_absolute_nm, dr/self.goal_nm

    def step_latman(self, x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent):
        x_start_nm+=self.atom_absolute_nm[0]
        x_end_nm+=self.atom_absolute_nm[0]
        y_start_nm+=self.atom_absolute_nm[1]
        y_end_nm+=self.atom_absolute_nm[1]
        if [x_start_nm, y_start_nm] != [x_end_nm, y_end_nm]:
            data = self.createc_controller.lat_manipulation(x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent, self.offset_nm, self.len_nm)
            if data is not None:
                current = np.array(data.current).flatten()
                x = np.array(data.x)
                y = np.array(data.y)
                d = np.sqrt(((x-x[0])**2 + (y-y[0])**2))
            else:
                current = None
                d = None
            return current, d
        else:
            return None, None
