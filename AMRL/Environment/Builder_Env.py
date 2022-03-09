from .Env_new import RealExpEnv
from .get_atom_coordinate import get_all_atom_coordinate_nm, get_atom_coordinate_nm_with_anchor
from .rrt import RRT
from .data_visualization import plot_atoms_and_design

from scipy.spatial.distance import cdist as cdist
import numpy as np
from scipy.optimize import linear_sum_assignment

def assignment(start, goal):
    """
    Assign start to goal with the linear_sum_assignment function and setting the cost matrix to the distance between each start-goal pair

    Parameters
    ----------
    start, goal: array_like
        start and goal positions

    Returns
    -------
    np.array(start)[row_ind,:], np.array(goal)[col_ind,:], cost: array_like
            sorted start and goal positions, and their distances

    total_cost: float
            total distances
    
    row_ind, col_ind: array_like
            Indexes of the start and goal array in sorted order
    """
    cost_matrix = cdist(np.array(start)[:,:2], np.array(goal)[:,:2])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cost = cost_matrix[row_ind, col_ind]
    total_cost = np.sum(cost)
    return np.array(start)[row_ind,:], np.array(goal)[col_ind,:], cost, total_cost, row_ind, col_ind

def align_design(atoms, design):
    """
    Move design positions and assign atoms to designs to minimize total manipulation distance 

    Parameters
    ----------
    atoms, design: array_like
        atom and design positions

    Returns
    -------
    atoms_assigned, design_assigned: array_like
            sorted atom and design (moved) positions
    
    anchor: array_like
            position of the atom that will be used as the anchor
    """
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
    return atoms_assigned, design_assigned, anchor

def align_deisgn_stitching(all_atom_absolute_nm, design_nm, align_design_params):
    """
    Shift the designs to match the atoms based on align_design_params. 
    Assign atoms to designs to minimize total manipulation distance.
    Get the obstacle list from align_design_params

    Parameters
    ----------
    all_atom_absolute_nm, design_nm: array_like
        atom and design positions

    align_design_params: dict
        {'atom_nm', 'design_nm', 'obstacle_nm'} 

    Returns
    -------
    atoms, designs: array_like
            sorted atom and design (moved) positions
    
    anchor_atom_nm: array_like
            position of the atom that will be used as the anchor
    """
    anchor_atom_nm = align_design_params['atom_nm']
    anchor_design_nm = align_design_params['design_nm']
    obstacle_nm = align_design_params['obstacle_nm']
    assert anchor_design_nm.tolist() in design_nm.tolist()
    dist = cdist(all_atom_absolute_nm, anchor_atom_nm.reshape((-1,2)))
    anchor_atom_nm = all_atom_absolute_nm[np.argmin(dist),:]
    atoms = np.delete(all_atom_absolute_nm, np.argmin(dist), axis=0)
    dist = cdist(design_nm, anchor_design_nm.reshape((-1,2)))
    designs = np.delete(design_nm, np.argmin(dist), axis=0)
    designs += (anchor_atom_nm - anchor_design_nm)
    if obstacle_nm is not None:
        obstacle_nm[:,:2] = obstacle_nm[:,:2]+(anchor_atom_nm - anchor_design_nm)
    return atoms, designs, anchor_atom_nm, obstacle_nm

def get_atom_and_anchor(all_atom_absolute_nm, anchor_nm):
    """
    Separate the positions of the anchor and the rest of the atoms 

    Parameters
    ----------
    all_atom_absolute_nm, anchor_nm: array_like
        positions of all the atoms and the anchor

    Returns
    -------
    atoms_nm, new_anchor_nm: array_like
            positions of all the atoms (except the anchor) and the anchor
    """
    new_anchor_nm, anchor_nm, _, _, row_ind, _ = assignment(all_atom_absolute_nm, anchor_nm)
    atoms_nm = np.delete(all_atom_absolute_nm, row_ind, axis=0)
    return atoms_nm, new_anchor_nm

class Structure_Builder(RealExpEnv):
    def __init__(self, step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, current_jump, im_size_nm, offset_nm,
                 pixel, scan_mV, max_len, safe_radius_nm = 1, speed = None, precision_lim = None):
        super(Structure_Builder, self).__init__(step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, None, current_jump,
                                                im_size_nm, offset_nm, None, pixel, None, scan_mV, max_len, None, random_scan_rate = 0)
        self.atom_absolute_nm_f = None
        self.atom_absolute_nm_b = None
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

    def reset_large(self, design_nm,
                    align_design_mode = 'auto', align_design_params = {'atom_nm':None, 'design_nm':None}, sequence_mode = 'design',
                    left = None, right = None, top = None, bottom = None):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
        self.sequence_mode = sequence_mode
        self.align_design_mode = align_design_mode
        self.num_atoms = design_nm.shape[0]
        self.all_atom_absolute_nm = self.scan_all_atoms(self.large_offset_nm, self.large_len_nm)
        if self.align_design_mode == 'auto':
            self.atoms, self.designs, anchor = align_design(self.all_atom_absolute_nm, design_nm)
            self.outside_obstacles = None
        elif self.align_design_mode =='manual':
            self.atoms, self.designs, anchor, obstacle_nm = align_deisgn_stitching(self.all_atom_absolute_nm, design_nm, align_design_params)
            if obstacle_nm is not None:
                self.outside_obstacles = list(obstacle_nm)
            else:
                self.outside_obstacles = None
        self.init_anchor = anchor
        plot_atoms_and_design(self.large_img_info, self.atoms,self.designs, self.init_anchor)
        self.design_nm = np.concatenate((self.designs, anchor.reshape((-1,2))))
        self.large_img_info |= {'design': self.design_nm}
        self.anchors = [self.init_anchor]
        offset_nm, len_nm = self.get_the_returns()
        return self.atom_chosen, self.design_chosen, self.next_destinatio_nm, self.paths, self.anchor_chosen, offset_nm, len_nm

    def step_large(self, succeed, new_atom_position):
        """
        Take a large STM scan and update the atoms and designs after a RL episode  

        Parameters
        ----------
        succeed: bool
                if the RL episode was successful
        
        new_atom_position: array_like
                the new position of the manipulated atom

        Returns
        -------
        self.atom_chosen, self.design_chosen, self.next_destinatio_nm, self.anchor_chosen: array_like
                the positions of the atom, design, target, and anchor to be used in the RL episode 
        
        self.paths: array_like
                the planned path between atom and design
        
        offset_nm: array_like
                offset value to use for the STM scan

        len_nm: float
                image size for the STM scan 
        
        done:bool 
        """
        self.all_atom_absolute_nm = self.scan_all_atoms(self.large_offset_nm, self.large_len_nm)
        self.large_img_info |= {'design': self.design_nm}
        self.atoms, new_anchor = get_atom_and_anchor(self.all_atom_absolute_nm, np.vstack(self.anchors))
        self.anchors = list(new_anchor)
        done = False
        if succeed and (np.linalg.norm(self.next_destinatio_nm - self.design_chosen)<0.01) and self.use_anchor:
            self.update_after_success(new_atom_position)
        done = (self.atoms.shape[0] == 0) and (self.designs.shape == 0)
        if not done:
            offset_nm, len_nm = self.get_the_returns()
        else:
            offset_nm, len_nm = None, None
        return self.atom_chosen, self.design_chosen, self.next_destinatio_nm, self.paths, self.anchor_chosen, offset_nm, len_nm, done

    def get_the_returns(self):
        """
        Get and set the parameters needed for defining a RL episode:
        offset_nm, len_nm,
        self.atom_chosen, self.design_chosen, self.obstacle_list,
        self.anchor_chosen,
        self.next_destinatio_nm, self.paths

        Parameters
        ----------
        None:None

        Returns
        -------
        offset_nm: array_like
                offset value to use for the STM scan

        len_nm: float
                image size for the STM scan
        """

        for i in range(self.atoms.shape[0]):
            self.atom_chosen, self.design_chosen, self.obstacle_list = self.match_atoms_designs(i, mode = self.sequence_mode)
            if self.outside_obstacles is not None:
                self.obstacle_list+=self.outside_obstacles
            self.anchor_chosen = self.init_anchor
            self.next_destinatio_nm, self.paths = self.find_path()
            if (self.next_destinatio_nm is not None) and (self.paths is not None) and (self.next_destinatio_nm[1]<self.large_offset_nm[1]+self.large_len_nm):
                break
        offset_nm, len_nm = self.get_offset_len()
        return offset_nm, len_nm

    def get_offset_len(self):
        """
        Get the offset and size for the STM scan around the atom to be manipulated

        Parameters
        ----------
        None:None

        Returns
        -------
        offset_nm: array_like
                offset value to use for the STM scan

        len_nm: float
                image size for the STM scan
        """
        left = np.min([self.anchor_chosen[0], self.atom_chosen[0],self.next_destinatio_nm[0]])-1.5
        right = np.max([self.anchor_chosen[0], self.atom_chosen[0],self.next_destinatio_nm[0]])+1.5
        top = np.min([self.anchor_chosen[1], self.atom_chosen[1],self.next_destinatio_nm[1]])-1.5
        bottom = np.max([self.anchor_chosen[1], self.atom_chosen[1],self.next_destinatio_nm[1]])+1.5

        left = max(left,self.large_offset_nm[0]-0.5*self.large_len_nm)
        right = min(right,self.large_offset_nm[0]+0.5*self.large_len_nm)
        top = max(top, self.large_offset_nm[1])
        bottom = min(bottom, self.large_offset_nm[1]+self.large_len_nm)
        len_nm = max(right - left, bottom - top)
        offset_nm = np.array([0.5*(left+right), 0.5*(top+bottom)])+np.array([0,-0.5*len_nm])
        self.use_anchor = True

        if offset_nm[0]+0.5*len_nm>self.large_offset_nm[0]+0.5*self.large_len_nm:
            offset_nm[0] = self.large_offset_nm[0]+0.5*self.large_len_nm - 0.5*len_nm
        if offset_nm[0]-0.5*len_nm<self.large_offset_nm[0]-0.5*self.large_len_nm:
            offset_nm[0] = self.large_offset_nm[0]-0.5*self.large_len_nm+0.5*len_nm
        if offset_nm[1]<self.large_offset_nm[1]:
            offset_nm[1] = self.large_offset_nm[1]
        if offset_nm[1]+len_nm>self.large_offset_nm[1]+self.large_len_nm:
            offset_nm[1] = self.large_offset_nm[1]+self.large_len_nm - len_nm
        return offset_nm, len_nm

    def update_after_success(self, new_atom_position):
        """
        Remove positions from self.atoms and self.designs array when the atom is moved to the correct position.

        Parameters
        ----------
        new_atom_position: array_like
            atom position

        Returns
        -------
        None: None
        """
        i = np.argmin(cdist(self.all_atom_absolute_nm, new_atom_position.reshape((-1,2))))
        new_atom_position = self.all_atom_absolute_nm[i,:]
        self.atoms = np.delete(self.atoms, (self.atoms == new_atom_position).all(axis=1).nonzero(), axis=0)
        self.designs = np.delete(self.designs, (self.designs == self.design_chosen).all(axis=1).nonzero(), axis=0)
        self.anchors.append(new_atom_position)

    def match_atoms_designs(self, i = 0, mode = 'design'):
        """
        Assign atoms to designs and choose the next atom to manipulate.
        If mode = 'design', choose the atom-design pair with the ith largest distance.
        If mode = 'anchor', choose the atom-design pair with ith shortest atom-anchor distance

        Parameters
        ----------
        i: int
            used to choose atom-design pair
        mode: str

        Returns
        -------
        atom_chosen, design_chosen: array_like
                the chosen atom and design positions
        obstacle_list: list
                list of obstacle positions. It will be used in the pathplanning algorithm
        """
        atoms, designs, costs, _, _, _ = assignment(self.atoms, self.designs)
        if mode=='design':
            j = np.flip(np.argsort(costs))[i]
        elif mode=='anchor':
            j = (np.argsort(cdist(atoms, np.vstack(self.anchors)).min(axis = 1)))[i]
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
        """
        Pathplanning between self.atom_chosen and self.design_chosen. Use RRT if the distance is larger than self.safe_radius_nm, and straight line otherwise.

        Parameters
        ----------
        max_step: float
                the maximum length (nm) for each step

        Returns
        -------
        next_target, : array_like
                next target position
        min_path: list
                list of steps in the planned path
        """
        print('start:',np.around(self.atom_chosen, decimals=2), 'goal',np.around(self.design_chosen,decimals=2))
        if np.linalg.norm(self.atom_chosen - self.design_chosen)< self.safe_radius_nm:
            print('direct step, RRT not used')
            return self.design_chosen, [self.design_chosen, self.atom_chosen]
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
        next_target = np.array(min_path[-2])
        return next_target, min_path

    def reset(self, destination_nm, anchor_nm, offset_nm, len_nm):
        """
        Reset the environment

        Parameters
        ----------
        destination_nm: array_like
                the final target position
        anchor_nm: array_like
                the anchor position in nm
        offset_nm: array_like
                offset value to use for the STM scan

        len_nm: float
                image size for the STM scan

        Returns
        -------
        self.state: array_like
        done: bool
        info: dict
        """
        self.len = 0
        self.scan_atom(anchor_nm, offset_nm, len_nm)

        self.atom_start_absolute_nm = self.atom_absolute_nm
        if self.use_anchor:
            print('anchor from small scan:', np.around(self.anchor_nm, decimals=3), 'large scan:', np.around(anchor_nm, decimals=3))
        else: print('not using anchor')
        if (self.anchor_nm is not None) and self.use_anchor:
            destination_nm_with_correction = destination_nm + self.anchor_nm - anchor_nm
        else:
            destination_nm_with_correction = destination_nm

        self.destination_absolute_nm, self.goal = self.get_destination(self.atom_start_absolute_nm, destination_nm_with_correction)
        if np.linalg.norm(self.goal)>=1:
            self.stop_lim = np.sqrt(3)*self.precision_lim
        else:
            self.stop_lim = self.precision_lim

        info = {'start_absolute_nm':self.atom_start_absolute_nm, 'goal_absolute_nm':self.destination_absolute_nm,
                'start_absolute_nm_f':self.atom_absolute_nm_f, 'start_absolute_nm_b':self.atom_absolute_nm_b, 'img_info':self.img_info}
        self.dist_destination = np.linalg.norm(self.atom_absolute_nm - self.destination_absolute_nm)
        self.state = np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm))
        done = self.dist_destination<self.stop_lim
        return self.state, done, info

    def step(self, action):
        """
        Take a step in the environment with the given action

        Parameters
        ----------
        action: array_like

        Return
        ------
        next_state: np.array
        done: bool
        info: dict
        """
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

    def scan_atom(self, anchor_nm = None, offset_nm = None, len_nm = None):
        """
        Take a STM scan and extract the atom position of the atom in the current manipulation episode

        Parameters
        ----------
        anchor_nm: array_like
                position of the anchor atom in nm
        offset_nm: array_like
                offset value to use for the STM scan
        len_nm: float
                image size for the STM scan

        Return
        ------
        self.atom_absolute_nm: array_like
                atom position in STM coordinates (nm)
        self.atom_relative_nm: array_like
                atom position relative to the template position in STM coordinates (nm)
        """
        if offset_nm is not None:
            self.offset_nm = offset_nm
        if len_nm is not None:
            self.len_nm = len_nm
        if anchor_nm is not None:
            self.anchor_nm = anchor_nm

        self.createc_controller.offset_nm = self.offset_nm
        self.createc_controller.im_size_nm = self.len_nm

        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image(speed= self.speed)
        if self.use_anchor:
            self.atom_absolute_nm_f, self.anchor_nm_f = get_atom_coordinate_nm_with_anchor(img_forward, offset_nm, len_nm, self.anchor_nm, self.obstacle_list)
            self.atom_absolute_nm_b, self.anchor_nm_b = get_atom_coordinate_nm_with_anchor(img_backward, offset_nm, len_nm, self.anchor_nm, self.obstacle_list)
            self.anchor_nm = 0.5*(self.anchor_nm_f+self.anchor_nm_b)
        else:
            self.atom_absolute_nm_f, _ = get_atom_coordinate_nm_with_anchor(img_forward, offset_nm, len_nm, None, self.obstacle_list)
            self.atom_absolute_nm_b, _ = get_atom_coordinate_nm_with_anchor(img_backward, offset_nm, len_nm, None, self.obstacle_list)

        self.atom_absolute_nm = 0.5*(self.atom_absolute_nm_f+self.atom_absolute_nm_b)
        self.img_info = {'img_forward':img_forward,'img_backward':img_backward, 'offset_nm':offset_nm, 'len_nm':len_nm,
                         'anchor': self.anchor_nm, 'atom_absolute_nm_f': self.atom_absolute_nm_f,
                         'atom_absolute_nm_b': self.atom_absolute_nm_b, 'atom_absolute_nm': self.atom_absolute_nm}
        return self.atom_absolute_nm, None


    def scan_all_atoms(self, offset_nm, len_nm):
        """
        Take a STM scan and extract the atom position of all the atoms in the current building task

        Parameters
        ----------
        offset_nm: array_like
                offset value to use for the STM scan
        len_nm: float
                image size for the STM scan

        Return
        ------
        self.atom_absolute_nm: array_like
                atom position in STM coordinates (nm)
        self.atom_relative_nm: array_like
                atom position relative to the template position in STM coordinates (nm)
        """
        self.createc_controller.offset_nm = offset_nm
        self.createc_controller.im_size_nm = len_nm
        self.offset_nm = offset_nm
        self.len_nm = len_nm
        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image(speed= self.speed)
        all_atom_absolute_nm_f = get_all_atom_coordinate_nm(img_forward, offset_nm, len_nm, self.left, self.right, self.top, self.bottom)
        all_atom_absolute_nm_b = get_all_atom_coordinate_nm(img_backward, offset_nm, len_nm, self.left, self.right, self.top, self.bottom)

        all_atom_absolute_nm_f = np.array(sorted(all_atom_absolute_nm_f, key = lambda x: (x[0], x[1])))
        all_atom_absolute_nm_b = np.array(sorted(all_atom_absolute_nm_b, key = lambda x: (x[0], x[1])))


        if all_atom_absolute_nm_b.shape[0]>all_atom_absolute_nm_f.shape[0]:
            all_atom_absolute_nm_b = all_atom_absolute_nm_f[:all_atom_absolute_nm_f.shape[0],:]

        if all_atom_absolute_nm_f.shape[0]>all_atom_absolute_nm_b.shape[0]:
            diff = all_atom_absolute_nm_f.shape[0] - all_atom_absolute_nm_b.shape[0]
            all_atom_absolute_nm_f = all_atom_absolute_nm_f[diff:,:]

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
        """
        Uniformly sample a new target that is within the self.inner_limit_nm

        Parameters
        ----------
        atom_absolute_nm: array_like
                atom position in STM coordinates (nm)

        destination_absolute_nm: array_like
                final target position in nm

        Return
        ------
        destination_absolute_nm: array_like
                target position in STM coordinates (nm)
        dr/self.goal_nm: array_like
                target position relative to the initial atom position in STM coordinates (nm)
        """
        angle = np.arctan2((destination_absolute_nm-atom_start_absolute_nm)[1],(destination_absolute_nm-atom_start_absolute_nm)[0])
        goal_nm = min(self.goal_nm, np.linalg.norm(destination_absolute_nm-atom_start_absolute_nm))
        dr = goal_nm*np.array([np.cos(angle),np.sin(angle)])
        destination_absolute_nm = atom_start_absolute_nm + dr
        return destination_absolute_nm, dr/self.goal_nm

    def step_latman(self, x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent):
        """
        Execute the action in Createc

        Parameters
        ----------
        x_start_nm, y_start_nm, x_end_nm, y_end_nm: float
                start and end position of the tip movements in nm
        mvolt: float
                bias in mV
        pcurrent: float
                current setpoint in pA

        Return
        ------
        current: array_like
                manipulation current trace
        d: float
                tip movement distance
        """
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
