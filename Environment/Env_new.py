from Environment.createc_control import Createc_Controller
import numpy as np
from matplotlib import pyplot as plt
#import get_atom_coordinate
import importlib
#importlib.reload(get_atom_coordinate)
from Environment.get_atom_coordinate import get_atom_coordinate_nm
import scipy.spatial as spatial
import findiff
import pdb
import scipy
from Environment.atom_jump_detection import AtomJumpDetector_conv
import Environment.atom_jump_detection
importlib.reload(Environment.atom_jump_detection)
from Environment.atom_jump_detection import AtomJumpDetector_conv
import Environment.get_atom_coordinate
importlib.reload(Environment.get_atom_coordinate)
from Environment.get_atom_coordinate import get_atom_coordinate_nm

class RealExpEnv:
    
    def __init__(self, step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, template, current_jump, im_size_nm, offset_nm,
                 manip_limit_nm, pixel, template_max_y, scan_mV, max_len, load_weight, random_scan_rate = 0.5,  correct_drift = False, bottom = True):
        
        self.step_nm = step_nm
        self.max_mvolt = max_mvolt
        self.max_pcurrent_to_mvolt_ratio = max_pcurrent_to_mvolt_ratio
        self.pixel = pixel
        self.goal_nm = goal_nm

        self.template = template
        self.createc_controller = Createc_Controller(im_size_nm, offset_nm, pixel, scan_mV)
        self.current_jump = current_jump
        self.manip_limit_nm = manip_limit_nm
        if self.manip_limit_nm is not None:
            self.inner_limit_nm = self.manip_limit_nm + np.array([1,-1,1,-1])
        self.offset_nm = offset_nm
        self.len_nm = im_size_nm

        self.default_reward = -1
        self.default_reward_done = 1
        self.max_len = max_len
        self.correct_drift = correct_drift
        self.atom_absolute_nm = None
        self.atom_relative_nm = None
        self.template_max_y = template_max_y

        self.lattice_constant = 0.288
        self.precision_lim = self.lattice_constant*np.sqrt(3)/3
        self.bottom = bottom
        self.atom_move_detector = AtomJumpDetector_conv(data_len=2048, load_weight = load_weight)
        self.random_scan_rate = random_scan_rate
        self.accuracy, self.true_positive, self.true_negative = [], [], []

    def reset(self, update_conv_net = True):
        self.len = 0

        if (len(self.atom_move_detector.currents_val)>self.atom_move_detector.batch_size) and update_conv_net:
            accuracy, true_positive, true_negative = self.atom_move_detector.eval()
            self.accuracy.append(accuracy)
            self.true_positive.append(true_positive)
            self.true_negative.append(true_negative)
            self.atom_move_detector.train()

        if (self.atom_absolute_nm is None) or (self.atom_relative_nm is None):
            self.atom_absolute_nm, self.atom_relative_nm = self.scan_atom()

        if self.out_of_range(self.atom_absolute_nm, self.inner_limit_nm):
            print('Warning: atom is out of limit')
            self.pull_atom_back()
            self.atom_absolute_nm, self.atom_relative_nm = self.scan_atom()
        #goal_nm is set between 0.28 - 2 nm
        goal_nm = self.lattice_constant + np.random.random()*(self.goal_nm - self.lattice_constant)
        print('goal_nm:',goal_nm)
        self.atom_start_absolute_nm, self.atom_start_relative_nm = self.atom_absolute_nm, self.atom_relative_nm
        self.destination_relative_nm, self.destination_absolute_nm, self.goal = self.get_destination(self.atom_start_relative_nm, self.atom_start_absolute_nm, goal_nm)
        
        self.state = np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm))
        self.dist_destination = goal_nm

        info = {'start_absolute_nm':self.atom_start_absolute_nm, 'start_relative_nm':self.atom_start_relative_nm, 'goal_absolute_nm':self.destination_absolute_nm, 
                'goal_relative_nm':self.destination_relative_nm, 'start_absolute_nm_f':self.atom_absolute_nm_f, 'start_absolute_nm_b':self.atom_absolute_nm_b, 
                'start_relative_nm_f':self.atom_relative_nm_f, 'start_relative_nm_b':self.atom_relative_nm_b}
        return self.state, info
    
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
            done = done or (dist_start > 1.5*self.goal_nm) or (self.dist_destination < self.precision_lim) or self.out_of_range(self.atom_absolute_nm, self.manip_limit_nm)
            self.atom_move_detector.push(current_series, dist_last)

        next_state = np.concatenate((self.goal, (self.atom_absolute_nm -self.atom_start_absolute_nm)/self.goal_nm))
        reward = self.compute_reward(self.state, next_state)

        info |= {'dist_destination':self.dist_destination,
                'atom_absolute_nm':self.atom_absolute_nm, 'atom_relative_nm':self.atom_relative_nm, 'atom_absolute_nm_f':self.atom_absolute_nm_f,
                'atom_relative_nm_f' : self.atom_relative_nm_f, 'atom_absolute_nm_b': self.atom_absolute_nm_b, 'atom_relative_nm_b':self.atom_relative_nm_b,
                'img_info':self.img_info}
        self.state = next_state
        return next_state, reward, done, info

    def calculate_potential(self,state):
        dist = np.linalg.norm(state[:2]*self.goal_nm - state[2:]*self.goal_nm)
        return -dist/self.lattice_constant, dist

    def compute_reward(self, state, next_state):
        old_potential, _ = self.calculate_potential(state)
        new_potential, dist = self.calculate_potential(next_state)
        #print('old potential:', old_potential, 'new potential:', new_potential)
        reward = self.default_reward_done*(dist<self.precision_lim) + self.default_reward*(dist>self.precision_lim) + new_potential - old_potential
        return reward
    
    def scan_atom(self):
        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image()
        self.img_info = {'img_forward':img_forward,'img_backward':img_backward, 'offset_nm':offset_nm, 'len_nm':len_nm}
        atom_absolute_nm_f, atom_relative_nm_f, template_nm_f, template_wh_f  = get_atom_coordinate_nm(img_forward, offset_nm, len_nm, self.template, self.template_max_y, self.bottom)
        atom_absolute_nm_b, atom_relative_nm_b, template_nm_b, template_wh_b  = get_atom_coordinate_nm(img_backward, offset_nm, len_nm, self.template, self.template_max_y, self.bottom)
        self.atom_absolute_nm_f = atom_absolute_nm_f
        self.atom_relative_nm_f = atom_relative_nm_f
        self.atom_absolute_nm_b = atom_absolute_nm_b
        self.atom_relative_nm_b = atom_relative_nm_b

        self.atom_absolute_nm, self.atom_relative_nm, template_nm, self.template_wh = 0.5*(atom_absolute_nm_f+atom_absolute_nm_b), 0.5*(atom_relative_nm_f+atom_relative_nm_b), 0.5*(template_nm_f+template_nm_b), 0.5*(template_wh_b+template_wh_f)
        
        if self.out_of_range(self.atom_absolute_nm, self.manip_limit_nm):
            print('Warning: atom is out of limit')
        if self.correct_drift:
            try:
                template_drift = template_nm - self.template_nm
                max_drift_nm = 0.5
                if (np.linalg.norm(template_drift)>max_drift_nm):
                    print('Move offset_nm from:{} to:{}'.format((self.createc_controller.offset_nm, self.createc_controller.offset_nm+template_drift)))
                    print('Move manip_limit_nm from:{} to:{}'.format((self.createc_controller.offset_nm, self.createc_controller.offset_nm+template_drift)))
                    self.createc_controller.offset_nm+=template_drift
                    self.manip_limit_nm += np.array((template_drift[0], template_drift[0], template_drift[1], template_drift[1]))
                    self.inner_limit_nm = self.manip_limit_nm + np.array([1,-1,1,-1])
                    self.offset_nm = offset_nm
                    template_nm = self.template_nm
            except AttributeError:
                self.template_nm = template_nm
        self.template_nm = template_nm
        return self.atom_absolute_nm, self.atom_relative_nm
    
    def get_destination(self, atom_relative_nm, atom_absolute_nm, goal_nm):
        while True:
            r = np.random.random()
            angle = 2*np.pi*r
            dr = goal_nm*np.array([np.cos(angle), np.sin(angle)])
            destination_absolute_nm = atom_absolute_nm + dr
            if not self.out_of_range(destination_absolute_nm, self.inner_limit_nm):
                break
        destination_relative_nm = atom_relative_nm + dr
        return destination_relative_nm, destination_absolute_nm, dr/self.goal_nm
        
    def action_to_latman_input(self, action):
        x_start_nm = action[0]*self.step_nm
        y_start_nm = action[1]*self.step_nm
        x_end_nm = action[2]*self.goal_nm
        y_end_nm = action[3]*self.goal_nm
        mvolt = np.clip(action[4], a_min = None, a_max=1)*self.max_mvolt
        pcurrent = np.clip(action[5], a_min = None, a_max=1)*self.max_pcurrent_to_mvolt_ratio*mvolt
        return x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent

    def old_detect_current_jump(self, current):
        if current is not None:
            diff = findiff.FinDiff(0,1,acc=6)(current)[3:-3]
            return np.sum(np.abs(diff)>self.current_jump*np.std(current)) > 2
        else:
            return False

    def detect_current_jump(self, current):
        old_prediction = self.old_detect_current_jump(current)
        print('Old prediction:', old_prediction)

        if current is not None:
            success, prediction = self.atom_move_detector.predict(current)
            if success:
                print('cnn thinks there is atom movement')
                return True
            elif old_prediction:
                print('old prediction thinks there is atom movement')
                return True
            elif (np.random.random()>(self.random_scan_rate-0.2)) and (prediction>0.2):
                print('Random scan')
                return True
            elif np.random.random()>self.random_scan_rate:
                print('Random scan')
                return True
            else:
                print('CNN and old prediction both say no movement')
                return False
        else:
            print('CNN and old prediction both say no movement')
            return False
        
    def step_latman(self, x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent):
        x_start_nm+=self.atom_absolute_nm[0]
        x_end_nm+=self.atom_absolute_nm[0]
        y_start_nm+=self.atom_absolute_nm[1]
        y_end_nm+=self.atom_absolute_nm[1]
        x_start_nm = np.clip(x_start_nm, a_min=self.manip_limit_nm[0], a_max=self.manip_limit_nm[1])
        y_start_nm = np.clip(y_start_nm, a_min=self.manip_limit_nm[2], a_max=self.manip_limit_nm[3])
        x_end_nm = np.clip(x_end_nm, a_min=self.manip_limit_nm[0], a_max=self.manip_limit_nm[1])
        y_end_nm = np.clip(y_end_nm, a_min=self.manip_limit_nm[2], a_max=self.manip_limit_nm[3])
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
        
    def check_similarity(self):
        old_atom_absolute_nm = self.atom_absolute_nm
        self.atom_absolute_nm, self.atom_relative_nm = self.scan_atom()
        dist_destination = np.linalg.norm(self.atom_absolute_nm - self.destination_absolute_nm)
        dist_start = np.linalg.norm(self.atom_absolute_nm - self.atom_start_absolute_nm)
        dist_last = np.linalg.norm(self.atom_absolute_nm - old_atom_absolute_nm)
        return dist_destination, dist_start, dist_last
    
    def out_of_range(self, nm, limit_nm):
        out = np.any((nm-limit_nm[[0,2]])*(nm - limit_nm[[1,3]])>0, axis=-1)
        return out

    def pull_atom_back(self):
        print('pulling atom back to center')
        self.createc_controller.lat_manipulation(self.atom_absolute_nm[0], self.atom_absolute_nm[1], np.mean(self.manip_limit_nm[:2])+2*np.random.random()-1, np.mean(self.manip_limit_nm[2:])+2*np.random.random()-1, 2500, 65000, self.offset_nm, self.len_nm)

        








    
    
