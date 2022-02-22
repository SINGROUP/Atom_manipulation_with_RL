from Environment.createc_control import Createc_Controller
import numpy as np
from matplotlib import pyplot as plt
#import get_atom_coordinate
#import importlib
#importlib.reload(get_atom_coordinate)
from Environment.get_atom_coordinate import get_atom_coordinate_nm
import scipy.spatial as spatial
import findiff
import pdb
import scipy
class RealExpEnv:
    
    def __init__(self, step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, template, current_jump, im_size_nm, offset_nm,
                 manip_limit_nm, pixel, template_max_y, template_min_x, scan_mV, max_len, correct_drift = False, bottom = True):
        
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
            self.goal_limit_nm = self.manip_limit_nm + np.array([2,-2,2,-2])
        self.offset_nm = offset_nm
        self.len_nm = im_size_nm

        self.default_reward = -0.2
        self.success_reward = 10
        self.max_len = max_len
        self.correct_drift = correct_drift
        self.atom_absolute_nm = None
        self.atom_relative_nm = None
        self.template_max_y = template_max_y
        self.template_min_x = template_min_x

        self.lattice_constant = 0.28
        self.bottom = bottom

        
        
    def reset(self):
        self.len = 0
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
        self.dist_destination = goal_nm
        self.cos_similarity_destination = 0
        self.old_value = self.calculate_value(self.dist_destination, self.cos_similarity_destination)
        info = {'start_absolute_nm':self.atom_start_absolute_nm, 'start_relative_nm':self.atom_start_relative_nm, 'goal_absolute_nm':self.destination_absolute_nm, 'goal_relative_nm':self.destination_relative_nm,
                'start_absolute_nm_f':self.atom_absolute_nm_f, 'start_absolute_nm_b':self.atom_absolute_nm_b, 'start_relative_nm_f':self.atom_relative_nm_f, 'start_relative_nm_b':self.atom_relative_nm_b}

        return np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm)), info
    
    def step(self, action):
        '''
        '''
        x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent = self.action_to_latman_input(action)
        current_series, d = self.step_latman(x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent)
        info = {'current_series':current_series}
        info['d'] = d
        info['start_nm'] = np.array([x_start_nm , y_start_nm])
        info['end_nm'] = np.array([x_end_nm , y_end_nm])

        done = False
        self.len+=1

        if self.len == self.max_len:
            done = True
            self.dist_destination, dist_start, self.cos_similarity_destination = self.check_similarity()

        else:
            jump = self.detect_current_jump(current_series)
            if jump:
                self.dist_destination, dist_start, self.cos_similarity_destination = self.check_similarity()
                print('atom moves by:', dist_start)
                if dist_start > 1.5*self.goal_nm or self.dist_destination < 0.5*self.lattice_constant:
                    done = True

        value = self.calculate_value(self.dist_destination, self.cos_similarity_destination)
        print('value:', value)
        info['value'] = value
        reward = self.default_reward + value - self.old_value
        self.old_value = value
        info['dist_destination'] = self.dist_destination
        info['cos_similarity_destination'] = self.cos_similarity_destination
        next_state = np.concatenate((self.goal, (self.atom_absolute_nm -self.atom_start_absolute_nm)/self.goal_nm))
        info['atom_absolute_nm'] = self.atom_absolute_nm
        info['atom_relative_nm'] = self.atom_relative_nm
        info['atom_absolute_nm_f'] = self.atom_absolute_nm_f
        info['atom_relative_nm_f'] = self.atom_relative_nm_f
        info['atom_absolute_nm_b'] = self.atom_absolute_nm_b
        info['atom_relative_nm_b'] = self.atom_relative_nm_b
        info['img_info'] = self.img_info

        return next_state, reward, done, info

    def calculate_value(self, dist_destination, cos_similarity_destination):
        value = self.success_reward*np.exp(-(dist_destination/(2*self.lattice_constant))**2) + 0.5*self.success_reward*cos_similarity_destination
        return value
    
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
        #print('forward:', atom_absolute_nm_f,'backward:',atom_absolute_nm_b)
        #self.atom_absolute_nm, self.atom_relative_nm, template_nm, self.template_wh = atom_absolute_nm_f, atom_relative_nm_f, template_nm_f, template_wh_f
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
            if not self.out_of_range(destination_absolute_nm, self.goal_limit_nm):
                break
        destination_relative_nm = atom_relative_nm + dr

        return destination_relative_nm, destination_absolute_nm, dr/self.goal_nm
        
    def action_to_latman_input(self, action):
        x_start_nm = action[0]*self.step_nm
        y_start_nm = action[1]*self.step_nm
        x_end_nm = action[2]*self.goal_nm
        y_end_nm = action[3]*self.goal_nm
        mvolt = np.clip(action[4], a_min = None, a_max=0.97)*self.max_mvolt
        pcurrent = np.clip(action[5], a_min = None, a_max=0.97)*self.max_pcurrent_to_mvolt_ratio*mvolt
        return x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent
    
    def detect_current_jump(self, current):
        
        if current is not None:
            diff = findiff.FinDiff(0,1,acc=6)(current)[3:-3]
            return np.sum(np.abs(diff)>self.current_jump*np.std(current)) > 2
        else:
            return False
        
    def step_latman(self, x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent):
        #pdb.set_trace()
        #print(x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent)
        x_start_nm+=self.atom_absolute_nm[0]
        x_end_nm+=self.atom_absolute_nm[0]
        y_start_nm+=self.atom_absolute_nm[1]
        y_end_nm+=self.atom_absolute_nm[1]
        x_start_nm = np.clip(x_start_nm, a_min=self.manip_limit_nm[0], a_max=self.manip_limit_nm[1])
        y_start_nm = np.clip(y_start_nm, a_min=self.manip_limit_nm[2], a_max=self.manip_limit_nm[3])
        x_end_nm = np.clip(x_end_nm, a_min=self.manip_limit_nm[0], a_max=self.manip_limit_nm[1])
        y_end_nm = np.clip(y_end_nm, a_min=self.manip_limit_nm[2], a_max=self.manip_limit_nm[3])
        if [x_start_nm, y_start_nm] != [x_end_nm, y_end_nm]:
            #print(x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent, self.offset_nm, self.len_nm)
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
        #pdb.set_trace()
        self.atom_absolute_nm, self.atom_relative_nm = self.scan_atom()
        dist_destination = np.linalg.norm(self.atom_absolute_nm - self.destination_absolute_nm)
        dist_start = np.linalg.norm(self.atom_absolute_nm - self.atom_start_absolute_nm)
        a = self.atom_absolute_nm - self.atom_start_absolute_nm
        b = self.destination_absolute_nm - self.atom_start_absolute_nm
        cos_similarity_destination = np.inner(a,b)/(self.goal_nm*np.clip(np.linalg.norm(a), a_min=self.goal_nm, a_max=None))
        return dist_destination, dist_start, cos_similarity_destination
    
    def out_of_range(self, nm, limit_nm):
        out = np.any((nm-limit_nm[[0,2]])*(nm - limit_nm[[1,3]])>0, axis=-1)
        return out

    def pull_atom_back(self):
        print('pulling atom back to center')
        self.createc_controller.lat_manipulation(self.atom_absolute_nm[0], self.atom_absolute_nm[1], np.mean(self.manip_limit_nm[:2])+2*np.random.random()-1, np.mean(self.manip_limit_nm[2:])+2*np.random.random()-1, 10, 57000, self.offset_nm, self.len_nm)

        








    
    
