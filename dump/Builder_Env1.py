from Env1 import RealExpEnv
from createc_control import Createc_Controller
import numpy as np
from matplotlib import pyplot as plt
import get_atom_coordinate
import importlib
importlib.reload(get_atom_coordinate)
from get_atom_coordinate import get_atom_coordinate_nm, get_all_atom_coordinate_nm
import scipy.spatial as spatial
import findiff
import pdb
import scipy

class Structure_Builder(RealExpEnv):
    def __init__(self, step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, current_jump, im_size_nm, offset_nm,
                 manip_limit_nm, pixel, scan_mV, max_len, correct_drift = False):
        super(Structure_Builder, self).__init__(step_nm, max_mvolt, max_pcurrent_to_mvolt_ratio, goal_nm, None, current_jump, im_size_nm, offset_nm,
                 manip_limit_nm, pixel, None, None, scan_mV, max_len, correct_drift = False)
        self.atom_absolute_nm_f = None
        self.atom_absolute_nm_b = None
        self.large_DX_DDeltaX = float(self.createc_controller.stm.getparam('DX/DDeltaX'))

    def reset(self, final_destination_absolute_nm, offset_nm, len_nm, large_len_nm):
        self.len = 0
        self.atom_absolute_nm = self.scan_atom(offset_nm, len_nm, large_len_nm)

        self.atom_start_absolute_nm = self.atom_absolute_nm
        self.destination_absolute_nm, self.goal = self.get_destination(self.atom_start_absolute_nm, final_destination_absolute_nm)
        info = {'start_absolute_nm':self.atom_start_absolute_nm, 'goal_absolute_nm':self.destination_absolute_nm,
                'start_absolute_nm_f':self.atom_absolute_nm_f, 'start_absolute_nm_b':self.atom_absolute_nm_b, 'img_info':self.img_info}
        return np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm)), info


    def step(self, action):
        x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent = self.action_to_latman_input(action)
        print(x_start_nm , y_start_nm, x_end_nm, y_end_nm, mvolt, pcurrent)
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
                if (dist_start/self.goal_nm) > 1.5 or self.dist_destination < 0.16:
                    done = True


        next_state = np.concatenate((self.goal, (self.atom_absolute_nm - self.atom_start_absolute_nm)/self.goal_nm))

        info['atom_absolute_nm'] = self.atom_absolute_nm
        info['atom_absolute_nm_f'] = self.atom_absolute_nm_f
        info['atom_absolute_nm_b'] = self.atom_absolute_nm_b
        info['img_info'] = self.img_info
        return next_state, None, done, info

    def scan_atom(self, offset_nm = None, len_nm = None, large_len_nm = None):
        if offset_nm is None:
            offset_nm = self.offset_nm
        if len_nm is None:
            len_nm = self.len_nm

        if large_len_nm is not None:
            small_DX_DDeltaX = int(self.large_DX_DDeltaX*len_nm/large_len_nm)
            self.createc_controller.stm.setparam('DX/DDeltaX', small_DX_DDeltaX)
        self.createc_controller.offset_nm = offset_nm
        self.createc_controller.im_size_nm = len_nm
        self.offset_nm = offset_nm
        self.len_nm = len_nm
        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image()
        self.img_info = {'img_forward':img_forward,'img_backward':img_backward, 'offset_nm':offset_nm, 'len_nm':len_nm}
        self.atom_absolute_nm_f, _, _, _ = get_atom_coordinate_nm(img_forward, offset_nm, len_nm, None, None)

        self.atom_absolute_nm_b, _, _, _ = get_atom_coordinate_nm(img_backward, offset_nm, len_nm, None, None)
        self.atom_absolute_nm = 0.5*(self.atom_absolute_nm_f+self.atom_absolute_nm_b)
        print('forward:', self.atom_absolute_nm_f,'backward:',self.atom_absolute_nm_b)
        return self.atom_absolute_nm

        '''    def remove_outliers(self,all_atom_absolute_nm_b,all_atom_absolute_nm_f):
        pdb.set_trace()
        shorter_array = all_atom_absolute_nm_b if len(all_atom_absolute_nm_b)<len(all_atom_absolute_nm_f) else all_atom_absolute_nm_f
        longer_array = all_atom_absolute_nm_b if len(all_atom_absolute_nm_b)>len(all_atom_absolute_nm_f) else all_atom_absolute_nm_f
        nearest_ys = []
        for atom in longer_array:
            nearest_ys.append(sorted(atom-shorter_array, key=lambda x: x[1])[0])
        return all_atom_absolute_nm_b, all_atom_absolute_nm_f'''


    def scan_all_atoms(self, offset_nm, len_nm):
        self.createc_controller.stm.setparam('DX/DDeltaX', self.large_DX_DDeltaX)
        self.createc_controller.offset_nm = offset_nm
        self.createc_controller.im_size_nm = len_nm
        self.offset_nm = offset_nm
        self.len_nm = len_nm
        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image()
        print(offset_nm)
        print(len_nm)
        all_atom_absolute_nm_f = get_all_atom_coordinate_nm(img_forward, offset_nm, len_nm)
        all_atom_absolute_nm_b = get_all_atom_coordinate_nm(img_backward, offset_nm, len_nm)

        all_atom_absolute_nm_f = np.array(sorted(all_atom_absolute_nm_f, key = lambda x: x[0]))
        all_atom_absolute_nm_f = np.array(sorted(all_atom_absolute_nm_f, key = lambda x: x[1]))

        all_atom_absolute_nm_b = np.array(sorted(all_atom_absolute_nm_b, key = lambda x: x[0]))
        all_atom_absolute_nm_b = np.array(sorted(all_atom_absolute_nm_b, key = lambda x: x[1]))

        self.all_atom_absolute_nm_f = all_atom_absolute_nm_f
        self.all_atom_absolute_nm_b = all_atom_absolute_nm_b

        if len(all_atom_absolute_nm_b)!=len(all_atom_absolute_nm_f):
            print('length of list of atoms found in b and f different')
         #   all_atom_absolute_nm_b,all_atom_absolute_nm_f = self.remove_outliers(all_atom_absolute_nm_b,all_atom_absolute_nm_f)

        all_atom_absolute_nm = 0.5*(all_atom_absolute_nm_f+all_atom_absolute_nm_b)
        self.all_atom_absolute_nm = all_atom_absolute_nm
        return all_atom_absolute_nm, img_forward, img_backward, offset_nm, len_nm

    def get_destination(self, atom_start_absolute_nm, final_destination_absolute_nm):
        angle = np.arctan2((final_destination_absolute_nm-atom_start_absolute_nm)[1],(final_destination_absolute_nm-atom_start_absolute_nm)[0])
        goal_nm = min(self.goal_nm, np.linalg.norm(final_destination_absolute_nm-atom_start_absolute_nm))
        destination_absolute_nm = atom_start_absolute_nm + goal_nm*np.array([np.cos(angle),np.sin(angle)])

        return destination_absolute_nm, goal_nm*np.array([np.cos(angle),np.sin(angle)])

    def check_similarity(self):
        #pdb.set_trace()
        self.atom_absolute_nm = self.scan_atom()

        dist_destination = np.linalg.norm(self.atom_absolute_nm - self.destination_absolute_nm)
        dist_start = np.linalg.norm(self.atom_absolute_nm - self.atom_start_absolute_nm)
        a = self.atom_absolute_nm - self.atom_start_absolute_nm
        b = self.destination_absolute_nm - self.atom_start_absolute_nm
        cos_similarity_destination = np.inner(a,b)/(self.goal_nm*np.clip(np.linalg.norm(a), a_min=self.goal_nm, a_max=None))
        return dist_destination, dist_start, cos_similarity_destination

    def get_script_start_end(self, atom_start_absolute_nm, final_destination_absolute_nm):
        goal_nm = np.linalg.norm(final_destination_absolute_nm - atom_start_absolute_nm)
        goal_nm = np.clip(goal_nm, a_min = None, a_max = self.goal_nm)

        angle = np.arctan2((final_destination_absolute_nm-atom_start_absolute_nm)[1],(final_destination_absolute_nm-atom_start_absolute_nm)[0])
        destination_absolute_nm = atom_start_absolute_nm + goal_nm*np.array([np.cos(angle),np.sin(angle)])
        return destination_absolute_nm


    def step_latman(self, x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent):
        #pdb.set_trace()
        #print(x_start_nm, y_start_nm, x_end_nm, y_end_nm, mvoltage, pcurrent)
        x_start_nm+=self.atom_absolute_nm[0]
        x_end_nm+=self.atom_absolute_nm[0]
        y_start_nm+=self.atom_absolute_nm[1]
        y_end_nm+=self.atom_absolute_nm[1]
        '''[x_start_nm, y_start_nm] = self.tip_nm
        [x_end_nm, y_end_nm] = self.tip_nm + np.array([dx_nm, dy_nm])
        x_end_nm = np.clip(x_end_nm, a_min=self.manip_limit_nm[0], a_max=self.manip_limit_nm[1])
        y_end_nm = np.clip(y_end_nm, a_min=self.manip_limit_nm[2], a_max=self.manip_limit_nm[3])
        norm = np.linalg.norm(np.array([x_end_nm, y_end_nm]) - self.atom_start_absolute_nm)
        if norm > self.goal_nm:
            [x_end_nm, y_end_nm] = self.atom_start_absolute_nm + self.goal_nm*(np.array([x_end_nm, y_end_nm]) - self.atom_start_absolute_nm)/norm
        x_end_nm = np.clip(x_end_nm, a_min=self.manip_limit_nm[0], a_max=self.manip_limit_nm[1])
        y_end_nm = np.clip(y_end_nm, a_min=self.manip_limit_nm[2], a_max=self.manip_limit_nm[3])'''
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
