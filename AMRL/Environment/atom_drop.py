from .createc_control import Createc_Controller
import numpy as np
from .get_atom_coordinate import get_all_atom_coordinate_nm
from .data_visualization import plot_atoms_and_design

class AtomDrop:
    def __init__(self, pixel, scan_mV, speed = None):
        self.createc_controller = Createc_Controller(pixel = pixel, scan_mV = scan_mV)
        if speed is not None:
            self.speed = speed
        else:
            self.speed = self.createc_controller.get_speed()
    def scan_all_atoms(self, offset_nm, len_nm):
        #self.createc_controller.stm.setparam('DX/DDeltaX', self.large_DX_DDeltaX)
        self.createc_controller.offset_nm = offset_nm
        self.createc_controller.im_size_nm = len_nm
        self.offset_nm = offset_nm
        self.len_nm = len_nm
        img_forward, img_backward, offset_nm, len_nm = self.createc_controller.scan_image(speed= self.speed)
        try:
            self.all_atom_absolute_nm_f = get_all_atom_coordinate_nm(img_forward, offset_nm, len_nm)
            self.all_atom_absolute_nm_b = get_all_atom_coordinate_nm(img_backward, offset_nm, len_nm)

            self.all_atom_absolute_nm_f = np.array(sorted(self.all_atom_absolute_nm_f, key = lambda x: (x[0], x[1])))
            self.all_atom_absolute_nm_b = np.array(sorted(self.all_atom_absolute_nm_b, key = lambda x: (x[0], x[1])))

            self.all_atom_absolute_nm = 0.5*(self.all_atom_absolute_nm_f+self.all_atom_absolute_nm_b)

            if len(self.all_atom_absolute_nm_b)!=len(self.all_atom_absolute_nm_f):
                print('length of list of atoms found in b and f different')
        except:
            self.all_atom_absolute_nm = self.all_atom_absolute_nm_f = self.all_atom_absolute_nm_b = None

        self.large_img_info = {'img_forward':img_forward,'img_backward':img_backward, 'offset_nm':offset_nm, 'len_nm':len_nm,
                                'all_atom_absolute_nm':self.all_atom_absolute_nm, 'all_atom_absolute_nm_f':self.all_atom_absolute_nm_f,
                                'all_atom_absolute_nm_b':self.all_atom_absolute_nm_b}
        return self.all_atom_absolute_nm

    def find_min_Z(self, init_Z, Z_step, x_nm, y_nm):
        all_atom = None
        Z = init_Z
        while (all_atom is None) or all_atom.size ==0:
            self.createc_controller.tip_form(Z, x_nm, y_nm)
            all_atom = self.scan_all_atoms(self.createc_controller.get_offset_nm(), self.createc_controller.get_len_nm())
            print('Z approach:', Z)
            plot_atoms_and_design(self.large_img_info, self.all_atom_absolute_nm, None, None, show_legend=False)
            Z+=Z_step
        print('Dropped atom:',all_atom,'using Approach Z:', Z)





    '''def find_drop_spot(self):
        pass

    def drop_new_atom(self):
        return new_atom_position, z_approach

    def pull_test(self):
        return pull_success


    def single_test_drop(self, z_approach, position):'''
