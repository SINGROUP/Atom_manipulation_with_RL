import win32com.client
import os, pdb
import time
import numpy as np
from collections import namedtuple

latmandata = namedtuple('latmandata',['time','x','y','current','dI_dV','topography'])

class Createc_Controller:
    """
    Get, set parameters, execute scan, tipform, manipulations in Createc STM
    """

    def __init__(self,
                 im_size_nm: float=None,
                 offset_nm: np.array=None,
                 pixel: int=None,
                 scan_mV: float=None) -> None:
        """Connect with Createc STM and set default scan size, offset, pixel, and bias

        Parameters
        ----------

        im_size_nm : float, optional
            image size in nm

        offset_nm : array_like, optional
            xy offset in nm

        pixel : int, optional
            scan pixel

        scan_mV : float
            scan bias in mV

        Returns
        -------

        None : None

        """

        if im_size_nm is not None:
            self.im_size_nm = im_size_nm
        if offset_nm is not None:
            self.offset_nm = offset_nm
        if pixel is not None:
            self.pixel = pixel
        if scan_mV is not None:
            self.scan_mV = scan_mV
        self.stm=win32com.client.Dispatch("pstmafm.stmafmrem")
        if self.stm.stmready()==1:
            print('succeed to connect')
        else:
            self.stm = win32com.client.DispatchEx("pstmafm.stmafmrem")
            if self.stm.stmready()==1:
                print('succeed to connect with DispatchEx')

    def scan_image(self,
                   DIR_NAME: str = None,
                   BASE_NAME: str = None,
                   counter: int = None,
                   save: bool = False,
                   speed: float = None) -> tuple:
        """Take a STM scan in Createc

        Parameters
        ----------

        pixel : int, optional

        offset_x_nm, offset_y_nm : float, optional
            offset xy value in nm

        DIR_NAME, BASE_NAME : str, optional
            image directory, name

        counter : int, optional
            image number

        Returns
        -------

        image: array_like

        image_size: tuple
            (x_length[nm], y_length[nm])

        """
        self.ramp_bias_mV(self.scan_mV)
        DAC_unit = 2**19
        self.stm.setparam('Num.X',self.pixel)
        GainX = float(self.stm.getparam("GainX"))
        Delta_X = self.get_Delta_X(self.im_size_nm)
        self.stm.setparam('Delta X [Dac]',Delta_X)
        time.sleep(0.1)
        scan_time = self.get_scan_time()
        print('The scan will take {0:.1f} seconds'.format(scan_time))
        if scan_time > 600:
            approve = input('Scan will take {} minutes, \
                type yes to continue and no to interrupt'.format(scan_time/60))
            if approve == 'no':
                return None, (None, None)
        Xpiezoconst = float(self.stm.getparam("Xpiezoconst"))
        Ypiezoconst = float(self.stm.getparam("Ypiezoconst"))
        xV = GainX*self.offset_nm[0]/Xpiezoconst
        yV = GainX*self.offset_nm[1]/Ypiezoconst
        self.stm.setxyoffvolt(xV, yV)
        if speed is not None:
            self.set_speed(speed)
        self.stm.scanstart() #Starts a new STM scan. Similar to pressing the button Scanstart
        time.sleep(scan_time -2)
        while True:
            time.sleep(2)
            scanstatus = self.stm.scanstatus
            if scanstatus!=2:
                break
        if save:
            if DIR_NAME and BASE_NAME and counter:
                path = os.path.join(DIR_NAME, BASE_NAME + str(counter))
                self.stm.filesave(path)
            else:
                self.stm.quicksave()

        img_forward = np.array(self.stm.scandata(1,4))
        img_backward = np.array(self.stm.scandata(257,4))

        x_length = 0.1*float(self.stm.getparam('Length x[A]'))
        y_length = 0.1*float(self.stm.getparam('Length y[A]'))
        imgs = img_forward, img_backward
        rets = *imgs, self.offset_nm, np.array([x_length, y_length])
        return rets

    def lat_manipulation(self,
                         x_start_nm: float,
                         y_start_nm: float,
                         x_end_nm: float,
                         y_end_nm: float,
                         mvoltage: float,
                         pcurrent: float,
                         offset_nm: float,
                         len_nm: float) -> namedtuple:
        """Execute lateral manipulation in Createc

        Parameters
        ----------

        x_start, y_start, x_end, y_end : float
            x, y start and end position in nm relative to global origin

        Returns
        -------

        namedtuple (['time','x','y','current','dI_dV','topography'])

        """
        args = x_start_nm, y_start_nm, x_end_nm, y_end_nm, offset_nm, len_nm
        rets = self.nm_to_pixel(*args)
        x_start_pixel, y_start_pixel, x_end_pixel, y_end_pixel = rets
        # print(x_start_pixel, y_start_pixel, x_end_pixel, y_end_pixel)
        if [x_start_pixel, y_start_pixel]!=[x_end_pixel,y_end_pixel]:
            self.ramp_bias_mV(mvoltage)
            preamp_grain = 10**float(self.stm.getparam("Latmangain"))
            self.stm.setparam("LatmanVolt",  mvoltage) #(mV)
            self.stm.setparam("Latmanlgi", pcurrent*1e-9*preamp_grain) #(pA)
            args = x_start_pixel,y_start_pixel,x_end_pixel,y_end_pixel
            self.stm.latmanip(*args) #unit: image pixel
            #Channel: 0: time in sec 1: X 2: Y 3: Current I 4: dI/dV 5: d2I/dV 6: ADC0 7: ADC1 8: ADC2 9: ADC3 10: df 11: Damping 12: Amplitude 13: di_q 14: di2_q 15: Topography(DAC0) 16: CP(DAC1)
            #Units: 0: Default 1: Volt 2: DAC 3: Ampere 4: nm 5: Hz
            time = self.stm.latmandata(0, 0)
            x= self.stm.latmandata(1,4)
            y = self.stm.latmandata(2,4)
            current = self.stm.latmandata(3,3)
            dI_dV = self.stm.latmandata(4,0)
            topography = self.stm.latmandata(15,4)
            data = latmandata(time,x,y,current, dI_dV,topography)
            return data
        else:
            return None

    def get_Delta_X(self,
                    im_size_nm: float) -> int:
        """Get the DeltaX value for a given image size

        Parameters
        ----------

        im_size_nm : float
            image size in nm

        Returns
        -------

        None : None

        """
        DAC_unit = 2**19
        volt_unit = 10
        GainX = float(self.stm.getparam("GainX"))
        assert GainX != 0, "GainX shouldn't be 0"
        Xpiezoconst = float(self.stm.getparam("Xpiezoconst"))
        Nx = float(self.stm.getparam('Num.X'))
        Delta_X = im_size_nm*10/(Nx*volt_unit*GainX*Xpiezoconst/DAC_unit)
        return int(Delta_X)

    def get_scan_time(self) -> float:
        """Estimate scan time

        Returns
        -------

        scan_time: float

        """
        scan_time = float(self.stm.getparam('Sec/Image:'))
        delayY = self.stm.getparam('Delay Y')
        scan_time = scan_time / 2 * (1 + 1 / float(delayY))
        return scan_time

    def nm_to_pixel(self,
                    x_start_nm: float,
                    y_start_nm: float,
                    x_end_nm: float,
                    y_end_nm: float,
                    offset_nm: np.array,
                    len_nm: float) -> np.array:
        """Convert values from STM coordinates (nm) to pixel coordinates

        Parameters
        ----------
        x_start_nm, y_start_nm, x_end_nm, y_end_nm : float
            tip movement positions in STM coordinates (nm)

        offset_nm : array_like
            the XY offset value in STM coordinates (nm)

        len_nm : float
            image size in nm

        Returns
        -------
        x_start_pixel, y_start_pixel, x_end_pixel, y_end_pixel : int
            tip movement positions in pixel coordinates

        """
        DeltaX = float(self.stm.getparam('Delta X [Dac]'))
        DAC_unit = 2**19
        volt_unit = 10
        GainX = float(self.stm.getparam("GainX"))
        Xpiezoconst = float(self.stm.getparam("Xpiezoconst"))
        Nx = float(self.stm.getparam('Num.X'))
        pixel_to_A_X = DeltaX*10*GainX*Xpiezoconst/DAC_unit
        x_start_pixel = (x_start_nm-(offset_nm[0]-0.5*len_nm))*10/pixel_to_A_X
        x_start_pixel = int(np.rint(x_start_pixel))
        y_start_pixel = (y_start_nm - offset_nm[1])*10/pixel_to_A_X
        y_start_pixel = int(np.rint(y_start_pixel))
        if x_end_nm is not None:
            x_end_pixel = (x_end_nm - (offset_nm[0]-0.5*len_nm))*10/pixel_to_A_X
            x_end_pixel = int(np.rint(x_end_pixel))
        else:
            x_end_pixel = None

        if y_end_nm is not None:
            y_end_pixel = (y_end_nm - offset_nm[1])*10/pixel_to_A_X
            y_end_pixel = int(np.rint(y_end_pixel))
        else:
            y_end_pixel = None

        return x_start_pixel, y_start_pixel, x_end_pixel, y_end_pixel

    def set_xy_nm(self, nm: float) -> None:
        """
        set xy offset value in nm

        Parameters
        ----------
        x_nm, y_nm: float
            target coordinate  in nm

        Returns
        -------
        None : None
        """
        Xpiezoconst = float(self.stm.getparam("Xpiezoconst"))
        Ypiezoconst = float(self.stm.getparam("Ypiezoconst"))
        self.stm.setxyoffvolt(10*nm[0]/Xpiezoconst,10*nm[1]/Ypiezoconst)

    def get_xy_nm(self) -> np.array:
        """
        get xy offset value in nm

        Return
        ------
        (x,y) : tuple
            x, y poisiton of the tip in nm
        """
        DAC_unit = 2**19
        volt_unit = 10
        Xgain = float(self.stm.getparam("GainX"))
        Ygain = float(self.stm.getparam("GainY"))
        Xpiezoconst = float(self.stm.getparam("Xpiezoconst"))
        Ypiezoconst = float(self.stm.getparam("Ypiezoconst"))
        offsetX = self.stm.getparam('OffsetX')
        offsetY = self.stm.getparam('OffsetY')
        x_nm = -0.1*Xpiezoconst*volt_unit*float(offsetX)*Xgain/DAC_unit
        y_nm = -0.1*Ypiezoconst*volt_unit*float(offsetY)*Ygain/DAC_unit
        return np.array([x_nm, y_nm])

    def _ramp_bias_same_pole(self,
                             _end_bias_mV: float,
                             _init_bias_mV: float,
                             _speed: float) -> None:
        """To be called by ramp_bias_mV().
        The end result is the machine will ramp the bias gradually to the target value.
        Code originally from py_createc package, author: Chen Xu

        Parameters
        ----------
        _end_bias_mV : float
            target bias in mV

        _init_bias_mV : float
            starting bias in mV, it should be of the same polarity of _end_bias_mV

        _speed : int
            speed is actually steps, it can be any integer larger than 0.
            1 means directly stepping to the final bias, it is default to 100.

        Returns
        -------
        None : None
        """
        bias_pole = np.sign(_init_bias_mV)
        init = _speed * np.log10(np.abs(_init_bias_mV))
        end = _speed * np.log10(np.abs(_end_bias_mV))
        sign = np.int(np.sign(end - init))
        for i in range(np.int(init) + sign, np.int(end) + sign, sign):
            time.sleep(0.01)
            self.stm.setparam('Biasvolt.[mV]', bias_pole * 10 ** ((i) / _speed))
        self.stm.setparam('Biasvolt.[mV]', _end_bias_mV)

    def ramp_bias_mV(self,
                     end_bias_mV: float,
                     speed: int = 2) -> None:
        """Ramp bias from current value to another value
        Code originally from py_createc package, author: Chen Xu
        https://py-createc.readthedocs.io/en/latest/

        Parameters
        ----------
        end_bias_mV : float
            target bias in mV

        speed : int
            speed is actually steps, it can be any integer larger than 0.
            1 means directly stepping to the final value, it is default to 100.

        Returns
        -------
        None : None
        """
        speed = int(speed)
        assert speed > 0, "speed should be larger than 0"

        init_bias_mV = float(self.stm.getparam('Biasvolt.[mV]'))
        if init_bias_mV * end_bias_mV == 0:
            pass
        elif init_bias_mV == end_bias_mV:
            pass
        elif init_bias_mV * end_bias_mV > 0:
            self._ramp_bias_same_pole(end_bias_mV, init_bias_mV, speed)
        else:
            if np.abs(init_bias_mV) > np.abs(end_bias_mV):
                self.stm.setparam('Biasvolt.[mV]', -init_bias_mV)
                self._ramp_bias_same_pole(end_bias_mV, -init_bias_mV, speed)
            elif np.abs(init_bias_mV) < np.abs(end_bias_mV):
                self._ramp_bias_same_pole(-end_bias_mV, init_bias_mV, speed)
                self.stm.setparam('Biasvolt.[mV]', end_bias_mV)
            else:
                self.stm.setparam('Biasvolt.[mV]', end_bias_mV)

    def set_speed(self,
                  speed: float) -> None:
        """Set the scan speed

        Parameters
        ----------

        speed: float
            speed in A/s

        Returns
        -------

        None : None

        """
        GainX = float(self.stm.getparam("GainX"))
        DeltaX = float(self.stm.getparam("Delta X [Dac]"))
        voltage_unit = 10
        Xpiezoconst = float(self.stm.getparam("Xpiezoconst"))
        #speed = DeltaX*voltage_unit*GainX*Piezoconstant/(2**19*DX_DDeltaX*20E-6)
        DX_DDeltaX = (DeltaX*voltage_unit*GainX*Xpiezoconst)/(speed*2**19*20E-6)
        self.stm.setparam('DX/DDeltaX', int(DX_DDeltaX))

    def get_speed(self) -> float:
        """Get the scan speed in A/s

        Returns
        -------

        speed : float

        """
        GainX = float(self.stm.getparam("GainX"))
        DeltaX = float(self.stm.getparam("Delta X [Dac]"))
        voltage_unit = 10
        Xpiezoconst = float(self.stm.getparam("Xpiezoconst"))
        DX_DDeltaX = float(self.stm.getparam("DX/DDeltaX"))
        speed = DeltaX*voltage_unit*GainX*Xpiezoconst/(2**19*DX_DDeltaX*20E-6)
        return speed

    def set_Z_approach(self, A: float) -> None:
        """Set Z approach value in A

        Parameters
        ----------

        A : float
            Z approach value in A

        Returns
        -------

        None : None

        """
        Zpiezoconst = float(self.stm.getparam("Zpiezoconst"))
        self.stm.setparam('TipForm_Z', 1000*A/Zpiezoconst)

    def tip_form(self, A: float, x_nm: float, y_nm: float) -> None:
        """Perform tip forming

        Parameters
        ----------

        A : float
            Z approach value in A

        x_nm, y_nm : float
            STM coordinates (nm)

        """
        offset_nm = self.get_offset_nm()
        len_nm = self.get_len_nm()
        self.set_Z_approach(A)
        args = x_nm, y_nm, None, None, offset_nm, len_nm
        x_pixel, y_pixel, _, _ = self.nm_to_pixel(*args)
        self.stm.btn_tipform(x_pixel, y_pixel)
        self.stm.waitms(50)

    def get_offset_nm(self) -> np.array:
        """Get XY offset value in nm

        Returns
        -------

        x_nm, y_nm : float
            XY offset value in nm

        """
        DAC_unit = 2**19
        volt_unit = 10
        Xgain = float(self.stm.getparam("GainX"))
        Ygain = float(self.stm.getparam("GainY"))
        Xpiezoconst = float(self.stm.getparam("Xpiezoconst"))
        Ypiezoconst = float(self.stm.getparam("Ypiezoconst"))
        offsetX = self.stm.getparam('OffsetX')
        offsetY = self.stm.getparam('OffsetY')
        x_nm = -0.1*Xpiezoconst*volt_unit*float(offsetX)*Xgain/DAC_unit
        y_nm = -0.1*Ypiezoconst*volt_unit*float(offsetY)*Ygain/DAC_unit
        return x_nm, y_nm

    def get_len_nm(self) -> float:
        """Get image size value in nm

        Returns
        -------

        len_nm : float
            image size in nm

        """
        DAC_unit = 2**19
        volt_unit = 10
        GainX = float(self.stm.getparam("GainX"))
        Xpiezoconst = float(self.stm.getparam("Xpiezoconst"))
        Nx = float(self.stm.getparam('Num.X'))
        Delta_X = float(self.stm.getparam('Delta X [Dac]'))
        len_nm = Delta_X*Nx*volt_unit*GainX*Xpiezoconst/(10*DAC_unit)
        return len_nm
