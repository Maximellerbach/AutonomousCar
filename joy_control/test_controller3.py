# script based on https://github.com/autorope/donkeycar/blob/dev/donkeycar/parts/controller.py

import os
import struct
import array
import threading


class Joystick(object):
    '''
    An interface to a physical joystick
    '''

    def __init__(self, dev_fn='/dev/input/js0'):
        self.axis_states = {}
        self.button_states = {}
        self.axis_names = {}
        self.button_names = {}
        self.axis_map = []
        self.button_map = []
        self.jsdev = None
        self.dev_fn = dev_fn

    def init(self):
        try:
            from fcntl import ioctl
        except ModuleNotFoundError:
            self.num_axes = 0
            self.num_buttons = 0
            print("no support for fnctl module. joystick not enabled.")
            return False

        if not os.path.exists(self.dev_fn):
            print(self.dev_fn, "is missing")
            return False

        '''
        call once to setup connection to device and map buttons
        '''
        # Open the joystick device.
        print('Opening %s...' % self.dev_fn)
        self.jsdev = open(self.dev_fn, 'rb')

        # Get the device name.
        buf = array.array('B', [0] * 64)
        # JSIOCGNAME(len)
        ioctl(self.jsdev, 0x80006a13 + (0x10000 * len(buf)), buf)
        self.js_name = buf.tobytes().decode('utf-8')
        print('Device name: %s' % self.js_name)

        # Get number of axes and buttons.
        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a11, buf)  # JSIOCGAXES
        self.num_axes = buf[0]

        buf = array.array('B', [0])
        ioctl(self.jsdev, 0x80016a12, buf)  # JSIOCGBUTTONS
        self.num_buttons = buf[0]

        # Get the axis map.
        buf = array.array('B', [0] * 0x40)
        ioctl(self.jsdev, 0x80406a32, buf)  # JSIOCGAXMAP

        for axis in buf[:self.num_axes]:
            axis_name = self.axis_names.get(axis, 'unknown(0x%02x)' % axis)
            self.axis_map.append(axis_name)
            self.axis_states[axis_name] = 0.0

        # Get the button map.
        buf = array.array('H', [0] * 200)
        ioctl(self.jsdev, 0x80406a34, buf)  # JSIOCGBTNMAP

        for btn in buf[:self.num_buttons]:
            btn_name = self.button_names.get(btn, 'unknown(0x%03x)' % btn)
            self.button_map.append(btn_name)
            self.button_states[btn_name] = 0
            # print('btn', '0x%03x' % btn, 'name', btn_name)

        th = threading.Thread(target=self.poll)
        th.start()

        return True

    def show_map(self):
        '''
        list the buttons and axis found on this joystick
        '''
        print('%d axes found: %s' % (self.num_axes, ', '.join(self.axis_map)))
        print('%d buttons found: %s' %
              (self.num_buttons, ', '.join(self.button_map)))

    def poll(self):
        '''
        query the state of the joystick, returns button which was pressed, if any,
        and axis which was moved, if any. button_state will be None, 1, or 0 if no changes,
        pressed, or released. axis_val will be a float from -1 to +1. button and axis will
        be the string label determined by the axis map in init.
        '''

        while True:
            if self.jsdev is None:
                break

            # Main event loop
            evbuf = self.jsdev.read(8)

            if evbuf:
                tval, value, typev, number = struct.unpack('IhBB', evbuf)

                if typev & 0x80:
                    # ignore initialization event
                    pass

                if typev & 0x01:
                    button = self.button_map[number]
                    # print(tval, value, typev, number, button, 'pressed')
                    if button:
                        self.button_states[button] = value

                if typev & 0x02:
                    axis = self.axis_map[number]
                    if axis:
                        fvalue = value / 32767.0
                        self.axis_states[axis] = fvalue


class XboxOneJoystick(Joystick):
    '''
    An interface to a physical joystick 'Xbox Wireless Controller' controller.
    This will generally show up on /dev/input/js0.
    - Note that this code presumes the built-in linux driver for 'Xbox Wireless Controller'.
      There is another user land driver called xboxdrv; this code has not been tested
      with that driver.
    - Note that this controller requires that the bluetooth disable_ertm parameter
      be set to true; to do this:
      - edit /etc/modprobe.d/xbox_bt.conf
      - add the line: options bluetooth disable_ertm=1
      - reboot to tha this take affect.
      - after reboot you can vertify that disable_ertm is set to true entering this
        command oin a terminal: cat /sys/module/bluetooth/parameters/disable_ertm
      - the result should print 'Y'.  If not, make sure the above steps have been done corretly.

    credit:
    https://github.com/Ezward/donkeypart_ps3_controller/blob/master/donkeypart_ps3_controller/part.py
    '''

    def __init__(self, *args, **kwargs):
        super(XboxOneJoystick, self).__init__(*args, **kwargs)

        self.axis_names = {
            0x00: 'left_stick_horz',
            0x01: 'left_stick_vert',
            0x05: 'right_stick_vert',
            0x02: 'right_stick_horz',
            0x0a: 'left_trigger',
            0x09: 'right_trigger',
            0x10: 'dpad_horiz',
            0x11: 'dpad_vert'
        }

        self.button_names = {
            0x130: 'a_button',
            0x131: 'b_button',
            0x133: 'x_button',
            0x134: 'y_button',
            0x13b: 'options',
            0x136: 'left_shoulder',
            0x137: 'right_shoulder',
        }


if __name__ == "__main__":
    import time

    # Testing the XboxOneJoystickController
    js = XboxOneJoystick('/dev/input/js0')
    js.init()

    while True:
        print(js.axis_states)
        print(js.button_states)
        time.sleep(0.05)
