import os
from glob import glob
import tkinter


class windowInterface(tkinter.Tk):  # screen to display interface for ALL client
    def __init__(self, name="basic interface"):
        super().__init__(name)
        self.n_client = 0
        self.off_y = 0

    def increment(self, space=2):
        self.n_client += 1
        self.off_y += space


class AutoInterface():  # single interface object for 1 client
    def __init__(self, window, client_class, screen_size=(512, 512), name="Auto_0"):
        self.window = window
        self.client_class = client_class

        self.screen_size = screen_size
        self.name = name

        self.scales_value = []  # list of scales objects in interface
        self.scale_default = self.client_class.PID_settings + \
            [self.client_class.buffer_time]
        self.box_default = self.client_class.loop_settings

        self.values = []  # list of values of scales in interface
        self.bool_checkbox = []  # list of checkbox objects in interface

        self.add_interface()

    def add_interface(self):
        off_y = self.window.off_y
        last_button = 0
        scale_labels = ["max_speed", "max_throttle", "min_throttle", "sq", "mult", "fake_delay"]
        from_to = [(1, 30), (0, 1), (0, 1), (0.5, 1.5), (0, 2), (0, 500)]

        tkinter.Button(self.window, text="Respawn", command=self.respawn).grid(
            row=off_y, column=last_button)
        last_button += 1
        tkinter.Button(self.window, text="Terminate", command=self.client_class.terminate).grid(
            row=off_y, column=last_button)
        last_button += 1
        tkinter.Button(self.window, text="Reset to default", command=self.reset).grid(
            row=off_y, column=last_button)
        last_button += 1
        tkinter.Button(self.window, text="init car", command=self.client_class.rdm_color_startv1).grid(
            row=off_y, column=last_button)
        last_button += 1

        bvar = tkinter.BooleanVar()
        b = tkinter.Checkbutton(self.window, text="Transform st", variable=bvar,
                                onvalue=True, offvalue=False, command=self.get_checkbox_value)
        b.grid(row=off_y, column=last_button)
        last_button += 1
        self.bool_checkbox.append(bvar)

        bvar = tkinter.BooleanVar()
        b = tkinter.Checkbutton(self.window, text="Smooth", variable=bvar,
                                onvalue=True, offvalue=False, command=self.get_checkbox_value)
        b.grid(row=off_y, column=last_button)
        last_button += 1
        self.bool_checkbox.append(bvar)

        bvar = tkinter.BooleanVar()
        b = tkinter.Checkbutton(self.window, text="Random", variable=bvar,
                                onvalue=True, offvalue=False, command=self.get_checkbox_value)
        b.grid(row=off_y, column=last_button)
        last_button += 1
        self.bool_checkbox.append(bvar)

        bvar = tkinter.BooleanVar()
        b = tkinter.Checkbutton(self.window, text="Do_overide_st", variable=bvar,
                                onvalue=True, offvalue=False, command=self.get_checkbox_value)
        b.grid(row=off_y, column=last_button)
        last_button += 1
        self.bool_checkbox.append(bvar)

        self.record_bool = tkinter.BooleanVar()
        b = tkinter.Checkbutton(self.window, text="Record", variable=self.record_bool,
                                onvalue=True, offvalue=False, command=self.get_record)
        b.grid(row=off_y, column=last_button)
        last_button += 1
        self.bool_checkbox.append(self.record_bool)

        bvar = tkinter.BooleanVar()
        b = tkinter.Checkbutton(self.window, text="Stop", variable=bvar,
                                onvalue=True, offvalue=False, command=self.get_checkbox_value)
        b.grid(row=off_y, column=last_button)
        last_button += 1
        self.bool_checkbox.append(bvar)

        for it, label, scale_range in zip(range(len(scale_labels)), scale_labels, from_to):
            value = tkinter.DoubleVar()  # dummy value
            s = tkinter.Scale(self.window, resolution=0.05, variable=value, command=self.get_slider_value,
                              label=label, length=75, width=15, from_=scale_range[0], to=scale_range[1])
            s.grid(row=off_y+1, column=it)
            self.scales_value.append(value)

        self.reset()
        # add 1 to the client counter on the screen and add some offset to avoid overlapping
        self.window.increment(space=2)

    def respawn(self):
        self.client_class.reset_car()

    def get_slider_value(self, v=0):
        values = []
        for scale in self.scales_value:
            values.append(scale.get())

        self.client_class.PID_settings = values[:-1]
        self.client_class.buffer_time = values[-1]/1000

    def get_checkbox_value(self, v=0):
        bools = []
        for box in self.bool_checkbox:
            bools.append(box.get())

        self.client_class.loop_settings = bools

    def get_record(self, v=0):
        self.get_checkbox_value()
        self.client_class.record = self.record_bool.get()

        # create the dir in order to be able to save img
        if not os.path.exists(self.client_class.default_dos) and self.client_class.record:
            os.makedirs(self.client_class.default_dos)

        # delete the dir if no items are inside
        if os.path.exists(self.client_class.default_dos) and self.client_class.record is False and len(glob(self.client_class.default_dos+"*")) == 0:
            os.rmdir(self.client_class.default_dos)

    def set_slider_value(self):
        assert len(self.scale_default) == len(self.scales_value)

        for value, scale in zip(self.scale_default, self.scales_value):
            scale.set(value)

    def set_checkbox_value(self):
        assert len(self.box_default) == len(self.bool_checkbox)

        for value, box in zip(self.box_default, self.bool_checkbox):
            box.set(value)

    def reset(self):
        self.set_slider_value()
        self.set_checkbox_value()

        self.get_slider_value()
        self.get_checkbox_value()
