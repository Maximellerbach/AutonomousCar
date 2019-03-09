# Control the car with an XBox controler

In order to control the car with an XBox controler, you'll need to install a specific driver and download one py control file. The generic pygame support unfortunately seems not to work correctly on py with xbox controlers. If at some point the issue is solved, the ```JoystickControl.py``` file contains the version with pygame.

The correct version to use is ```RaspberryControl.py```. To install and get the right to run everything without priviledge rights, follow the steps from [https://github.com/FRC4564/Xbox](https://github.com/FRC4564/Xbox):

```bash
sudo apt-get install xboxdrv
# To test installation:
sudo xboxdrv --detach-kernel-driver
# To run code in low priviledges:
sudo usermod -a -G root pi
sudo nano /etc/udev/rules.d/55-permissions-uinput.rules
# add the following line in the file:
# KERNEL=="uinput", MODE="0660", GROUP="root"
# Save and close the file
```

then download and add the code to your solution:

```bash
wget https://raw.githubusercontent.com/FRC4564/Xbox/master/xbox.py
```

Usage is very simple, check the GitHub for more information.