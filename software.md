# Install everything you'll need on the board

The RockPro64 is based on arm64v8 (aarch64) architecture. It is very different from what you have on a x86/64 bit processor and different as well from the arm32 one. Clearly the arm64 is a very powerfull and light consumption energy familly.

For some elements which we'll need like OpenCV, a recompilation will be needed, for some other components like Tensorflow, we will use a precompiled package. And we will of course as many Advanced Package Tool (apt) as possible. They are great as already packaged for the platform. 

## Based image

To install a based image for the RockPro64, go to [this GitHub](https://github.com/ayufan-rock64/linux-build/releases) and install the latest build. The one we are using is based on a Debian Stretch.

Download the image, use [Etcher](https://etcher.io/) to flash it and you're good to go! By default the image embedded a minimum Debian Stretch image with all what is needed to support the RockPro64 including the GPIO.

We strongly recommend to use an eMMC rather than a traditionnal SD card. eMMC are a bit more expensive but are 30 to 50 times fater than SD cards as well as much much much more reliable than SD cards. We started with a standard SD card as the eMMC didn't arrive on time. And to compile OpenCV, it was more than 10 times faster with the eMMC than the SD card!

## Installing the basic components and Python

We'll install basic compenents which are necessary for the next compenent.

```bash
sudo apt-get update
sudo apt-get upgrade

sudo apt-get install -y build-essential cmake pkg-config gfortran wget unzip git
```

Now let's go for the Python familly. Dev versions are needed as OpenCV will require them:

```bash
sudo apt-get install -y python-dev python3-dev

sudo apt-get install -y python-pip python3-pip \
        python-setuptools python3-setuptools \
        python-wheel python3-wheel \
        python-numpy python3-numpy
```

There is currently a bug in Debian Stretch for arm64 platforms. If you upgrade your pip version, it will not work. So we will use both ```pip install``` and ```pip3 install``` commands to make sure we will install in both environement. When the bug will be fixed, updating pip to the very last version will allow to use only pip to install for both versions.

## Building and installing OpenCV

OpenCV is a critical element we'll need to capture the image. We will rebuild it. It has tons of dependencies and a lot needs to be installed on this minimal version of Debian.

```bash
sudo apt-get install -y --no-install-recommends \
        libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
        libxvidcore-dev libx264-dev \
        libgdk-pixbuf2.0-dev \
        libfontconfig1-dev \
        libcairo2-dev \
        libpango1.0-dev \
        libgdk-pixbuf2.0-dev \
        libpango1.0-dev \
        libxft-dev \
        libfreetype6-dev \
        libpng-dev \
        libgtk2.0-dev \
        libgtk-3-dev \
        libatlas-base-dev
```

Once done, we'll download and unzip OpenCV. In our case, the latest available stable version was 3.4.2.

```bash
wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.2.zip \
    && unzip opencv.zip \
    && rm -rf opencv.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.2.zip \
    && unzip opencv_contrib.zip \
    && rm -rf opencv_contrib.zip

pip install tokenizer && pip3 install tokenizer
```

Now it's time to create the build directory

```bash
cd opencv-3.4.2/ \
    && mkdir build \
    && cd build
```

And prepare the compilation.

```bash
cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.4.2/modules \
        -D BUILD_EXAMPLES=ON \
        -D BUILD_WITH_DEBUG_INFO=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_opencv_ts=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=ON \
        ..
```

Compilation time, so time to go for a coffe, tea or equivalent, it will take about 20 minutes on the 6 cores using an eMMC, about 1h on an SD card.

```bash
sudo make -j6
```

We can now finalize the installation and cleaning the install directory

```bash
sudo make install
sudo ldconfig
cd
sudo rm -rf opencv-3.4.2 opencv_contrib-3.4.2
```

## Installing Tensorflow

We will use existing wheels. Please check the GitHub website for the most recent version.

```bash
pip install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.10.0/tensorflow-1.10.0-cp27-none-linux_aarch64.whl

pip3 install https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.10.0/tensorflow-1.10.0-cp35-none-linux_aarch64.whl
```

## Install h5py

HDF5 is necessary for Keras. So installation needs to be done prior installing Keras. Installation needs to be done thru the source code. Explanations [here](http://docs.h5py.org/en/latest/build.html). But it's as simple as the following lines but as always, be patient :-)

```bash
pip install setuptools
pip3 install setuptools

sudo apt-get install libhdf5-dev

git clone https://github.com/h5py/h5py
pip install -v .
pip3 install -v .
```

## Serial, Flask and others

We'll use couple of Python libraries, so we'll need to install them and some are needed for Keras as well. Note that installing scipy is quite long, count at least 1h.

```bash
pip install scipy
pip3 install scipy
pip install pyserial Flask imutils tqdm glob2 scikit-learn pillow
pip3 install pyserial Flask imutils tqdm glob2 scikit-learn pillow
```

Note: if you have any issue, split this command in multiple install, isolate the one generating the issue. Try to check if any dependency is missing and install it, try again.

## Installing Keras

Keras is a framework used to facilitate usage of Tensorflow of CNTK. It needs to be installed from the source code. You'll find it on the official Keras [GitHub](https://github.com/keras-team/keras). Follow the simple installation instructions and you'll be good to go. This installation is very long as well. So be patient!

```bash
git clone https://github.com/keras-team/keras.git

cd keras
git checkout 2.2.2
pip install -v .
pip3 install -v .
```

Note: code is working only with 2.2.2 so far. Compatibility broke on 2.2.3 with a new way of saving and loading the models.

## Setup the serial port

This is to make sure you'll be able to run the serial port command without sudo. It basically gives you all the rights!

```bash
sudo chmod 0777 /dev/ttyS2
```

**Warning**: during boot time, the serial port is used for the kernel.

To deactivate the warning thru the serial port, you can modify the level of warnings from the kernel level via sysctl.with sudo priviledges, edit /etc/sysctl.conf. Specifically, you want to tweak the kernel.printk line.

```bash
# Uncomment the following to stop low-level messages on console
kernel.printk = 0 4 1 3
```

This will disable all the consol logs excet if the system is about to crash. More infor on what those numbers means on [this StackOverflow thread](https://unix.stackexchange.com/questions/13019/description-of-kernel-printk-values)

You can see your current settings with the following command:

```bash
sudo sysctl -a|grep "kernel.printk\b"
kernel.printk = 4   4   1   7
```

Changing the file will require a reboot to take into account. Still keep in mind, the console debug will be activated at boot time. Reason why you'll need to use a simple switch on your Arduino.

## Setup the camera

Make sure you have a camera plugged and give you all the rights.

```bash
sudo chmod 0777 /dev/video0
```

## wifi

There are many ways to get connected to wifi on a Linux. Let's go for a very simple method.

```bash
apt-get install wpasupplicant
```

with sudo priviledges edit the file ```/etc/network/interfaces``` or create a file with the name of yourt wifi card in ```/etc/network/interfaces.d```. To get the wifi card name, just do a ```sudo ifconfig``` and note it, I'll will it NAME. Names can be complicated, write it down.

Then create the followinf entries. Those entries will give a static IP address to the board:

```
auto NAME
iface NAME inet static
  wpa-ssid YOURSSID
  wpa-psk YOURPASSWORD
  address IPADDRESS
  netmask NETWORKMASK
  gateway GATEWAYIPADDRESS
  dns-nameservers DNSIPADDRESS
```