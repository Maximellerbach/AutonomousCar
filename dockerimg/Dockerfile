FROM arm64v8/debian:latest

RUN apt-get update && apt-get upgrade \
    && apt-get install -y --no-install-recommends \                              
        build-essential cmake pkg-config \
        libjpeg-dev libtiff5-dev libpng-dev \
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
        libatlas-base-dev gfortran \
        wget \  
        unzip \                         
        python-dev python3-dev \
        python-pip python3-pip \
        python-setuptools python3-setuptools \
        python-wheel python3-wheel \
        python-numpy python3-numpy \
    && wget -O opencv.zip https://github.com/opencv/opencv/archive/3.4.2.zip \
    && unzip opencv.zip \
    && rm -rf opencv.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/3.4.2.zip \
    && unzip opencv_contrib.zip \
    && rm -rf opencv_contrib.zip \
    #&& pip3 install --upgrade pip \
    && pip install tokenizer \
    && pip3 install tokenizer \              
    && cd opencv-3.4.2/ \
    && mkdir build \
    && cd build \
    && cmake -D CMAKE_BUILD_TYPE=RELEASE \
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
        .. \
    # this is for building on a RPI to use all the cores
    # && make -j4 \ 
    && make \
    && make install \
    && ldconfig \
    && rm -rf ../../opencv-3.4.2 ../../opencv_contrib-3.4.2