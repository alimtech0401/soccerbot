FROM utrarobosoccer/noetic as dependencies
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
WORKDIR /root/src
RUN apt update && rosdep update --rosdistro noetic
ADD . .
RUN rosdep install --from-paths . --ignore-src -r -s  | grep 'apt-get install' | awk '{print $3}' | sort  >  /tmp/catkin_install_list
RUN mv requirements.txt /tmp/requirements.txt
WORKDIR /root/dependencies

FROM utrarobosoccer/noetic as builder
SHELL ["/bin/bash", "-c"]

# Install dependencies
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN apt update && \
    apt install -q -y software-properties-common && \
    add-apt-repository ppa:apt-fast/stable -y && \
    echo debconf apt-fast/maxdownloads string 16 | debconf-set-selections && \
    echo debconf apt-fast/dlflag boolean true | debconf-set-selections && \
    echo debconf apt-fast/aptmanager string apt-get | debconf-set-selections && \
    apt install -q -y apt-fast && \
    apt clean
RUN apt update && apt-fast install -y \
    screen \
    vim \
    python3-pip \
    python3-catkin-tools \
    python3-protobuf \
    protobuf-compiler \
    libprotobuf-dev \
    libjpeg9-dev \
    wget \
    ccache \
    dirmngr \
    gnupg2 \
    lsb-release \
    net-tools \
    iputils-ping \
    apt-utils \
    software-properties-common \
    sudo \
    ros-noetic-robot-state-publisher \
    curl

COPY --from=dependencies /tmp/requirements.txt /tmp/requirements.txt
RUN pip install --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --trusted-host=pytorch.org --trusted-host=download.pytorch.org --trusted-host=files.pypi.org --trusted-host=files.pytorch.org \
    -r /tmp/requirements.txt --find-links https://download.pytorch.org/whl/cu113/torch_stable.html

COPY --from=dependencies /tmp/catkin_install_list /tmp/catkin_install_list
RUN apt-get update && apt-fast install -y $(cat  /tmp/catkin_install_list)

# Build
WORKDIR /root/catkin_ws
COPY --from=dependencies /root/src src/soccerbot
RUN source /opt/ros/noetic/setup.bash && catkin config --cmake-args -DCMAKE_BUILD_TYPE=Debug
RUN source /opt/ros/noetic/setup.bash && catkin build soccerbot
RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc
RUN pip install PyQt6
RUN sudo apt install -y xvfb
