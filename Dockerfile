FROM tensorflow/tensorflow:1.14.0-gpu-py3 

RUN apt update && apt install -y --allow-unauthenticated wget git vim apt-utils build-essential libosmesa6-dev libglew-dev
RUN apt install -y python3-dev patchelf
RUN apt install -y openmpi-bin openmpi-common openssh-client libopenmpi-dev zlib1g-dev unzip
RUN pip3 install pip --upgrade pip

ARG UID
ARG USER
RUN useradd -u $UID --create-home $USER
USER $USER
WORKDIR /home/$USER

USER root
RUN mkdir -p /home/$USER/.mujoco 

RUN wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /home/$USER/.mujoco \
    && rm mujoco.zip
RUN wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /home/$USER/.mujoco \
    && rm mujoco.zip

# put the liscense under the current folder
COPY mjkey.txt /home/$USER/.mujoco/mjkey.txt

ENV LD_LIBRARY_PATH /home/$USER/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /home/$USER/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}

USER $USER
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

RUN echo 'set editing-mode vi' >> /home/$USER/.inputrc
RUN echo 'set keymap vi' >> /home/$USER/.inputrc

WORKDIR /home/$USER/data
