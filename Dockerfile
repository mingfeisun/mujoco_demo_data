FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-devel

RUN apt update && apt install -y --allow-unauthenticated wget git vim apt-utils build-essential libosmesa6-dev libglew-dev
RUN apt-get update && apt-get install -y --allow-unauthenticated libopenmpi-dev patchelf unzip
RUN python3 -m pip install pip --upgrade pip


ARG UID
RUN useradd -u $UID --create-home mingfei
USER mingfei
WORKDIR /home/mingfei

RUN mkdir -p /home/mingfei/.mujoco 

RUN wget http://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /home/mingfei/.mujoco \
    && rm mujoco.zip
RUN wget http://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /home/mingfei/.mujoco \
    && rm mujoco.zip

# put the liscense under the current folder
COPY mjkey.txt /home/mingfei/.mujoco/mjkey.txt

ENV LD_LIBRARY_PATH /home/mingfei/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /home/mingfei/.mujoco/mjpro200_linux/bin:${LD_LIBRARY_PATH}

# install all requirements
USER root
COPY ./spinningup ./spinningup
RUN python3 -m pip install -e ./spinningup

USER mingfei
COPY requirements.txt requirements.txt
RUN python3 -m pip install -r requirements.txt

RUN echo 'set editing-mode vi' >> /home/mingfei/.inputrc
RUN echo 'set keymap vi' >> /home/mingfei/.inputrc

WORKDIR /home/mingfei/data
