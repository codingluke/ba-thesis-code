# Docker container that spins up a Jupyter notebook server
# with CUDA accelerated Theano support. Assumes the host
# system has CUDA drivers installed that match the version below.
#
# Launch with a comman line similar to the following:
#
# docker run -it \
#   --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0
#   -p 8888:8888
#   -v /[where notebooks are on your local machine]:/notebooks
#   -v /[optional data directory that the notebooks process]:/data
#   ipynbsrv
#
#   NOTE: Lunching directly into jupyter via the CMD statement can lead to
#   the ipython kernel starting and stopping. If you instead launch docker
#   into /bin/bash and then run 'jupyter notebook' by hand it seems to work.
#   This with Docker 1.8 - YMMV
#
FROM ubuntu:14.04
FROM python:2.7.10

RUN apt-get update -qq

#
# CUDA: See https://hub.docker.com/r/kaixhin/cuda/~/dockerfile/
# Would have rather done a FROM but the above included ubuntu so I can't upgrade python
#

# Install wget and build-essential
RUN apt-get install -yq build-essential wget module-init-tools

# Change to the /tmp directory
RUN cd /tmp
# Download run file - add so we don't download every time we try and build
#RUN wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run
RUN wget http://developer.download.nvidia.com/compute/cuda/6_5/rel/installers/cuda_6.5.14_linux_64.run
# Make the run file executable and extract
# ADD cuda_7.5.18_linux.run .
RUN chmod +x cuda_*_linux.run
RUN ./cuda_*_linux.run -extract=`pwd`
# Install CUDA drivers (silent, no kernel)
RUN ./NVIDIA-Linux-x86_64-*.run -s --no-kernel-module
# Install toolkit (silent)  
RUN ./cuda-linux64-rel-*.run -noprompt
# Clean up
RUN rm -rf /tmp/*
# Add CUDA to path
ENV PATH=/usr/local/cuda/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#
# Theano & Keras
#
RUN DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
	libfreetype6-dev libpng-dev \
  gfortran libblas-dev liblapack-dev libatlas-base-dev
RUN pip install -U pip
RUN pip install git+git://github.com/Theano/Theano.git@rel-0.7#egg=Theano
ENV THEANO_FLAGS='floatX=float32,device=gpu,exception_verbosity=high,optimizer=fast_compile'

RUN pip install git+git://github.com/fchollet/keras.git@0.2.0#egg=Keras

#
# Jupyter Notebook
#
RUN apt-get install -yq libzmq-dev
RUN pip install jupyter
RUN mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.server_extensions.append('ipyparallel.nbextension')" >> /root/.jupyter/jupyter_notebook_config.py

#
# Other libraries and python packages
# Explicitly add requirements.txt so its cached and pip only runs
# if it changes. See https://github.com/docker/docker/pull/2809
RUN apt-get install -y libsamplerate0-dev
ADD ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Finally clean up any side affects from apt-get
RUN apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

VOLUME /notebooks
VOLUME /data
WORKDIR /notebooks

EXPOSE 8888

CMD ["jupyter", "notebook"]
