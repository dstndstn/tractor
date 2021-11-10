FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && apt install -y apt-utils && \
    apt install -y --no-install-recommends \
    build-essential \
    make \
    gcc \
    g++ \
    gfortran \
    git \
    ssh \
    file \
    pkg-config \
    wget \
    curl \
    swig \
    netpbm \
    wcslib-dev \
    wcslib-tools \
    zlib1g-dev \
    libbz2-dev \
    libcairo2-dev \
    libcfitsio-dev \
    libcfitsio-bin \
    libgsl-dev \
    libjpeg-dev \
    libnetpbm10-dev \
    libpng-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libceres-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-pil \
    python3-tk \
    python3-setuptools \
    python3-wheel \
    python3.7 \
    python3.7-dev \
    python3.8 \
    python3.8-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# multiprocessing.pool from official python 3.6.8 release
# The python 3.6.7 version shipped in current Ubuntu 18.04 breaks the
# Astrometry.net "timingpool" class.  This changed from 3.6.6 to 3.6.7 and
# then was reverted in 3.6.8.
COPY pool.py /usr/lib/python3.6/multiprocessing/pool.py

# Python related stuff
RUN echo "../site-packages" > /usr/local/lib/python3.6/dist-packages/site-packages.pth

# Pip installs
RUN for x in \
    cython \
    numpy \
    scipy \
    fitsio \
    matplotlib \
    ; do \
    python3.6 -m pip install $x; \
    python3.7 -m pip install $x; \
    python3.8 -m pip install $x; \
    done; \
    rm -R /root/.cache/pip

RUN python3.7 -m pip install --no-cache-dir coverage coveralls

RUN mkdir /src
WORKDIR /src

# Astrometry.net
RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make \
    && make extra \
    && make py PYTHON=python3.6 \
    && make py PYTHON=python3.7 \
    && make py PYTHON=python3.8 \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.6 \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.7 \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.8 \
    && make clean

ENV PYTHON=python3
# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHONPATH=/usr/local/lib/python

# # The Tractor
# RUN git clone http://github.com/dstndstn/tractor.git tractor \
#     && cd tractor \
#     && python3.6 setup-cython.py --with-ceres install --home /usr/local \
#     && python3.7 setup-cython.py --with-ceres install --home /usr/local \
#     && python3.8 setup-cython.py --with-ceres install --home /usr/local \
#     && make version && cp tractor/version.py /usr/local/lib/python/tractor/ \
#     && rm -R $(find . -name "*.o" -o -name "*.so") \
#     && (cd && PYTHONPATH=/usr/local/lib/python python3 -c "import tractor; print(tractor.__version__)") \
#     && echo 2

# ENV PYTHONPATH=/usr/local/lib/python:/src/unwise_psf/py:/src/legacypipe/py
