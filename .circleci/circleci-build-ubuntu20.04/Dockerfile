FROM ubuntu:20.04

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
    python3-pip \
    python3.8 \
    python3.8-dev \
    python3.9 \
    python3.9-dev \
    source-extractor \
    psfex \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# ubuntu renames the (awfully named) executable
RUN rm /usr/bin/sex; ln -s /usr/bin/source-extractor /usr/bin/sex

# Pip installs
RUN for x in \
    cython \
    numpy \
    scipy \
    fitsio \
    matplotlib \
    ; do \
    python3.8 -m pip install $x; \
    python3.9 -m pip install $x; \
    done; \
    rm -R /root/.cache/pip

RUN mkdir /src
WORKDIR /src

# Astrometry.net
RUN git clone http://github.com/dstndstn/astrometry.net.git astrometry \
    && cd astrometry \
    && make \
    && make extra \
    && make py PYTHON=python3.8 \
    && make py PYTHON=python3.9 \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.8 \
    && make install INSTALL_DIR=/usr/local PYTHON=python3.9 \
    && make clean

ENV PYTHON=python3
# python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python
ENV PYTHONPATH=/usr/local/lib/python

# # The Tractor
# RUN git clone http://github.com/dstndstn/tractor.git tractor \
#     && cd tractor \
#     && python3.8 setup-cython.py --with-ceres install --home /usr/local \
#     && python3.9 setup-cython.py --with-ceres install --home /usr/local \
#     && make version && cp tractor/version.py /usr/local/lib/python/tractor/ \
#     && rm -R $(find . -name "*.o" -o -name "*.so") \
#     && (cd && PYTHONPATH=/usr/local/lib/python python3 -c "import tractor; print(tractor.__version__)")
