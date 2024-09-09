FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
LABEL maintainer="williamhilton.works@gmail.com"

# Note: This Dockerfile is a copy of the Kaldi official Dockerfile as of commit #ebf6245

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        make \
        automake \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        subversion \
        python2.7 \
        python3 \
        zlib1g-dev \
        ca-certificates \
        gfortran \
        patch \
        ffmpeg \
        vim && \
    apt-get update && \
    apt-get install -y --no-install-recommends\
        software-properties-common && \
    apt-add-repository multiverse && \
    apt-get update && \
    yes | DEBIAN_FRONTEND=noninteractive apt-get install -yqq --no-install-recommends\
        intel-mkl && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python2.7 /usr/bin/python

RUN git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi && \
    cd /opt/kaldi/tools && \
    make -j $(nproc) && \
    cd /opt/kaldi/src && \
    ./configure --shared --use-cuda=yes && \
    make depend -j $(nproc) && \
    make -j $(nproc) && \
    find /opt/kaldi  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
    rm -rf /opt/kaldi/.git

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yqq --no-install-recommends\
        python3-pip \
        libboost-system-dev \
        libboost-thread-dev \
        libboost-program-options-dev \
        libboost-test-dev \
        libeigen3-dev \
        zlib1g-dev \
        libbz2-dev \
        liblzma-dev \
        cmake && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/kpu/kenlm/archive/f6c947dc943859e265fabce886232205d0fb2b37.zip -O kenlm.zip && \
    unzip kenlm.zip && \
    mv kenlm-f6c947dc943859e265fabce886232205d0fb2b37 /opt/kenlm && \
    cd /opt/kenlm && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j $(nproc) lmplz

RUN pip install \
    pandas \
    sequitur-g2p

# This script is currently broken on platforms that do not have the en_US.UTF-8
# locale. See: https://github.com/kaldi-asr/kaldi/pull/4934
RUN sed -i 's/if locale -a | grep "C.UTF-8"/if locale -a | grep "C.utf8"/' \
    /opt/kaldi/egs/wsj/s5/utils/validate_data_dir.sh

WORKDIR /opt/kaldi/

ENV KALDI_ROOT=/opt/kaldi
ENV PATH="/opt/kenlm/build/bin:$PATH"
ENV LC_ALL=C.UTF-8
