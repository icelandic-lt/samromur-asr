# Building the Kaldi Docker image

This repository contains a Dockerfile for building a container image with Kaldi installed. It can be used to train and
run ASR models using Kaldi via Nvidia GPUs. The image is based on the `nvidia/cuda:12.2.0` image and installs Kaldi
from source. It has been tested on Ubuntu 20.04.

Use the following commands to build the image:

```bash
cd /path/to/samromur-asr-repo/docker
docker build --tag samromur-asr/kaldi:gpu-latest .
```

# Use the built image

```bash
mkdir -p data/
docker run -v /path/to/samromur-asr-repo:/samromur-asr -v data:/data --gpus all -it samromur-asr/kaldi:gpu-latest bash
```

# How to continue ?

All descriptions inside this repository don't cover the use of Docker and you will have to adapt most scripts
accordingly. A typical workflow is to mount the repository itself inside the container as well as a data directory
you want to use. This is shown in the example above. The repository is mounted to `/samromur-asr` and the data directory
is mounted to `/data`.

Then you can run the scripts as you would on a normal Linux system. You can also edit the scripts in the repository with
your normal IDE of choice while you are running the container and all changes are reflected immediately.
