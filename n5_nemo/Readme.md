# Samrómur NeMo Recipe 22.06

## Description

This is a code recipe to demonstrate how to train an Nvidia NeMo QuartzNet ASR model with the ASR corpus [Samrómur 21.05](http://hdl.handle.net/20.500.12537/189) and infer the model in combination with the Icelandic [6-GRAM Language Model 22.06](http://hdl.handle.net/20.500.12537/226). For further background about the ASR QuartzNet model architecture, refer to the [paper](https://arxiv.org/abs/1910.10261) or [Nvidia QuartzNet model description](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/quartznet_pyt).

## Status
![Development/Experimental](https://img.shields.io/badge/Experimental-darkviolet)

This part of the repository is still experimental. A QuartzNet model can be trained successfully, but it's currently not possible to inference the trained model in combination with the 6gram language model. Please follow the state of this [issue](https://github.com/icelandic-lt/samromur-asr/issues/3) for the progress on this topic.

## Installation

There are at least 2 options to train a model: either use a preconfigured Docker container from NVidia via [nvcr.io/nvidia/nemo](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) or follow the standard installation procedure of NVidia NeMo as described [here](https://github.com/NVIDIA/NeMo/tree/v1.23.0#installation). In every case, you should base your installation on a NeMo v1.xx branch instead of the current v2.xx versions.

## Training preparations

### Download Samrómur 21.05 dataset (~8GB size)

The Samrómur 21.05 Dataset can be downloaded from [Clarin.is](http://hdl.handle.net/20.500.12537/189).

Instructions:

Execute from the top of the repository directory:

```bash
mkdir -p data && cd data
curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/189{/samromur_21.05.multi.zip,/samromur_21.05.multi.z01,/samromur_21.05.multi.z02,/samromur_21.05.multi.z03}
unar samromur_21.05.multi.z01
```

Note: the command `unar` can be used to extract a multi-zipfile archive. You need to provide the first file of these archives, and it will automatically unzip all the other files as well.

### Download 6Gram model (~5.4GB size)

```bash
curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/226/6GRAM_ARPA_MODEL.bin
```

### Install prerequisites inside NeMo container

Start the training container via:

```bsah
docker run --gpus all -it nvcr.io/nvidia/nemo:23.10
```
This will start a shell. Here you need to install the following packages that are not part of the prebuilt NeMo container:

```bash
pip install flashlight-text kenlm
```

### Run.sh

The main file for executing all actions is the file [run.sh](run.sh). Single stages like data preparation, training, inference etc. are executed in order, but it's possible to control the execution flow by setting the variables `from_stage` and `to_stage` to just those stages one is interested in. You can then skip previous or later stages, e.g. for debugging purposes. Alternatively, you can also execute each stage separately by executing one of the scripts inside the `local/` or `steps/` directory. Please refer to the file `run.sh` for the necessary command-line parameters.

## Training

Training is demonstrated using a NeMo `v1.xx` based Docker container. Newer `v2.xx` based versions are not compatible with the training and/or inference scripts in this repository.

Change into your `samromur-asr` repository root folder and run a NeMo container via the following command:

```bash
docker run --gpus all -it -v ./:/asr --shm-size=8g -p 8888:8888 -p 6006:6006 \
  --ulimit memlock=-1 --ulimit stack=67108864 --device=/dev/snd nvcr.io/nvidia/nemo:23.10
```
This downloads the container if not already available on your machine and starts a shell session.

Inside the container, navigate to the folder `/asr/n5_nemo` and start training via:

```bash
cd /asr/n5_nemo
bash run.sh
```

You can watch the training progress by starting a Tensorboard inside the container from withing the same directory as you started the training:

```bash
tensorboard --logdir exp/model_training/lightning_logs/
```

If you used the above example to start the NeMo Docker container, the Tensorboard port 6006 is already exported from withing the container. Therefore, you can navigate with the browser to your training machine and connect to port 6006, e.g. `http://localhost:6006`.

By default, the number of epochs for training is set to 50. This needs a few hours depending on the number and performance of your GPU's. You can change the number of epochs and the batch size inside the file [steps/nemo_training.py](steps/nemo_training.py).
The ASR model configuration [conf/Config_QuartzNet15x1SEP_Icelandic.yaml](conf/Config_QuartzNet15x1SEP_Icelandic.yaml) is used per default.

The models are saved inside the `exp/model_training/lightning_logs/version_X/checkpoints` subdirectory. Additionally, the latest checkpoint path is saved inside the file `exp/model_training/final_model.path`. Each checkpoint is a `tar.gz` archive containing the model file itself and its configuration.

## Inference and evaluation

In general, inference and evaluation of the model is run automatically after training has finished. You can change the script `run.sh` to only run inference together with a language model in setting the variables `from_stage` and `to_stage` to 7. By default, this will use the 6gram model downloaded in the training preparation step.

However: the current state of the dependencies for the inference script [steps/inference_with_lm.py](steps/inference_with_lm.py) needs an outdated [CTC_DECODER]() implementation with Python 3.7 version dependency.
