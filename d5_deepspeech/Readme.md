# Samrómur DeepSpeech Recipe 22.06

## Description

This is a code recipe to create an ASR system based on the ASR corpus [Samrómur 21.05](http://hdl.handle.net/20.500.12537/189) and the [DeepSpeech Scorer for Icelandic 22.06](http://hdl.handle.net/20.500.12537/227) using Mozilla's [DeepSpeech recognizer](https://github.com/mozilla/DeepSpeech).

## Status
![Development/Experimental](https://img.shields.io/badge/Experimental-darkviolet)

This recipe depends on an unmaintained and abandonded project Mozilla DeepSpeech. Although following the installation steps succeeds, the training process itself within the file [run.sh](run.sh) cannot be reproduced anymore in 2024 and results in syntax errors because of probably incompatible 3rdparty dependencies. 

## Training preparations

### Download Samrómur 21.05 dataset

You can use the instructions inside the [n5_nemo](../n5_nemo/Readme.md#download-samrómur-2105-dataset-8gb-size) recipe to download and unzip the dataset locally.

### Download DeepSpeech Scorer for Icelandic 22.06 (~1GB size)

From the top of the repository, execute the following commands:

```bash
mkdir -p data/deepspeech_scorer && pushd data/deepspeech_scorer
curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/227{/10_trials_optim_kenlm.scorer,/Readme.txt}
pushd
```

### Install prerequisites

Install in your system:

	* sox
	* libsndfile1
	* ffmpeg

Create Conda environment for DeepSpeech and install dependencies.

```bash
conda create -n deepspeech python=3.7 anaconda
conda activate deepspeech
conda install cudatoolkit=10.1
conda install numpy=1.18 scipy
conda install -c conda-forge git-lfs
conda install -c conda-forge nvidia-apex
conda install pycodestyle=2.7.0 pyflakes=2.3.1
conda install lxml requests python-dateutil pytz
```

Install DeepSpeech

```bash
pip install deepspeech-gpu==0.10.0-alpha.3 ds-ctcdecoder==0.10.0-alpha.3
pip install tensorflow-gpu==1.15.4
git clone https://github.com/mozilla/DeepSpeech.git
cd DeepSpeech
python setup.py install
```

## Train

Verify that all paths are correctly setup and contain the necessary data and model inside the file `run.sh`.<br>
Execute from withing this recipe's subdirectories the following command:

```bash
bash run.sh
```
