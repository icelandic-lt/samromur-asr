<!-- omit in toc -->
# LVL Samrómur ASR

![Version](https://img.shields.io/badge/Version-M9_l2-darkviolet)
![Python](https://img.shields.io/badge/python-3.8-blue?logo=python&logoColor=white)
![Python](https://img.shields.io/badge/python-3.9-blue?logo=python&logoColor=white)
![CI Status](https://img.shields.io/badge/CI-[unavailable]-red)
![Docker](https://img.shields.io/badge/Docker-[Kaldi:available]-green)

Creation of an Automatic Speech Recognition (ASR) system for the Samrómur speech corpus using
[Kaldi](http://kaldi-asr.org/doc/about.html), [NeMo](https://github.com/NVIDIA/NeMo) or [Mozilla DeepSpeech](https://github.com/mozilla/DeepSpeech).

## Overview

This repository has been created by the [Language and Voice Lab](https://lvl.ru.is/) at Reykjavík University and is
part of the [Icelandic Language Technology Programme](https://github.com/icelandic-lt/icelandic-lt).

- **Category:** [ASR](https://github.com/icelandic-lt/icelandic-lt/blob/main/doc/asr.md)
- **Domain:** Server
- **Languages:** C++, Python, Shell
- **Language Version/Dialect:**
  - Python: 3.7+
  - C++: C++14 (Kaldi)
- **Audience**: Developers, Researchers
- **Origins:** [samromur-asr](https://github.com/cadia-lvl/samromur-asr), [Samrómur NeMo Recipe 22.06](http://hdl.handle.net/20.500.12537/228), [Samrómur DeepSpeech Recipe 22.06](http://hdl.handle.net/20.500.12537/229)

## Status
![Development/Experimental](https://img.shields.io/badge/Experimental-darkviolet)

For the Kaldi recipes inside the directories `s5_XXX` it is assumed that Kaldi is installed on the running system itself. In 2024 however, one should use the official [Docker](https://hub.docker.com/r/kaldiasr/kaldi/tags) images for Kaldi or build your own container images as described [here](https://github.com/kaldi-asr/kaldi/blob/master/docker/README.md).<br>
We have provided a Kaldi Docker image that can be built on `Ubuntu 22.04` or `Ubuntu 20.04` [here](docker/Dockerfile). Please refer to the [README](docker/README.md) for how to use it.
Most of the documentation in this repository for the Kaldi-based recipes refers to the setup/installation of the ASR system on the Terra Linux cluster at LVL. Therefore, these scripts and documentations are tailored for that particular environment and cannot be used simply as-is.<br>
Please follow [this issue](https://github.com/icelandic-lt/samromur-asr/issues/2) for an update on the progress of making the scripts more general.

The Nvidia NeMo recipe in the subdirectory [n5_nemo/](n5_nemo/) uses a different framework and a different model architecture for ASR. It's experimental, as the inferencing side uses an outdated dependency. See issue [#3](https://github.com/icelandic-lt/samromur-asr/issues/3) for an explanation.

The Mozilla DeepSpeech recipe in the subdirectory [d5_deepspeech/](d5_deepspeech/) uses the Mozilla DeepSpeech framework, which is nowadays unmaintained and unfortunately the models cannot be built anymore, at least with the provided instructions.

Although the last 2 recipes are currently not usable as-is, they might give some valuable insights for future research and development.

## System Requirements
- Operating System: Linux
- Training: one or multiple GPU's with CUDA support

## Description

<img src="https://user-images.githubusercontent.com/9976294/84160937-4042f880-aa5e-11ea-8341-9f1963e0e84e.png" alt="Cover Image" align="center"/>

<p align="center"><i>
  NOTE! This is a project in development.
  
  Automatic Speech Recognition (ASR) system for the Samrómur speech corpus using <a href="http://kaldi-asr.org/doc/">Kaldi</a><br/>
  Center for Analysis and Design of Intelligent Agents, Language and Voice Lab <br/>
  <a href="https://ru.is">Reykjavik University</a>
  
  This project is a research project on ASR creation. It does not contain trained ASR models or scripts on how to perform speech recognition using the models trained with the recipes provided here. [The Althingi recipe](https://github.com/cadia-lvl/kaldi/tree/master/egs/althingi) provides example scripts for how to run a Kaldi trained speech recognizer.
  
  We plan to have the recipes ready by October 2021 and create a Docker with the trained models.
</i></p>

<!-- omit in toc -->
## Table of Contents

<details>
<summary>Click to expand</summary>

- [1. Introduction](#1-introduction)
- [2. The Dataset](#2-the-dataset)
- [3. Setup](#3-setup)
- [4. Computing Requirements](#4-computing)
- [5. License](#5-license)
- [6. References](#6-references)
- [7. Contributing](#7-contributing)
- [8. Contributors](#8-contributors)

</details>

## 1. Introduction

Samrómur ASR is a collection of scripts, recipes, and tutorials for training an ASR using the [Kaldi-ASR](http://kaldi-asr.org/) toolkit.

[s5_base](s5_base/) is the regular ASR recipe. It's meant to be the foundation of our Samrómur recipes.
[s5_subwords](s5_subwords/) is a subword ASR recipe.
[s5_children](s5_children/) is a standard ASR recipe adapted towards children speech.

[documentation](documentation/) contains information on data preparation for Kaldi and setup scripts
[preprocessing](preprocessing/) contains external tools for preprocessing and data preprocessing examples

## 2. The Dataset

The Samrómur speech corpus is an open (CC-BY 4 licence) and accessible database of voices that everyone is free to use when developing software in Icelandic.
The database consists of sentences and audio clips from the reading of those sentences as well as metadata about the speakers. Each entry in the database contains a WAVE audio clip and the corresponding text file.

The Samrómur speech corpus is available for download at [OpenSLR](https://www.openslr.org/112/), [CLARIN-IS](http://hdl.handle.net/20.500.12537/189) and [LDC](https://doi.org/10.35111/thx3-f170)

For more information about the dataset visit [https://samromur.is/gagnasafn](https://samromur.is/gagnasafn).

You can use the instructions inside the [n5_nemo](n5_nemo/Readme.md#download-samrómur-2105-dataset-8gb-size) recipe to download and unzip the dataset locally.

## 3. Setup

First clone this repository and also the submodules via:

```bash
git clone --recurse-submodules https://github.com/icelandic-lt/samromur-asr.git
```

You can use these guides for reference even if you do not use Terra (a cloud cluster at LVL).

- [Setup Guide for Kaldi-ASR](documentation/setup_kaldi.md)
- [Setup Guide for Samrómur-ASR](documentation/setup_samromur-asr.md)

See [training instructions](n5_nemo/Readme.md) of the NeMo recipe for training a QuartzNet ASR model.

See [training instructions in s5_children](s5_children/README.md) of how to train a Kaldi chain TDNN ASR model on both the base Samrómur corpus and Samrómur Children.

## 4. Computing Requirements

This project is developed on a computing cluster with 112 CPUs and 10 GPUs (2 GeForce GTX Titan X, 4 GeForce GTX 1080 Ti, 4 GeForce RTX 2080 Ti). All of that is definitely not needed but the neural network acoustic model training scripts are intended to be used with GPUs. No GPUs are needed to use the trained models.

**To do: Add training time info.**

## 5. License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 6. References
- [Samrómur](https://samromur.is/)
- [Language and Voice Lab](https://lvl.ru.is/)
- [Reykjavik University](https://www.ru.is/)
- [Kaldi-ASR](http://kaldi-asr.org/)
- [NeMo](https://github.com/NVIDIA/NeMo)

## 7. Contributing

Pull requests are welcome. Please open an issue first to discuss what you would like to change.

## 8 Acknowledgements

This project was funded by the Language Technology Programme for Icelandic 2019-2023. The programme, which is managed and coordinated by [Almannarómur](https://almannaromur.is/), was funded by the Icelandic Ministry of Education, Science and Culture.
