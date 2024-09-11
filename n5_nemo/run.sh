#!/usr/bin/env bash
#--------------------------------------------------------------------#
# Copyright   2022 Reykjavik University
# Author: Carlos Daniel Hernández Mena - carlosm@ru.is
#
#--------------------------------------------------------------------#
this_script="run.sh"
echo " "
echo "+++++++++++++++++++++++++++++++++++++++++++++"
echo "INFO ($this_script): Starting Recipe"
date
echo "+++++++++++++++++++++++++++++++++++++++++++++"
echo " "

#--------------------------------------------------------------------#
# To be run from within the n5_nemo/ directory

echo "-----------------------------------"
echo "Initialization ..."
echo "-----------------------------------"

#--------------------------------------------------------------------#
# Setting up paths and variables
#--------------------------------------------------------------------#

# Note: /asr is the mapped path of the repository inside the Docker
# container, see Readme.md and adapt to your environment if necessary
corpus_root=/asr/data/samromur_21.05/
arpa_lm=/asr/data/6GRAM_ARPA_MODEL.bin

# Destination of the corpus in converted wav version
corpus_wav_path=/asr/data/samromur_21.05/WAV
corpus_wav_name=samromur_21.05_wav

#--------------------------------------------------------------------#
# CONTROL PANEL
#--------------------------------------------------------------------#

# Use deterministic GPU id's as provided by the PCI bus
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

## One GPU
#export CUDA_VISIBLE_DEVICES="1"
#num_gpus=1

##Multiple GPUs, e.g.
#export CUDA_VISIBLE_DEVICES="0,2,1"
num_gpus=2  # this is not needed in newer Lightning versions

nj_train=4    # training worker processes
nj_decode=2   # inference worker processes

# Adjust these 2 variables to just run the stages you want.
# Especially useful for debugging purposes
from_stage=0
to_stage=7

#--------------------------------------------------------------------#
# Exit immediately in case of error.
set -eo pipefail

#--------------------------------------------------------------------#
echo " "
echo "INFO ($this_script): Initialization Done!"
echo " "

#--------------------------------------------------------------------#
# Verifiying that some important files are in place.
#--------------------------------------------------------------------#

[ ! -d "$corpus_root" ] && echo "$0: expected $corpus_root to exist" && exit 1;
[ ! -f "$arpa_lm" ] && echo "$0: expected $arpa_lm to exist" && exit 1;

#--------------------------------------------------------------------#
# Flac to WAV conversion.
#--------------------------------------------------------------------#
current_stage=0
mkdir -p $corpus_wav_path/$corpus_wav_name

if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
    echo "-----------------------------------"
    echo "Stage $current_stage: Converting from flac to wav"
    echo "-----------------------------------"
    
    python3 local/flac2wav.py $corpus_root $corpus_wav_path $corpus_wav_name
    
    echo " "
    echo "INFO ($this_script): Stage $current_stage Done!"
    echo " "
fi

#--------------------------------------------------------------------#
# Flac to WAV conversion.
#--------------------------------------------------------------------#
current_stage=1
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
    echo "-----------------------------------"
    echo "Stage $current_stage: Creating ./data directories and manifests"
    echo "-----------------------------------"
    
    python3 local/create_manifests.py $corpus_wav_path/$corpus_wav_name
    
    mkdir -p data/
    
    mkdir -p data/train
    mkdir -p data/test
    mkdir -p data/dev
    
    mv train_manifest.json data/train
    mv test_manifest.json data/test
    mv dev_manifest.json data/dev
    
    echo " "
    echo "INFO ($this_script): Stage $current_stage Done!"
    echo " "
fi

#--------------------------------------------------------------------#
# Training
#--------------------------------------------------------------------#
current_stage=2
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
    echo "-----------------------------------"
    echo "Stage $current_stage: Training Process"
    echo "-----------------------------------"
    
    # Prepare experiment
    num_epochs=50
    
    exp_name=model_training
    exp_dir=exp/$exp_name
    mkdir -p $exp_dir
    
    cp steps/nemo_training.py $exp_dir

    # Start training
    python3 $exp_dir/nemo_training.py $num_gpus $nj_train $num_epochs $exp_dir \
                                      conf/Config_QuartzNet15x1SEP_Icelandic.yaml \
                                      data/train/train_manifest.json \
                                      data/dev/dev_manifest.json
    
    echo " "
    echo "INFO ($this_script): Stage $current_stage Done!"
    echo " "
fi

#--------------------------------------------------------------------#
# Inference without Language Model
#--------------------------------------------------------------------#
current_stage=3
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
    echo "-----------------------------------"
    echo "Stage $current_stage: Inference without Language Model"
    echo "-----------------------------------"
    
    # Prepare experiment for dev
    exp_name=dev_inference_no_lm
    exp_dir=exp/$exp_name
    mkdir -p $exp_dir
    
    cp steps/inference_no_lm.py $exp_dir
    
    # Start nference process
    python3 $exp_dir/inference_no_lm.py $nj_decode $exp_dir \
                                        conf/Config_QuartzNet15x5_Icelandic.yaml \
                                        data/dev/dev_manifest.json \
                                        exp/model_training

    #----------------------------------------------------------------#
    # Prepare experiment for test
    exp_name=test_inference_no_lm
    exp_dir=exp/$exp_name
    mkdir -p $exp_dir
    
    cp steps/inference_no_lm.py $exp_dir
    
    # Start inference process
    python3 $exp_dir/inference_no_lm.py $nj_decode $exp_dir \
                                        conf/Config_QuartzNet15x5_Icelandic.yaml \
                                        data/test/test_manifest.json \
                                        exp/model_training
    
    echo " "
    echo "INFO ($this_script): Stage $current_stage Done!"
    echo " "
fi

#--------------------------------------------------------------------#
# Inference with Language Model
#--------------------------------------------------------------------#
current_stage=4
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
    echo "-----------------------------------"
    echo "Stage $current_stage: Inference with Language Model"
    echo "-----------------------------------"
    
    # Prepare experiment
    exp_name=dev_inference_lm
    exp_dir=exp/$exp_name
    mkdir -p $exp_dir
    
    cp steps/inference_lm.py $exp_dir

    # Start inference process
    python3 $exp_dir/inference_lm.py $nj_decode $exp_dir $arpa_lm \
                                     data/dev/dev_manifest.json \
                                     exp/model_training
    # Prepare experiment
    exp_name=test_inference_lm
    exp_dir=exp/$exp_name
    mkdir -p $exp_dir
    
    cp steps/inference_lm.py $exp_dir

    # Start inference process
    python3 $exp_dir/inference_lm.py $nj_decode $exp_dir $arpa_lm \
                                     data/test/test_manifest.json \
                                     exp/model_training

    echo " "
    echo "INFO ($this_script): Stage $current_stage Done!"
    echo " "
fi

#--------------------------------------------------------------------#
# Report WER Results
#--------------------------------------------------------------------#
current_stage=5
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
    echo "-----------------------------"
    echo "Stage $current_stage: Printing WER Results"
    echo "-----------------------------"
    
    python3 utils/report_wer_results.py exp
    
    echo " "
    echo "INFO ($this_script): Stage $current_stage Done!"
    echo " "
fi

#--------------------------------------------------------------------#
# Example: Transcribe 1 audio with no Language Model
#--------------------------------------------------------------------#
current_stage=6
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
    echo "-----------------------------"
    echo "Stage $current_stage: Example: Transcribe 1 audio with no Language Model"
    echo "-----------------------------"

    # Prepare experiment
    nemo_model=utils/example_model.ckpt
    audio_in=utils/example_audio.wav
    
    python3 utils/trans_1_no_lm.py $nemo_model $audio_in
    
    echo " "
    echo "INFO ($this_script): Stage $current_stage Done!"
    echo " "
fi

#--------------------------------------------------------------------#
# Example: Transcribe 1 audio with an ARPA Language Model
#--------------------------------------------------------------------#
current_stage=7
if  [ $current_stage -ge $from_stage ] && [ $current_stage -le $to_stage ]; then
    echo "-----------------------------"
    echo "Stage $current_stage: Example: Transcribe 1 audio with an ARPA Language Model"
    echo "-----------------------------"

    # Prepare experiment
    nemo_model=utils/example_model.ckpt
    arpa_lang_model=utils/example_lm.arpa
    audio_in=utils/example_audio.wav
    
    python3 utils/trans_1_with_lm.py $nemo_model $arpa_lang_model $audio_in
    
    echo " "
    echo "INFO ($this_script): Stage $current_stage Done!"
    echo " "
fi

#--------------------------------------------------------------------#
echo " "
echo "+++++++++++++++++++++++++++++++++++++++++++++"
echo "INFO ($this_script): All Stages Done Successfully!"
date
echo "+++++++++++++++++++++++++++++++++++++++++++++"
echo " "
#--------------------------------------------------------------------#

