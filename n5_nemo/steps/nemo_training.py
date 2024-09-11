#-*- coding: utf-8 -*- 
########################################################################
#nemo_training.py

#Author   : Carlos Daniel Hernández Mena
#Date     : December 05th, 2021
#Location : Reykjavík University

#Usage:

#	$ python3 nemo_training.py <num_gpus> <num_jobs> <num_epochs> <experiment_path> <config_file> <training_manifest> <dev_manifest>

#Example:

#    #Start the training process.
#    python3 $exp_dir/nemo_training.py $num_gpus $nj_train $num_epochs $exp_dir \
#                                      conf/Config_QuartzNet15x5_Icelandic.yaml \
#                                      data/train/train_manifest.json \
#                                      data/dev/dev_manifest.json

#Description:

#This script performs a training process in NeMo.
########################################################################
#Imports

import sys
import re
import os

#Importing NeMo Modules
import nemo
import nemo.collections.asr as nemo_asr

########################################################################
# Input Parameters

NUM_GPUS=int(sys.argv[1])
NUM_JOBS=int(sys.argv[2])
NUM_EPOCHS=int(sys.argv[3])
EXPERIMENT_PATH=sys.argv[4]

# Model Architecture
config_path = sys.argv[5]

# Path to our training manifest
train_manifest = sys.argv[6]

# Path to our validation manifest.
# development portion in this case.
dev_manifest = sys.argv[7]

########################################################################
# Reading Model definition
from ruamel.yaml import YAML

yaml = YAML(typ='safe')
with open(config_path) as f:
    model_definition = yaml.load(f)

########################################################################
# Creating trainer

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

tb_logger = pl_loggers.TensorBoardLogger(save_dir=EXPERIMENT_PATH,name="lightning_logs")
trainer = pl.Trainer(accelerator="gpu", devices="auto", max_epochs=NUM_EPOCHS,logger=tb_logger,strategy="ddp")

########################################################################
# Adjusting model parameters
from omegaconf import DictConfig

model_definition['model']['train_ds']['manifest_filepath'] = train_manifest
model_definition['model']['train_ds']['num_workers'] = NUM_JOBS
model_definition['model']['validation_ds']['manifest_filepath'] = dev_manifest
model_definition['model']['validation_ds']['num_workers'] = NUM_JOBS
model_definition['model']['optim']['lr'] = 0.05
model_definition['model']['optim']['weight_decay'] = 0.0001
model_definition['dropout']=0.2
model_definition['repeat']=1

########################################################################
# Adjusting parameters for SpecAugment

# Rectangles placed randomly
model_definition['model']['spec_augment']['rect_masks'] = 5
# Vertical Stripes (Time)
model_definition['model']['spec_augment']['rect_time'] = 120
# Horizontal Stripes (Frequency)
model_definition['model']['spec_augment']['rect_freq'] = 50

########################################################################
# Creating the ASR system which is a NeMo object
nemo_asr_model = nemo_asr.models.EncDecCTCModel(cfg=DictConfig(model_definition['model']), trainer=trainer)

########################################################################
# START TRAINING!
trainer.fit(nemo_asr_model)

########################################################################
# Saving Model

# Calculating current date and time to label the checkpoint
from datetime import datetime
time_now=str(datetime.now())
time_now=time_now.replace(" ","_")

# Creating Checkpoint directory
dir_checkpoints=os.path.join(EXPERIMENT_PATH,"CHECKPOINTS")
name_checkpoints= "model_weights_"+time_now+".ckpt"
if not os.path.exists(dir_checkpoints):
    os.mkdir(dir_checkpoints)

# Save checkpoint
path_checkpoint=os.path.join(dir_checkpoints, name_checkpoints)
nemo_asr_model.save_to(path_checkpoint)

# Write the path to the last checkpoint in an output file
file_path=os.path.join(EXPERIMENT_PATH,"final_model.path")
file_checkpoint=open(file_path,'w')
file_checkpoint.write(path_checkpoint)
file_checkpoint.close()

print("\nINFO: Final Checkpoint in "+path_checkpoint)
print("\nINFO: MODEL SUCCESSFULLY TRAINED!")
