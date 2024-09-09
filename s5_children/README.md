# TDNN recipe for Samrómur 21.05 and Samrómur Children 21.09

The file [./run.sh](./run.sh) contains a recipe for training a Kaldi chain TDNN
model from the published Samrómur and Samrómur Children datasets. The recipe
assumes you have already downloaded both datasets, have available training texts
for a language model and a base pronunciation dictionary that uses the same
phoneme set as the Sequitur G2P model in
[../preprocessing/g2p/](../preprocessing/g2p/), e.g. the [Icelandic
Pronunciation Dictionary for Language Technology
22.01](http://hdl.handle.net/20.500.12537/331) or [the one in this
repo](../preprocessing/g2p/transcribed). Using an empty file for the
pronunciation dictionary will also work. In that case the G2P model is used to
generate a lexicon for the words in the training set.

## Training a model

Assuming you have already built the [Docker image in this
repository](../docker/Dockerfile) you can run the recipe in a container using
the following command (`--gpus all` can be left out if you do not have GPUs, but
training without GPUs will take a very long time):

``` shell
docker run -v $PWD/..:/samromur-asr \
           -v BASE_PATH_FOR_BOTH_SAMROMUR_DATASETS:/datasets \
           -w /samromur-asr/s5_children \
           -it samromur-asr/kaldi:gpu-latest \
           --gpus all \
           ./run.sh --samromur-root /datasets/samromur_21.05 \
                    --samromur-teen /datasets/samromur_children_21.09 \
                    --lm-train lm_training_text.txt \
                    --prondict-orig /samromur-asr/preprocessing/g2p/transcribed
```

The script will incrementally train a few GMM models (used for alignment) and
evaluate their performance on the test sets using these before training the TDNN
model. The decoding stage uses a language model trained by this script with
[KenLM](https://kheafield.com/code/kenlm/) on the text given by the `--lm-train`
argument to `run.sh`.

If the recipe finishes successfully you'll end up with a Kaldi chain model in
`exp/chain/tdnn_7n_sp`.


## Datasets

The Samrómur and Samrómur Children corpora is available for download at
CLARIN-IS: [Samrómur 21.05](http://hdl.handle.net/20.500.12537/189) and
[Samrómur Children 21.09](http://hdl.handle.net/20.500.12537/185).

