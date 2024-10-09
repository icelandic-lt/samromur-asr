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

## Inference

To use the trained model for inference you can e.g. run a [Tiro Speech
Core](https://github.com/icelandic-lt/tiro-speech-core) server by packaging the
model using
[prepare_chain_dist.sh](https://github.com/icelandic-lt/tiro-speech-core/blob/master/tools/models/prepare_chain_dist.sh).

Inference on single audio files can also be done using the Kaldi CLI
tools. First you have to prepare an "online decoding" directory:

``` shell
docker run -v $PWD/..:/samromur-asr \
           -v BASE_PATH_FOR_BOTH_SAMROMUR_DATASETS:/datasets \
           -w /samromur-asr/s5_children \
           -it samromur-asr/kaldi:gpu-latest \
           --gpus all \
           steps/online/nnet3/prepare_online_decoding.sh \
             data/lang exp/nnet3/extractor exp/chain/tdnn_7n_sp exp/chain/tdnn_7n_sp_online
```

And then you decode an audio file in `BASE_PATH_FOR_AUDIO`:

``` shell
docker run -v $PWD/..:/samromur-asr \
           -v BASE_PATH_FOR_AUDIO:/audio \
           -v BASE_PATH_FOR_BOTH_SAMROMUR_DATASETS:/datasets \
           -w /samromur-asr/s5_children \
           -it samromur-asr/kaldi:gpu-latest \
           --gpus all \
           bash -c '\
           . path.sh && \
           online2-wav-nnet3-latgen-faster \
             --acoustic-scale=1.0 \
             --online=false \
             --mfcc-config=exp/chain/tdnn_7n_sp_online/conf/mfcc.conf \
             --ivector-extraction-config=exp/chain/tdnn_7n_sp_online/conf/ivector_extractor.conf \
             exp/chain/tdnn_7n_sp/final.mdl \
             exp/chain/tdnn_7n_sp/graph/HCLG.fst \
             ark:<(echo "speaker1 utterance1") \
             scp:<(echo "utterance1 sox /audio/example.wav -esigned -c1 -r16k -b16 -twav - |") \
             ark:- \
             | lattice-best-path \
                 --acoustic-scale=1.0 \
                 ark:- \
                 "ark,t:|utils/int2sym.pl -f 2- data/lang/words.txt"'
```

Note that the Kaldi CLI tools operate on [tables of
data](http://kaldi-asr.org/doc/io_tut.html#io_tut_table), so `ark:<(echo
"speaker1 utterance1")` creates a temporary table that maps the audio
`utterance1` to a speaker called `speaker1` and `scp:<(echo "utterance1 sox
/audio/example.wav -esigned -c1 -r16k -b16 -twav - |")` is a temporary table that for
the audio `utterance1` invokes a `sox` command to convert the input audio file
to the correct format.

## Datasets

The Samrómur and Samrómur Children corpora is available for download at
CLARIN-IS: [Samrómur 21.05](http://hdl.handle.net/20.500.12537/189) and
[Samrómur Children 21.09](http://hdl.handle.net/20.500.12537/185).
