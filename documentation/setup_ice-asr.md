<!-- omit in toc -->
# Setup LVL ICE-ASR

<!-- omit in toc -->
## Introduction

We will be using [LVL ICE-ASR](https://github.com/cadia-lvl/ice-asr) as the foundation for our Kaldi recipe.
Since ICE-ASR hasn't been updated since 2017, it won't run succesfully on top of the latest version of Málrómur on [Málföng.is](http://malfong.is).

Please follow the instructions below for the latest setup:

<!-- omit in toc -->
## Table of Contents

<details>
<summary>Click to expand</summary>

- [1. Activate your Conda environment](#1-activate-your-conda-environment)
- [2. Download ICE-ASR](#2-download-ice-asr)
- [3. Download General Icelandic Pronunciation Dictionary For ASR](#3-download-general-icelandic-pronunciation-dictionary-for-asr)
  - [3.1. Extract the prondict_sr.zip](#31-extract-the-prondict_srzip)
  - [3.2. Creating the "lang" directory](#32-creating-the-lang-directory)
- [4. Download the Malromur Corpus (ísl. Málrómur)](#4-download-the-malromur-corpus-ísl-málrómur)
- [5. Extract the malromur.zip](#5-extract-the-malromurzip)
- [6. Select the data to be used](#6-select-the-data-to-be-used)
- [7. Create a wav_info.txt file for the selected data](#7-create-a-wav_infotxt-file-for-the-selected-data)
- [8. Train an acoustic model with Málrómur](#8-train-an-acoustic-model-with-málrómur)
- [9. Run clean data directories](#9-run-clean-data-directories)
- [10. Feature extraction](#10-feature-extraction)
- [11. Language Model](#11-language-model)
- [12. Train an LDA+MLLT acoustic model](#12-train-an-ldamllt-acoustic-model)
- [13. Normalize Text](#13-normalize-text)

</details>

## 1. Activate your Conda environment

Make sure you have followed our [Setup Kaldi-ASR](setup_kaldi.md) guide, installed [Conda](https://docs.conda.io/projects/conda/en/latest/index.html) and have created a conda environment with the name `kaldi-env`.

```console
USER@terra:~$ conda activate kaldi-env
```

## 2. Download ICE-ASR

```console
(kaldi-env) USER@terra:~$ git clone https://github.com/cadia-lvl/ice-asr.git
```

## 3. Download General Icelandic Pronunciation Dictionary For ASR

If you are using Terra, the Icelandic Pronunciation Dictionary is available at `/data/asr/prondict_sr/v1/frambordabok_asr_v1.txt`

If you are not using Terra, retrieve the General Pronunciation Dictionary For
ASR at [Clarin.is](http://hdl.handle.net/20.500.12537/199).


```console
(kaldi-env) USER@terra:~$ wget -P /home/USER -O prondict_sr.zip <URL>
```

<!-- Example -->
<!-- wget -P /home/egillanton -O prondict_sr.zip http://www.malfong.is/tmp/28AFC730-D7D3-C902-B45B-D9C78812058D.zip -->

### 3.1. Extract the prondict_sr.zip

```console
(kaldi-env) USER@terra:~$ mkdir prondict_sr; unzip prondict_sr.zip -d prondict_sr; rm prondict_sr.zip
```

### 3.2. Creating the "lang" directory

Let's make a copy of `prondict_sr/lang_v1` and name it `lang` under `ice-kaldi/s5/data`:

```console
(kaldi-env) USER@terra:~$ cp -rf ~/prondict_sr/lang_v1/ ~/kaldi/egs/ice-kaldi/s5/data/lang
```

## 4. Download the Malromur Corpus (ísl. Málrómur)

Retrieve the Malromur corpus at <http://www.malfong.is/index.php?dlid=64>.

You will receive a URL in your email to download the `malromur.zip` file.

```console
(kaldi-env) USER@terra:~$ wget -P /home/USER -O malromur.zip <URL>
```

<!-- Example -->
<!-- wget -P /home/egillanton -O malromur.zip http://www.malfong.is/tmp/BC1B8748-1C10-FF86-462F-831E28A0035D.zip -->

## 5. Extract the malromur.zip

```console
(kaldi-env) USER@terra:~$ mkdir malromur; unzip malromur.zip -d malromur; rm malromur.zip
(kaldi-env) USER@terra:~$ cd malromur/
```

## 6. Select the data to be used

We will include all the data

```console
(kaldi-env) USER@terra:~/malromur$ mkdir wav
(kaldi-env) USER@terra:~/malromur$ find -maxdepth 2 -name '*.wav' -exec cp -n -t ./wav/ {} +
```

 `mkdir wav` - create a `./wav` dirrectory.

`find -maxdepth 2 -name '*.wav'` - find all the `.wav` files from all the subdirectories.

`-exec ... -t {} +` - execute the following command to each line,  `-t` is used as the placeholder for the line value.

`cp -n <source> <destination>` - copy the source file to destination, and if the file exists ignore with the `-n` flag.

## 7. Create a wav_info.txt file for the selected data

We are using all the `.wav` files.

We will need to do two things to the `info.txt` file so the Ice-ASR scripts will work:

1. Replace comma with tab
2. Add the `.wav` file extension to the file name

```console
(kaldi-env) USER@terra:~/malromur$ sed 's/,/\t/g' info.txt | awk -F'\t' -vOFS='\t' '{ $1 = $1 ".wav" }1' > wav_info.txt
```

`sed 's/,/\t/g'` - it is necessary to replace `,` with `\t` for `local/malromur_prep_data.sh` to run successfully in the following step.

`awk -F'\t' -vOFS='\t' '{ $1 = $9 ".wav" ";" }1'`:

- `-F'\t'` tells it to use tab-separated fields.

- `-vOFS='\t'` tells it to use tabs in the output, too.

- The actual body of it is the last argument: It's a little program that says
for every line to change the value of $1 (the first field) to the concatenation
of its original value and ".wav"

- The final `1` is to tell `awk` to print the new line out even though we did something to it.

If you are using only a specific subdirectory, use this instead:

Example, if I only want to use the data fram the correct sub directory:

```console
(kaldi-env) USER@terra:~/malromur$ mkdir wav
(kaldi-env) USER@terra:~/malromur$ find ./correct -maxdepth 1 -name '*.wav' -exec cp -n -t ./wav/ {} +
(kaldi-env) USER@terra:~/malromur$ cat info.txt | grep ",correct$" | sed 's/,/\t/g' >> wav_info.txt
```

`grep ",<<pattern>>$"` - returns only the lines that end with the following pattern.

`>>` - you are appending to the file instead of overwriting it; that way, you can run the comand for each of your selected directories.

## 8. Train an acoustic model with Málrómur

Run `malromur_prep_data.sh` on the whole corpus and then divide the generated data randomly:

```console
(kaldi-env) USER@terra:~/malromur$ cd ~/kaldi/egs/ice-kaldi/s5
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ local/malromur_prep_data.sh ~/malromur/wav ~/malromur/wav_info.txt data/all/
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ utils/subset_data_dir_tr_cv.sh --cv-utt-percent 10 data/{all,training_data,test_data}
```

The prepared data is now in data/all and after the subset command the prepared
files are divided such that 10% of the data in `data/all` is now in
`data/test_data` and the rest in `data/training_data`.

## 9. Run clean data directories

We will need to run the following directory clean up script to make the file
directory compatible with Kaldi feature extraction. We need to run it both on
the the `training_data` and `test_data` directories.

```console
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ utils/fix_data_dir.sh data/training_data/
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ utils/fix_data_dir.sh data/test_data/
```

We can validate that the directories are ready for data preprocessing by running:

```console
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ utils/validate_data_dir.sh data/training_data/
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ utils/validate_data_dir.sh data/test_data/
```

## 10. Feature extraction

Run the feature extraction commands on our training and test folders

```console
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc.conf data/training_data exp/make_mfcc/training_data mfcc
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ steps/compute_cmvn_stats.sh data/training_data exp/make_mfcc/training_data mfcc
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc.conf data/test_data exp/make_mfcc/test_data mfcc
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ steps/compute_cmvn_stats.sh data/test_data exp/make_mfcc/test_data mfcc
```

We should now have successfully created Mel-frequency cepstral coefficients
(MFCC) features and  Cepstral mean and variance normalization (CMVN) for our
training and test data.

## 11. Language Model

ICE-ASR `run.sh` script:

```bash
if [ ! -d data/lang_bi_small ]; then
    echo "Unpacking data/lang_bi_small.tar.gz"
    mkdir data/lang_bi_small
    tar xzf data/lang_bi_small.tar.gz -C data
fi
```

```console
(kaldi-env) USER@terra:~/$ wget -P /home/USER -O rmh2018_2.zip <URL>
```
```console
(kaldi-env) USER@terra:~$ mkdir rmh; unzip rmh2018_2.zip -d rmh; rm rmh2018_2.zip
```

<!-- wget -P /home/USER -O rmh2018_2.zip http://www.malfong.is/tmp/AFD47CF9-8D48-E58C-6025-4A50058EAC66.zip -->

We don't have it, so we need to create our own small bigram model.

### Data

You need a lexicon and a text corpus to build a language model. For Icelandic, you can use the pronunciation dictionary found on http://malfong.is. The text corpus should represent the task of the ASR system, i.e. contain similar language as the utterances the system is expected to recognize. For a general language ASR, one can start experiments with 100,000-200,000 tokens. But depending on the task at hand a larger corpus will improve the results substantially. For an open vocabulary system, a corpus size of about 15 million tokens gives reasonable results (for Icelandic). Although a much larger corpus can be used for a language model used for rescoring of the ASR results.

### Text preparation
For the first experiments, no preprocessing of the text corpus is needed (given that it is stored in plain text format), except for lowercasing, if all other data (dictionary and speech data prompts) have been lowercased as well. 

### Language modeling
In the ice-kaldi implementation we use the MIT Language Modeling Toolkit for language modeling: https://github.com/mitlm/mitlm. The script `s5/local/create_language_model.sh` shows the steps needed to prepare data for language modeling, create a language model and convert it to .fst, the format used by Kaldi to create a final HCLG.fst.

To create a language model used for rescoring the results from the ASR system, a language model has to be created as in `create_language_model.sh` and then the model has to be converted:

    utils/build_const_arpa_lm.sh data/lang_large/trigram.arpa.gz data/lang data/const_arpa_lm_large

Notes:
Til þess búa til LM


## 12. Train an LDA+MLLT acoustic model

Ice-ASR includes a script to train a Latent Dirichlet allocation and Maximum Likelihood Linear Transform acoustic model which is speaker-independent.

```console
(kaldi-env) USER@terra:~/kaldi/egs/ice-kaldi/s5$ local/train_lda_mllt.sh data/training_data/ data/test_data/
```

## 13. Normalize Text

```console
(kaldi-env) USER@terra:~$ cd ~/ice-asr/ice-norm/text-cleaning
(kaldi-env) USER@terra:~/ice-asr/ice-norm/text-cleaning$ python main.py ~/rmh/rmh.txt ~/rmh/rmh_normalized.txt
(kaldi-env) USER@terra:~/ice-asr/ice-norm/text-cleaning$ tail -3 ~/rmh/rmh_normalized.txt 
með vísan til framangreinds ber að sýkna stefnda af bótakröfum stefnanda og viðurkenningu á bótaskyldu
iv 3
málskostnaður
eftir úrslitum málsins að teknu tilliti til niðurstöðu í frávísunarhluta þess og með vísan til þess að samhliða þessu máli er dæmt í sex öðrum málum um sama álitaefni verður stefnanda gert að greiða stefnda málskostnað skv 1 mgr 130 gr laga nr 91 1991 um meðferð einkamála er hæfilegur þykir svo sem í dómsorði greinir að meðtöldum virðisaukaskatti
```
