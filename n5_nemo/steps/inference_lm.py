import sys
import re
import os
import json
import torch
import torchaudio
from torchaudio.models.decoder import ctc_decoder
from tqdm import tqdm
import nemo.collections.asr as nemo_asr
import numpy as np
from nemo.collections.asr.metrics import wer

# Input Parameters
NUM_JOBS = int(sys.argv[1])
EXPERIMENT_PATH = sys.argv[2]
LM_PATH = sys.argv[3]
MANIFEST_PATH = sys.argv[4]
TRAINING_DIR = sys.argv[5]

# Important Variables and Paths
final_model_pointer = os.path.join(TRAINING_DIR, "final_model.path")
with open(final_model_pointer, 'r') as file_pointer:
    model_checkpoint = file_pointer.read().strip()

manifest_name = os.path.basename(MANIFEST_PATH)
PORTION = manifest_name.split("_")[0]

# Loading the pretrained model
nemo_asr_model = nemo_asr.models.EncDecCTCModel.restore_from(model_checkpoint)

# Extract the audio list and reference transcripts from the json file
audio_list = []
id_list = []
ref_trans_list = []
REF_TEXT_LIST = []

with open(MANIFEST_PATH, 'r') as manifest_in:
    for line in manifest_in:
        hash_line = json.loads(line)
        audio_filepath = hash_line['audio_filepath']
        text = re.sub('\s+', ' ', hash_line['text']).strip()
        REF_TEXT_LIST.append(text)
        audio_list.append(audio_filepath)
        audio_id = os.path.basename(audio_filepath).replace(".wav", "")
        id_list.append(audio_id)
        ref_trans_list.append(f"{audio_id} {text}")

# Writing the reference transcriptions in an output file
trans_file_name = f"{PORTION}_reference.trans"
trans_file_path = os.path.join(EXPERIMENT_PATH, trans_file_name)
ref_trans = trans_file_path
with open(trans_file_path, "w") as file_trans:
    for trans in ref_trans_list:
        file_trans.write(f"{trans}\n")

# Initialize the CTC Decoder with Language Model
LM_WEIGHT = 3.23
WORD_SCORE = -0.26

original_vocabulary = nemo_asr_model.decoder.vocabulary
lexicon_path = os.path.join(EXPERIMENT_PATH, "lexicon.txt")

print("Tokens:", original_vocabulary)
print("Number of tokens:", len(original_vocabulary))
print("Lexicon path: ", lexicon_path)

# the ctc_decoder needs a sil_token, but we don't train the model with it, so we extend the tokens with an artificial
# one and need to give it a very low score at inference time. Also accomodate for this artificial token when
# using the ctc_decoder
extended_vocabulary = original_vocabulary + ['<sil>']

beam_search_decoder = ctc_decoder(
    #lexicon=lexicon_path,
    lexicon=None,
    #tokens=extended_vocabulary,
    tokens=original_vocabulary,
    lm=LM_PATH,
    nbest=1,
    beam_size=16,
    lm_weight=LM_WEIGHT,
    word_score=WORD_SCORE,
    blank_token=' ',
    sil_token='z',
)

# Batch size for transcription
BATCH_SIZE = 64

# Generating the hypothesis transcriptions
trans_file_name = f"{PORTION}_hypothesis.trans"
trans_file_path = os.path.join(EXPERIMENT_PATH, trans_file_name)
HYP_TEXT_LIST = []

with open(trans_file_path, "w") as file_trans:
    for i in tqdm(range(0, len(audio_list), BATCH_SIZE), desc="Processing audio files"):
        batch_files = audio_list[i:i+BATCH_SIZE]
        batch_ids = id_list[i:i+BATCH_SIZE]

        # Transcribe the audio batch and get log probabilities
        with torch.no_grad():
            log_probs_batch = nemo_asr_model.transcribe(paths2audio_files=batch_files,
                                                        logprobs=True,
                                                        verbose=False)

        # Find the maximum sequence length in the batch
        max_length = max(log_probs.shape[0] for log_probs in log_probs_batch)
        num_tokens = log_probs_batch[0].shape[1]

        # Prepare batch tensors
        batch_log_probs = torch.zeros(len(log_probs_batch), num_tokens, max_length)
        batch_lengths = []

        for j, log_probs in enumerate(log_probs_batch):
            seq_length = log_probs.shape[0]
            batch_log_probs[j, :, :seq_length] = torch.tensor(log_probs).transpose(0, 1)
            batch_lengths.append(seq_length)

        log_probs_length = torch.tensor(batch_lengths)

        # CTC decoding with torchaudio ctc_decoder
        hypotheses = beam_search_decoder(batch_log_probs, log_probs_length)

        for j, hypothesis in enumerate(hypotheses):
            best_hypothesis = hypothesis[0].words
            HYP_TEXT_LIST.append(best_hypothesis)

            # Write the hypothesis to the output file
            current_id = batch_ids[j]
            line_out = f"{current_id} {best_hypothesis}"
            line_out = re.sub(r'\s+', ' ', line_out).strip()
            file_trans.write(f"{line_out}\n")

        # Optional: Print shapes for debugging
        print(f"Batch size: {len(batch_files)}")
        print(f"batch_log_probs shape: {batch_log_probs.shape}")
        print(f"log_probs_length: {log_probs_length}")
        print(f"Sample hypothesis: {hypotheses[0][0].words}")

# Calculating the WER
# Verbinde die Wörter in jeder Hypothese zu einem einzigen String
HYP_TEXT_LIST = [' '.join(hyp) if isinstance(hyp, list) else hyp for hyp in HYP_TEXT_LIST]

# Stellen Sie sicher, dass REF_TEXT_LIST auch eine Liste von Strings ist
REF_TEXT_LIST = [' '.join(ref) if isinstance(ref, list) else ref for ref in REF_TEXT_LIST]

# Dann berechnen Sie den WER wie zuvor
wer_value = round(100 * wer.word_error_rate(HYP_TEXT_LIST, REF_TEXT_LIST, False), 2)
line_wer = f"WER ({PORTION}) = {wer_value}%"

file_wer_path = os.path.join(EXPERIMENT_PATH, "best_wer")
with open(file_wer_path, "w") as file_wer:
    file_wer.write(line_wer)

# Inform the user
print("\n")
print("######################################")
print(line_wer)
print("######################################")

print(f"\nINFO: ({PORTION}) Hypothesis transcriptions in file: {trans_file_path}")
print(f"INFO: ({PORTION}) Reference transcriptions in file  : {ref_trans}")
print(f"INFO: ({PORTION}) Best WER registered in            : {file_wer_path}")

print("\nINFO: INFERENCE WITH LANGUAGE MODEL SUCCESSFULLY DONE!")