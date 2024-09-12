import kenlm
import argparse

def extract_vocabulary(lm_path):
    lm = kenlm.Model(lm_path)
    vocabulary = set()

    # Use a sentence with all possible characters to extract vocabulary
    all_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789áðéíóúýþæöÁÐÉÍÓÚÝÞÆÖ"
    sentence = " ".join(all_chars)

    # Get the state after processing the sentence
    state = kenlm.State()
    lm.BeginSentenceWrite(state)

    for word in sentence.split():
        # Check if the word is in the vocabulary
        if not lm.BaseScore(state, word, None) == 0:
            vocabulary.add(word)

    return vocabulary

def generate_lexicon(vocabulary, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for word in sorted(vocabulary):
            f.write(f"{word} {' '.join(word)}\n")

def main():
    parser = argparse.ArgumentParser(description="Generate lexicon from KenLM model")
    parser.add_argument("--model", required=True, help="Path to KenLM model")
    parser.add_argument("--lexicon", required=True, help="Output path for lexicon file")
    args = parser.parse_args()

    vocabulary = extract_vocabulary(args.model)
    generate_lexicon(vocabulary, args.lexicon)
    print(f"Lexicon generated and saved to {args.lexicon}")

if __name__ == "__main__":
    main()