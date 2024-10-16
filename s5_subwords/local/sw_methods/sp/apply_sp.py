#!/usr/bin/env python3

import sentencepiece as spm
import re   
import argparse
import sys

def parse(i_path:str, o_stream:str, model, kaldi_text=True, separator='+', bm='lr'):
    with open(i_path) as f_in:
        for line in f_in:
            if kaldi_text:
                id = line.split(' ')[0]
                line = ' '.join(line.split(' ')[1:])
            line = line.rstrip()
            line_tok = model.encode_as_pieces(line)
            parsed = ''

            print(line_tok)
            if bm == 'wb':
                parsed = ' '.join(line_tok)
                print(parsed)
                parsed = re.sub('^▁', '', parsed)
                parsed = re.sub(' ▁', f' {separator} ', parsed)
                print(parsed)

            else:
                parsed = ''
                tok_iter = iter(line_tok)
                while True:
                    try:
                        tok = next(tok_iter)
                        if tok == '▁':
                            # This indicates that the next token is a whole word  
                            # and we will add that word to the parsed text

                            parsed += ' ' + next(tok_iter)
                        elif tok[0] == '▁':
                            # If the first sign is a "_" then we know that the word was not split 
                            # up here but in the next one. We will add this word but we don't need the 
                            # sign 
                
                            parsed += ' ' + tok[1:]
                        elif '▁' not in tok:
                            # if there is no sign here then the word was split up add this
                            # character. We will add it to the parsed sentece with a space
                            # and a sign
                        
                            parsed += ' ▁' + tok 
                        else:
                            # Nothing should end here
                            sys.exit('Error in apply_sp_bpe.py this should not be here\n'+ tok)

                    except StopIteration:
                        break

            # Lets change the format of the separtor

            if bm =='r':
                parsed = re.sub(' ▁', f'{separator} ', parsed)
            elif bm == 'lr':
                parsed = re.sub(' ▁', f'{separator} {separator}', parsed)
            elif bm == 'l':
                parsed = re.sub(' ▁', f'{separator} ', parsed)


            if kaldi_text:
                o_stream.write(id + ' ' + parsed+'\n')
            else:
                o_stream.write(parsed + '\n')    

def boolean(s):
    if s not in {'False', 'True', 'false', 'true', "1", "0"}:
        raise ValueError('Not a valid boolean')
    return s == 'True' or s=='true' or s=="1"

def model_check(s):
    s = s.lower()
    if s not in {'unigram', 'bpe', 'sp_bpe'}:
        raise ValueError('Nota valid input, should be unigram/bpe')
    if s == 'sp_bpe':
        s = 'bpe'
    return s

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply Unigram/BPE subword tokenzation to a text file using a model trained in train_sp.py,\
            when tokenzing the language model corpus make sure to change the parameter kaldi_text to False')
    parser.add_argument('-m', '--model', required=True, help='Path to the model')
    parser.add_argument('-i', '--input', required=True, help='Path to the text to tokenize')
    parser.add_argument('-o', '--output', required=True, help='Path to the output file')
    parser.add_argument('--kaldi_text', required=False, type=boolean, default=False, help= 'If the tokenizing the "text" in kaldi training/test/eval dirs set as true,\
                                                                                            it will store the leading filnames in those files')
    parser.add_argument('-t', '--type', required=False, type=model_check, default=False, help='In the SentencePiece library we can train either a Unigram or BPE model')
    parser.add_argument( '--separator', '-s', type=str, default='+', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument( '--boundary_marker', '-bm', type=str, default='lr', metavar='STR',
        help="Boundary marker (default: '%(default)s'))")

    args = parser.parse_args()
    
    model = args.type 
    
    sp = spm.SentencePieceProcessor()
    sp.load(args.model + '.model')
    with open(args.output, 'w') as f_out:
        parse(args.input, 
            f_out, 
            sp, 
            args.type, 
            args.separator, 
            args.boundary_marker)