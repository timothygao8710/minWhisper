import numpy as np
import whisper

tokens = np.load('out_toks.npy')
print(tokens.shape, tokens.dtype)
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
print(tokenizer.decode(tokens[0]))