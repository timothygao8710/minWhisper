import numpy as np
import whisper

tokens = np.load('out_toks.npy')
tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False) # type: ignore
print(tokenizer.decode(tokens[0]))