import torch
import numpy as np
import whisper

audio_path = "/Users/timothyg/Documents/whisper_numpy/example.wav"

n_mels = torch.load('/Users/timothyg/Documents/whisper_numpy/tiny.pt')['dims']['n_mels']
# n_mels = torch.load('/Users/timothyg/Documents/whisper_numpy/med.pt')['dims']['n_mels']

audio = whisper.load_audio(audio_path)
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
mel = mel.unsqueeze(0) # add batch dimension

print("audio:", audio.shape, audio.dtype)
print("mel:", mel.shape, mel.dtype)

tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)

tokens = np.array(
    [tokenizer.sot_sequence_including_notimestamps],
    dtype=np.int64,
)

print(tokens.shape)

np.save('example_aud.npy', mel)
np.save('example_toks.npy', tokens)