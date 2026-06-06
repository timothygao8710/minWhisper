import torch
import numpy as np
import whisper

audio_path = "example.mp3"
n_mels = torch.load('tiny.pt')['dims']['n_mels']

audio = whisper.load_audio(audio_path)
audio = whisper.pad_or_trim(audio)

mel = whisper.log_mel_spectrogram(audio, n_mels=n_mels)
mel = mel.unsqueeze(0) # add batch dimension

print("audio:", audio.shape, audio.dtype)
print("mel:", mel.shape, mel.dtype)

tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False) # type: ignore

tokens = np.array(
    [tokenizer.sot_sequence_including_notimestamps],
    dtype=np.int32,
)

print(tokens.shape)

np.save('example_aud.npy', mel)
np.save('example_toks.npy', tokens)