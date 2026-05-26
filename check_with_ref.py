import whisper

model = whisper.load_model("/Users/timothyg/Documents/whisper_numpy/tiny.pt")
result = model.transcribe("/Users/timothyg/Documents/whisper_numpy/example__.wav", language="en")
print(result["text"])