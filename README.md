# minWhisper

This repo implements all of OpenAI Whisper's forward pass in under 150 lines of Numpy using Einsum / Einops.

https://github.com/user-attachments/assets/f1fbdad8-87c0-4d1b-bebc-2f9301481574

- KV cache is 7 lines on top of main.py (O(seq_len ^ 3) -> O(seq_len ^ 2))
- Supports any model size in the Whisper family, batched inference, and different audio formats
- Details like layernorm and approximate gelu differ slightly from huggingface's implementation to prefer conciseness

# Quickstart

1. Download any choice of model checkpoint:

- curl -L -o tiny.pt https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt

- curl -L -o small.pt https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt

- curl -L -o med.pt https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt

More are avaliable at: https://github.com/openai/whisper/blob/main/whisper/__init__.py. Note multilingal versions require different tokenization.

2. Run preprocess.py. This handles converting to mel-spectogram and tokenization (usually done on-host), the correct input format. It will also generate a numpy file containing template text token scaffolding.

3. Run main_kv.py or main.py

4. Run post-process to detokenize the model's output tokens into human-readable form (usually done on-host)

# KV Cache Benchmarks

<img width="600" alt="inference_benchmark" src="https://github.com/user-attachments/assets/745eff2a-e89a-45fd-8e7e-4511e6a51739" />

Ran on MacBook Pro M2 Pro, 2023

# Model Architecture Implemented

<img width="648" alt="whisper_model" src="https://github.com/user-attachments/assets/e748d28e-797f-43fa-80a6-d761e41211ab" />

From https://cdn.openai.com/papers/whisper.pdf

# References

- https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
- https://cdn.openai.com/papers/whisper.pdf
- https://jessicastringham.net/2018/01/01/einsum/
- https://einops.rocks/1-einops-basics/
