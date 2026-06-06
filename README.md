# minWhisper

This repo implements all of OpenAI Whisper's forward pass in under 150 lines of Numpy using Einsum / Einops.

https://github.com/user-attachments/assets/f1fbdad8-87c0-4d1b-bebc-2f9301481574

- KV cache is 7 lines on top of main.py (O(seq_len ^ 3) -> O(seq_len ^ 2))
- Supports any model size in the Whisper family, batched inference, and different audio formats
- Details like layernorm and approximate gelu differ slightly from huggingface's implementation to prefer conciseness

```
Compare to main.py, the key changes in main_kv.py are

+ kv_cache = {}

+ if name not in kv_cache: 
        kv_cache[name] = np.array([kv_x @ W_k.T, kv_x @ W_v.T + B_v]) # prefill
    elif is_casual:
        kv_cache[name], _ = pack([kv_cache[name], np.array([kv_x @ W_k.T, kv_x @ W_v.T + B_v])], 'm b * c') # decode
        is_casual = False # casual attention reduces to cross attention

Implements KV cache for cross attention (prefill-only), and decode in masked attention by viewing it as cross attention with one query

And of course

tokens_input, _ = pack([tokens_input, x[:, -1:]], 'b *') --> tokens_input = x[:, -1:]

Is what actually buys us the reduction in complexity, by only doing the "new" work incurred for each new token
```

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

# Model Architecture

<img width="887" height="665" alt="Screenshot 2026-06-06 at 1 43 22 AM" src="https://github.com/user-attachments/assets/864d6076-e5b7-4904-870b-21dcbb9bc2a6" />


From https://cdn.openai.com/papers/whisper.pdf

# References

- https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py
- https://cdn.openai.com/papers/whisper.pdf
- https://jessicastringham.net/2018/01/01/einsum/
- https://einops.rocks/1-einops-basics/
