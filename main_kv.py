import numpy as np
from einops import *
from tqdm import trange

import time
start_time = time.time()

import torch
whisper_weights = torch.load('tiny.pt')
del torch # We only use torch for loading the state dict

EOS_ID = 50256
max_gen_toks = 100

# Run preprocessing / tokenization first (on host), must be a padded / trimmed mel-spectrogram with n_mels = dims['n_mels']
audio_input = np.load('example_aud.npy')
audio_input = rearrange(audio_input, 'b c t -> b t c') # audio_input.shape = Batch, Time, n_mels (different from dims['n_audio_state'] = hidden dim size / feature / channel dim)

tokens_input = np.load('example_toks.npy') # Run preprocessing first
sd, dims = whisper_weights['model_state_dict'], whisper_weights['dims']
C = dims['n_audio_state'] # Hidden dim size / feature / channel dim
assert C == dims['n_text_state'] 

### INGREDIENTS ###
def gelu(x_arr): # docs.jax.dev/en/latest/_autosummary/jax.nn.gelu.html#jax.nn.gelu
    sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x_arr.dtype)
    cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x_arr + 0.044715 * (x_arr ** 3))))
    return x_arr * cdf

def ln(x, name):
    W, b = np.array([sd[f'{name}.weight'], sd[f'{name}.bias']])
    x = x - np.mean(x, axis=-1, keepdims=True)
    x = x / (np.std(x, axis=-1, keepdims=True) + 1e-9)
    return W * x + b

kv_cache = {}

def attn(q_x, kv_x, name, is_casual=False):
    # We need 
    # - full self attention (in encoder)
    # - casual/masked self attention (in decoder)
    # - cross attention (in decoder)

    W_q, W_k, W_v, W_o = [np.array(sd[f'{name}.{item}.weight']) for item in ['query', 'key', 'value', 'out']] # load weights / biases
    B_q, B_v, B_o = [np.array(sd[f'{name}.{item}.bias']) for item in ['query', 'value', 'out']] # No bias for key, softmax is invariant to constant addition

    resid_x = q_x.copy()

    q_x = ln(q_x, f'{name}_ln')
    Q = q_x @ W_q.T + B_q

    if kv_x is None: 
        kv_x = q_x # Not cross attention

    if name not in kv_cache: 
        kv_cache[name] = np.array([kv_x @ W_k.T, kv_x @ W_v.T + B_v]) # prefill
    elif is_casual:
        kv_cache[name], _ = pack([kv_cache[name], np.array([kv_x @ W_k.T, kv_x @ W_v.T + B_v])], 'm b * c') # decode
        is_casual = False # casual attention reduces to cross attention

    K, V = rearrange(kv_cache[name], 'm b kt (n_heads c) -> m b n_heads kt c', n_heads = n_heads)
    Q = rearrange(Q, 'b qt (n_heads c) -> b n_heads qt c', n_heads = n_heads) # Q time dim may be diff from KV time dim for cross attn

    scores = einsum(Q, K, 'b n qt c, b n kt c -> b n qt kt') # contract feature dim
    scores /= np.sqrt(C / n_heads)

    scores = scores - np.max(scores, axis=-1, keepdims=True) # if rows are Q, cols are K, here we're normalizing along rows (t)

    if is_casual: # only relevant in prefill for casual / masked self attention
        scores += np.triu(np.full(scores.shape, -np.inf), k=1)

    scores = np.exp(scores)
    scores /= np.sum(scores, axis=-1, keepdims=True)

    x = einsum(scores, V, 'b n qt kt, b n kt c -> b n qt c') # contracting along time axis of KV (post softmax scores)
    x = rearrange(x, 'b n qt c -> b qt (n c)')

    x = x @ W_o.T + B_o

    return x + resid_x

def mlp(x, name):
    W_up, W_down = [np.array(sd[f'{name}.{item}.weight']) for item in ['0', '2']] # load weights / biases
    B_up, B_down = [np.array(sd[f'{name}.{item}.bias']) for item in ['0', '2']] 

    resid_x = x.copy()

    x = ln(x, f'{name}_ln')
    x = gelu(x @ W_up.T + B_up)
    x = x @ W_down.T + B_down

    return x + resid_x

### END OF INGREDIENTS ###

### AUDIO ENCODER ###

x = audio_input

for i in [1,2]:
    W, b = np.array(whisper_weights['model_state_dict'][f'encoder.conv{i}.weight']), np.array(whisper_weights['model_state_dict'][f'encoder.conv{i}.bias'])
    pad = np.zeros((x.shape[0], 1, x.shape[2]))
    x, _ = pack([pad, x, pad], 'b * c') # padding = 1 for all whisper models

    x = einsum(np.array([
        x[:, :-2:i, :],  # stride 2 for layer 2 (stride i for layer i)
        x[:, 1:-1:i, :],
        x[:, 2::i, :]

    ]), W, 'convdim b t c, cc c convdim -> b t cc') # contract feature dimension, convdim (element-wise mult then sum them)

    x += b
    x = gelu(x)

x += np.array(sd['encoder.positional_embedding'])

n_heads, n_layers = dims['n_audio_head'], dims['n_audio_layer']

for i in trange(n_layers, desc="Audio Encoder - Layers"):
    x = attn(q_x = x, kv_x = None, name = f"encoder.blocks.{i}.attn")
    x = mlp(x = x, name = f"encoder.blocks.{i}.mlp")

encoded_audio = ln(x, 'encoder.ln_post') # final layernorm

### END OF AUDIO ENCODER ###

### TEXT DECODER ###

res = tokens_input
vocab = np.array(sd['decoder.token_embedding.weight']) # map from token_id -> length-C feature vector
pos_embed = np.array(sd['decoder.positional_embedding'])
n_heads, n_layers = dims['n_text_head'], dims['n_text_layer']

for tok in trange(max_gen_toks, desc="Text Decoder - Num Output Tokens"):
    x = vocab[tokens_input]
    _, T, _ = x.shape
    x += pos_embed[:T, :]
    pos_embed = pos_embed[T:]

    for i in range(n_layers):
        x = attn(x, None, f'decoder.blocks.{i}.attn', is_casual=True)
        x = attn(x, encoded_audio, f'decoder.blocks.{i}.cross_attn')
        x = mlp(x, f'decoder.blocks.{i}.mlp')

    x = ln(x, 'decoder.ln') # final / model norm

    x = x @ vocab.T # dot product with each word embedding
    x = np.argmax(x, axis=-1)

    if np.all(x[:, -1:] == EOS_ID):
        break

    tokens_input = x[:, -1:] # O(N^3) -> O(N^2) attention with KV Caching & compute-bound -> BW-bound
    res, _ = pack([res, tokens_input], 'b *')

print(f"{time.time() - start_time:.3f}")  # milliseconds

### END OF TEXT DECODER ###

np.save('out_toks.npy', res)