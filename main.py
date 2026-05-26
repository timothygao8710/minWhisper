import numpy as np
from template import *
from einops import *
from tqdm import trange
import torch
whisper_weights = torch.load('tiny.pt')
torch = None

sampling_t = None # TODO, currently greedy
max_gen_toks = 100

audio_input = np.load('/Users/timothyg/Documents/whisper_numpy/example_aud.npy')
audio_input = rearrange(audio_input, 'b c t -> b t c') # Run preprocessing first, must be padded / trimmed, mel-spectrogram with n_mels = dims['n_mels']

tokens_input = np.load('/Users/timothyg/Documents/whisper_numpy/example_toks.npy') # Run preprocessing first
sd, dims = whisper_weights['model_state_dict'], whisper_weights['dims']

B, T, n_mels = audio_input.shape # Batch, Time, n_mels (different from Channel / Feature / Hidden-Dim)
C = dims['n_audio_state']
assert dims['n_audio_state'] == dims['n_text_state'] 

dbg("audio_input:", audio_input.shape, audio_input.dtype)
dbg("tokens_input:", tokens_input.shape, tokens_input.dtype)

### INGREDIENTS ###
def gelu(x_arr): # docs.jax.dev/en/latest/_autosummary/jax.nn.gelu.html#jax.nn.gelu
    sqrt_2_over_pi = np.sqrt(2 / np.pi).astype(x_arr.dtype)
    cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x_arr + 0.044715 * (x_arr ** 3))))
    return x_arr * cdf

def ln(x, W, b):
    x = x - np.mean(x, axis=-1, keepdims=True)
    x = x / (np.std(x, axis=-1, keepdims=True) + 1e-9)
    # var = np.var(x, axis=-1, keepdims=True)
    # x = (x - np.mean(x, axis=-1, keepdims=True)) / np.sqrt(var + 1e-5)
    return W * x + b

def attn(q_x, kv_x, name, is_casual=False):
    
    # we need 
    # - full self attention (in encoder)
    # - casual/masked self attention (in decoder)
    # - cross attention (in decoder)

    # load weights / biases
    W_q, W_k, W_v, W_o = [np.array(sd[f'{name}.{item}.weight']) for item in ['query', 'key', 'value', 'out']]
    B_q, B_v, B_o = [np.array(sd[f'{name}.{item}.bias']) for item in ['query', 'value', 'out']] # No bias for key, softmax is invariant to constant addition
    ln_w, ln_b = np.array([sd[f'{name}_ln.weight'], sd[f'{name}_ln.bias']])

    resid_x = q_x.copy()

    q_x = ln(q_x, ln_w, ln_b)

    if kv_x is None:
        kv_x = q_x
    
    Q, K, V = q_x @ W_q.T + B_q, kv_x @ W_k.T, kv_x @ W_v.T + B_v 

    K, V = rearrange([K, V], 'm b kt (n_heads c) -> m b n_heads kt c', n_heads = n_heads)
    Q = rearrange(Q, 'b qt (n_heads c) -> b n_heads qt c', n_heads = n_heads) # Q time dim may be diff from KV time dim for cross attn

    scores = einsum(Q, K, 'b n qt c, b n kt c -> b n qt kt') # contract feature dim
    scores /= np.sqrt(C / n_heads)

    scores = scores - np.max(scores, axis=-1, keepdims=True) # if rows are Q, cols are K, here we're normalizing along rows (t)

    if is_casual:
        scores += np.triu(np.full(scores.shape, -np.inf), k=1)

    scores = np.exp(scores)
    scores /= np.sum(scores, axis=-1, keepdims=True)

    x = einsum(scores, V, 'b n qt kt, b n kt c -> b n qt c') # contracting along time axis of KV (post softmax scores)
    x = rearrange(x, 'b n qt c -> b qt (n c)')

    x = x @ W_o.T + B_o

    return x + resid_x

def mlp(x, name):
    # load weights / biases
    W_up, W_down = [np.array(sd[f'{name}.{item}.weight']) for item in ['0', '2']]
    B_up, B_down = [np.array(sd[f'{name}.{item}.bias']) for item in ['0', '2']] 
    ln_w, ln_b = np.array([sd[f'{name}_ln.weight'], sd[f'{name}_ln.bias']])

    resid_x = x.copy()

    x = ln(x, ln_w, ln_b)
    x = gelu(x @ W_up.T + B_up)
    x = x @ W_down.T + B_down

    return x + resid_x

### END OF INGREDIENTS ###







### AUDIO ENCODER ###

encoded_audio = None

dbg("Encoding audio input...")

x = audio_input

for i in [1,2]:
    W, b = np.array(whisper_weights['model_state_dict'][f'encoder.conv{i}.weight']), np.array(whisper_weights['model_state_dict'][f'encoder.conv{i}.bias'])
    pad = np.zeros((B, 1, x.shape[2]))
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

for i in trange(n_layers, desc="layers"):
    x = attn(q_x = x, kv_x = None, name = f"encoder.blocks.{i}.attn")
    x = mlp(x=x, name=f"encoder.blocks.{i}.mlp")

ln_w, ln_b = np.array([sd['encoder.ln_post.weight'], sd['encoder.ln_post.bias']])
encoded_audio = ln(x, ln_w, ln_b) # final layernorm

### END OF AUDIO ENCODER ###



### TEXT DECODER ###

dbg("Decoding text...")

vocab = np.array(sd['decoder.token_embedding.weight']) # map from token_id -> length-C feature vector

for tok in trange(max_gen_toks, desc="num generated tokens"):

    x = vocab[tokens_input]
    _, T, _ = x.shape
    pos_embed = np.array(sd['decoder.positional_embedding'])[:T, :]
    x += pos_embed

    n_heads, n_layers = dims['n_text_head'], dims['n_text_layer']
    for i in range(n_layers):
        x = attn(x, None, f'decoder.blocks.{i}.attn', is_casual=True)
        x = attn(x, encoded_audio, f'decoder.blocks.{i}.cross_attn')
        x = mlp(x, f'decoder.blocks.{i}.mlp')

    ln_w, ln_b = np.array([sd['decoder.ln.weight'], sd['decoder.ln.bias']]) # final / model norm
    x = ln(x, ln_w, ln_b)

    res = x @ vocab.T # dot product with each word embedding
    res = np.argmax(res, axis=-1)
    # TODO: Stop once we each eos

    tokens_input, _ = pack([tokens_input, res[:, -1:]], 'b *')


### END OF TEXT DECODER ###

np.save('/Users/timothyg/Documents/whisper_numpy/out_toks.npy', tokens_input)