import torch
import torch.nn as nn
import math
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

def exists(_value):
    return _value is not None

def default(_value, _default):
    return _value if exists(_value) else _default

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        readable_start_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
        print(f"Timer: Start {func.__name__} at {readable_start_time}")
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        minutes, seconds = divmod(elapsed_time, 60)
        print(f"{func.__name__} took {int(minutes):02d}:{seconds:02.0f} to run.")
        return result
    return wrapper

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)
    
def fourier_encode(x:torch.Tensor, max_freq:int, num_bands:int=4, base:int=2):
    x = x.unsqueeze(-1).float()
    orig_x = x
    scales = torch.logspace(
        1.0,
        math.log(max_freq / 2) / math.log(base),
        steps=num_bands,
        base=base,
        dtype=torch.float32,
    ).to(x.device)
    
    scales = scales.view(*([1] * (len(x.shape) - 1)), -1)

    x = x * scales * math.pi
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x

def mask_tokens(tokens, input_vocab=5**4, prob=0.15):
    tokens = tokens.clone()
    batch_size, total_tokens = tokens.shape
    num_mask = int(total_tokens*prob)
    mask_positions = torch.randperm(total_tokens)[:num_mask]
    # mask = torch.zeros_like(tokens, dtype=torch.bool).cpu()
    mask = torch.zeros((1,total_tokens), dtype=torch.bool).cpu()
    mask_map = mask.clone()
    random_map = mask.clone()
    mask[:,mask_positions] = True
    
    probs = torch.rand(num_mask)
    mask_indices = mask_positions[probs < 0.8]
    mask_map[:,mask_indices] = True
    random_indices = mask_positions[(probs >= 0.8) & (probs < 0.9)]
    random_map[:, random_indices] = True

    tokens[mask_map.expand_as(tokens)] = input_vocab
    random_map = random_map.expand_as(tokens)
    tokens[random_map] = torch.randint(0, input_vocab, size=(len(random_indices)*batch_size,)).to(tokens.device)
    
    return tokens, (mask.expand_as(tokens)|random_map).to(tokens.device)


def check_nan(tensor, name):
    if isinstance(tensor, torch.Tensor):
        if torch.isnan(tensor).any():
            print(f"NaN found in {name}")
            return True
    elif isinstance(tensor, (list, tuple)):
        for t in tensor:
            if torch.isnan(t).any():
                print(f"NaN found in {name}")
                return True
    return False
    
def check_model(model, input):
    for name, layer in model.named_modules():
        output = layer(input)
        check_nan(output, f"Output of {name}")
        if hasattr(layer, 'weight') and layer.weight is not None:
            check_nan(layer.weight, f"Weight of {name}")
        if hasattr(layer, 'bias') and layer.bias is not None:
            check_nan(layer.bias, f"Bias of {name}")
        input = output

def forward_hook(module, input, output):
    check_nan(output, f"Output of {module}")

def backward_hook(module, grad_input, grad_output):
    if any(torch.isnan(g).any() for g in grad_input):
        print(f"NaN in grad_input of {module}")
    if any(torch.isnan(g).any() for g in grad_output):
        print(f"NaN in grad_output of {module}")


class Record(object):
    def __init__(self, *keys):
        for key in keys:
            setattr(self, key, [])

    def save(self, output_dir):
        for key in vars(self).keys():
            data = np.array(getattr(self, key))
            np.save(os.path.join(output_dir, f'record.data.{key}.npy'), data)
            plt.figure(figsize=(10, 6))
            plt.plot(data, label=key, linewidth=2)
            plt.title(f'Training Metrics: {key}')
            plt.xlabel(key.split("_")[0])
            plt.ylabel(key.split("_")[1])
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f'record.fig.{key}.png'))
            plt.close()