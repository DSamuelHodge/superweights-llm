# LLMSuperWeight Analysis

## Repository Summary

- **Generated:** 2025-01-25T23:15:35.065133
- **Total Files:** 16
- **Total Lines:** 1,685
- **Total Size:** 735.4KB
- **Total Tokens:** 493,523

## File Type Statistics

| Extension | Files | Lines | Size |
|-----------|-------|-------|------|
| .py | 5 | 1,015 | 53.5KB |
| .ipynb | 1 | 515 | 674.5KB |
| .sh | 10 | 155 | 7.4KB |

## Source Tree

```
├── outliers/
│   ├── functional/
│   │   ├── quantization.py                [  6.3KB,  130 lines]
│   │   └── utils.py                       [  9.8KB,  175 lines]
│   └── model.py                       [ 24.0KB,  438 lines]
├── scripts/
│   ├── figure3_how_to_identify_superweight.sh [  45.0B,    1 lines]
│   ├── figure4_how_superweight_affect_superactivation.sh [  46.0B,    1 lines]
│   ├── figure5_token_probability.sh   [  34.0B,    1 lines]
│   ├── figure6_super_weight_scaling.sh [ 473.0B,   11 lines]
│   ├── run_SW_importance_ablation.sh  [  1.1KB,   23 lines]
│   ├── run_test_manual_ablation.sh    [ 654.0B,   15 lines]
│   ├── run_test_quantize_loop.sh      [  3.8KB,   72 lines]
│   ├── run_test_sensitivity.sh        [ 485.0B,   11 lines]
│   ├── table1_superweight_importance.sh [ 786.0B,   20 lines]
│   └── table2_find_superweight.sh     [   0.0B,    0 lines]
├── analyze.py                     [ 13.2KB,  268 lines]
├── evaluate.py                    [ 167.0B,    4 lines]
└── figures.ipynb                  [674.5KB,  515 lines]
```

## File Contents

### analyze.py

```py
import matplotlib.pyplot as plt
import torch
from clize import run
from outliers.functional.utils import add_global_plot_styles
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_module_hidden_states(model, tokenizer, test_text, layer_path, module_name, input_or_output="output", plot_fname=None, spike_threshold=100):
    if input_or_output not in ["input", "output"]:
        raise ValueError("input_or_output should be 'input' or 'output', instead of", input_or_output)
    
    all_activations = {}

    def get_activations(layer_index):
        def hook(model, inputs, outputs):
            hidden_states = inputs if input_or_output == "input" else outputs
            all_activations.setdefault(layer_index, {})[f"{module_name}_{input_or_output}_hidden_states"] = hidden_states
        return hook   

    all_hooks = []

    def get_layers(model, layer_path):
        attributes = layer_path.split('.')
        layers = model
        for attr in attributes:
            layers = getattr(layers, attr)
        return layers

    attributes = module_name.split('.') if module_name != "layer" else []
    layers = get_layers(model, layer_path)

    for layer_index, layer in enumerate(layers):
        current_attr = layer
        valid = True
        for attr in attributes:
            if hasattr(current_attr, attr):
                current_attr = getattr(current_attr, attr)
            else:
                valid = False
                break
        
        if valid:
            hook = current_attr.register_forward_hook(get_activations(layer_index))
            all_hooks.append(hook)

    inputs = tokenizer(test_text, return_tensors='pt').to(model.device)
    model.eval()
    with torch.no_grad():
        model(**inputs)

    for hook in all_hooks:
        hook.remove()

    top1_values_all_layers = []
    top1_indexes_all_layers = []
    for layer_index, outputs in all_activations.items():
        values = outputs[f'{module_name}_{input_or_output}_hidden_states']
        tensor = values[0] if isinstance(values, tuple) else values
        tensor = tensor.detach().cpu()
        tensor_abs = tensor.view(-1).abs().float()

        max_value, max_index = torch.max(tensor_abs, 0)
        max_index = torch.unravel_index(max_index, tensor.shape)
        top1_values_all_layers.append(tensor[max_index])
        top1_indexes_all_layers.append(max_index)

    return top1_values_all_layers, top1_indexes_all_layers

def plot_down_proj_input_output(pretrained="allenai/OLMo-7B-0724-hf", module_name="mlp.down_proj"):
    model = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

    test_text = "Apple Inc. is a worldwide tech company."
    layer_path = "model.layers"

    for name in ("input", "output"):
        magnitude, index = check_module_hidden_states(
            model, tokenizer, test_text, layer_path, module_name, input_or_output=name, spike_threshold=50)

        # Report any spikes
        spikes_input = [i for i, value in enumerate(magnitude) if abs(value) > 50]
        print(f"Activation spikes for {module_name} {name}:")
        for i in spikes_input:
            spike_index = index[i]
            print(f" - layer {i}, value {magnitude[i]}, index {tuple(i.item() for i in spike_index)}")

        # Plot input activations
        plt.figure(figsize=(5,3.5))
        add_global_plot_styles()
        plt.plot(range(len(magnitude)), magnitude, color='blue', marker='o', markersize=5)
        plt.xlabel('Layer Number')
        plt.ylabel('Max Activation Value')
        plt.title(f"OLMo-7B Max down_proj {name}")
        plt.yticks(rotation=90, va='center')
        plt.savefig(f"outputs/figures/{name}_down_proj.pdf", bbox_inches='tight')
        print(f"Plot saved to 'outputs/figures/{name}_down_proj.pdf'")

        # Print output magnitudes
        print(f"largest_activations_down_proj_{name}={list(map(float, magnitude))}")

def record_SO(model, pretrained):
    '''Record SO values for original models'''
    SUPER_WEIGHTS_MAP = {
        "Mistral-7B-v0.1": [(1, 2070, 7310)],
        "llama-7B": [(2, 3968, 7003)],
        "llama-13B": [(2, 2231, 2278), (2, 2231, 6939)],
        "llama-30B": [(3, 5633, 12817), (3, 5633, 17439), (10, 5633, 14386)],
        "Meta-Llama-3-8B": [(1, 788, 2427), (1, 1384, 2427), (1, 4062, 2427)],
        "OLMo-1B-0724-hf": [(1, 1764, 1710), (2, 1764, 8041)],
        "OLMo-7B-0724-hf": [(1, 269, 7467), (2, 269, 8275), (7, 269, 453), (24, 269, 2300)],
        "Phi-3-mini-4k-instruct": [(2, 525, 808), (2, 1693, 808), (2, 1113, 808), (4, 525, 2723),  (4, 1113, 2723), (4, 1693, 2723)],
    }
    
    def _record_SO(SO_map, layer, row, col):
        if pretrained in [
            "tiiuae/falcon-7b",
        ]:
            SO_map[(layer, row, col)] = model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col].item()
        else:
            SO_map[(layer, row, col)] = model.model.layers[layer].mlp.down_proj.weight.data[row, col].item()

    SO_values = {}
    for model_name, coordinates in SUPER_WEIGHTS_MAP.items():
        if model_name in pretrained:
            for layer, row, col in coordinates:
                _record_SO(SO_values, layer, row, col)
            break
    return SO_values

def scale_SO(model, pretrained, SO_values, scaling_factor):
    if pretrained in [
        "huggyllama/llama-30B", 
        "huggyllama/llama-13B", 
        "huggyllama/llama-7B", 
        "mistralai/Mistral-7B-v0.1",
        "meta/Meta-Llama-3-8B",
        "allenai/OLMo-1B-0724-hf",
        "allenai/OLMo-7B-0724-hf",
        "microsoft/Phi-3-mini-4k-instruct"
        ]:
        for (layer, row, col), value in SO_values.items():  
            old_value = model.model.layers[layer].mlp.down_proj.weight.data[row, col].item()
            new_value = value * scaling_factor
            model.model.layers[layer].mlp.down_proj.weight.data[row, col] = new_value
            print(f"Layer {layer}, Index [{row}, {col}], Old value: {old_value}, New value: {new_value}")

def remove_outliers(model, pretrained, percentage_threshold):

    num_selected_elements = []
    for name, param in model.named_parameters():
        if not name.endswith("weight"):
            continue

        weight = param.data
        num_top_elements = int(weight.numel() * percentage_threshold)
        # print("# Total params:", weight.numel(), "# top params", num_top_elements)
        if num_top_elements < 1: # too few elements to apply on
            continue 
        threshold = torch.topk(weight.view(-1).abs(), num_top_elements).values[-1]
        mask = weight.abs() >= threshold
        true_indices = mask.nonzero(as_tuple=False)
        num_selected_elements.append(len(true_indices))

        weight[mask] = 0.
        param = torch.nn.Parameter(weight)   
    


def plot_max_activation_ablation(pretrained="allenai/OLMo-7B-0724-hf"):
    model = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)

    
    test_text = "Apple Inc. is a worldwide tech company."
    layer_path = "model.layers"
    module_name = "layer"
    name = "output"
    # original
    magnitude, index = check_module_hidden_states(
            model, tokenizer, test_text, layer_path, module_name, input_or_output=name, spike_threshold=50)
    # Print output magnitudes
    print(f"original={list(map(float, magnitude))}")
    
    # remove SO
    SO_values = record_SO(model, pretrained)
    scale_SO(model, pretrained, SO_values, 0)
    magnitude, index = check_module_hidden_states(
            model, tokenizer, test_text, layer_path, module_name, input_or_output=name, spike_threshold=50)
    # Report any spikes
    spikes_input = [i for i, value in enumerate(magnitude) if abs(value) > 50]
    print(f"Activation spikes for {module_name} {name}:")
    for i in spikes_input:
        spike_index = index[i]
        print(f" - layer {i}, value {magnitude[i]}, index {tuple(i.item() for i in spike_index)}")
    # Print output magnitudes
    print(f"super_weight_removed={list(map(float, magnitude))}")

    # remove outliers
    percentage = 5e-7
    remove_outliers(model, pretrained, percentage)
    magnitude, index = check_module_hidden_states(
            model, tokenizer, test_text, layer_path, module_name, input_or_output=name, spike_threshold=50)
    # Print output magnitudes
    print(f"all_outliers_removed={list(map(float, magnitude))}")

    # restore SO
    scale_SO(model, pretrained, SO_values, 1)
    magnitude, index = check_module_hidden_states(
            model, tokenizer, test_text, layer_path, module_name, input_or_output=name, spike_threshold=50)
    # Print output magnitudes
    print(f"all_other_outliers_removed={list(map(float, magnitude))}")

def plot_token_probs(pretrained="mistralai/Mistral-7B-v0.1"):

    model_name = pretrained
    MODEL_ID = model_name.split('/')[1]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
        low_cpu_mem_usage=True,
    )

    model_map = {
        "Mistral-7B-v0.1": [(1, 2070, 7310)],
        "llama-7B": [(2, 3968, 7003)],
        "llama-13B": [(2, 2231, 2278), (2, 2231, 6939)],
        "llama-3v0B": [(3, 5633, 12817), (3, 5633, 17439), (10, 5633, 14386)],
        "Meta-Llama-3-8B": [(1, 788, 2427), (1, 1384, 2427), (1, 4062, 2427)],
        "OLMo-1B-0724-hf": [(1, 1764, 1710), (2, 1764, 8041)],
        "OLMo-7B-0724-hf": [(1, 269, 7467), (2, 269, 8275), (7, 269, 453), (24, 269, 2300)],
        "gemma-7b": [(0, 1995, 21041)], # not sufficient
        "Phi-3-mini-4k-instruct": [(2, 525, 808), (2, 1693, 808), (2, 1113, 808), (4, 525, 2723),  (4, 1113, 2723), (4, 1693, 2723)],
        # "tiiuae/falcon-7b": [(3, 2002, 10708), (4, 2002, 5921)]
    }
    sw_map = {}

    def remove_SO(model):
        sw_map[MODEL_ID] = []
        for (layerno, y, x) in model_map[MODEL_ID]:
            weight = model.model.layers[layerno].mlp.down_proj.weight.data
            sw_map[MODEL_ID].append(float(weight[y, x]))
            weight[y, x] = 0.
            model.model.layers[layerno].mlp.down_proj.weight = torch.nn.Parameter(weight)

    def restore_SO(model):
        assert sw_map.get(MODEL_ID, None), "Run remove_SO before running restore_SO"
        for value, (layerno, y, x) in zip(sw_map[MODEL_ID], model_map[MODEL_ID]):
            weight = model.model.layers[layerno].mlp.down_proj.weight.data
            weight[y, x] = value
            model.model.layers[layerno].mlp.down_proj.weight = torch.nn.Parameter(weight)

    def print_SO(model):
        for weight, (layerno, y, x) in zip(sw_map[MODEL_ID], model_map[MODEL_ID]):
            weight = model.model.layers[layerno].mlp.down_proj.weight.data
            print(weight[y, x])


    def get_next_token_probs(model, tokenizer, input_text):
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        # Get model outputs
        with torch.no_grad():
            outputs = model(input_ids)
            
            # Get the logits for the next token prediction
            next_token_logits = outputs.logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
        return next_token_probs


    from datasets import load_dataset
    import json
    from tqdm import tqdm

    N_SAMPLES = 500
    dataset = load_dataset("EleutherAI/lambada_openai", "en", split="test")

    all_difference = []

    all_probs_SO_removed = []
    all_probs_Original = []


    # Original model
    for text in tqdm(dataset[:N_SAMPLES]["text"]):
        prompt = ' '.join(text.split(' ')[:-1])
        target = text.split(' ')[-1]
        next_token_probs = get_next_token_probs(model, tokenizer, prompt)
        all_probs_Original.append(next_token_probs)

    avg_probs_Original = (sum(all_probs_Original)  / len(all_probs_Original))[0] 
    avg_probs_Original = avg_probs_Original.to('cpu')
    sorted_probs_Original, sorted_indices_Original = torch.sort(avg_probs_Original, descending=True)
    top_n = 100
    top_n_probs_Original = sorted_probs_Original[:top_n].tolist()
    top_n_indices_Original = sorted_indices_Original[:top_n]
    top_tokens = [tokenizer.decode(i) for i in top_n_indices_Original]

    # Remove super weight
    remove_SO(model)
    print(sw_map[MODEL_ID])

    for text in tqdm(dataset[:N_SAMPLES]["text"]):
        prompt = ' '.join(text.split(' ')[:-1])
        target = text.split(' ')[-1]
        next_token_probs = get_next_token_probs(model, tokenizer, prompt)
        all_probs_SO_removed.append(next_token_probs)

    # Average probabilities acorss all samples
    avg_probs_SO_removed = (sum(all_probs_SO_removed) / len(all_probs_SO_removed))[0]
    avg_probs_SO_removed = avg_probs_SO_removed.to('cpu')
    selected_token_probs_SO_removed = []
    for i in top_n_indices_Original:
        selected_token_probs_SO_removed.append(avg_probs_SO_removed[i].item())

    print("Top n tokens:")
    print(top_tokens)
    print("Original")
    print(top_n_probs_Original)
    print("SW removed")
    print(selected_token_probs_SO_removed)
    



if __name__ == '__main__':
    run([plot_down_proj_input_output, plot_max_activation_ablation, plot_token_probs])

```

### evaluate.py

```py
from lm_eval.__main__ import cli_evaluate
import outliers.model  # support hf-outliers model by importing into namespace

if __name__ == '__main__':
    cli_evaluate()
```

### figures.ipynb


### outliers/functional/quantization.py

```py
import torch



def pack_4bit_to_int8(quantized_4bit_weights):
    # Ensure the quantized weights are uint8
    quantized_4bit_weights = quantized_4bit_weights.to(torch.uint8)
    
    # Reshape to ensure even number of elements per row
    if quantized_4bit_weights.size(1) % 2 != 0:
        quantized_4bit_weights = torch.cat((quantized_4bit_weights, torch.zeros((quantized_4bit_weights.size(0), 1), dtype=torch.uint8)), dim=1)
    
    # Packing two 4-bit values into one int8
    packed_weights = (quantized_4bit_weights[:, ::2] << 4) | (quantized_4bit_weights[:, 1::2] & 0xF)
    
    return packed_weights



def unpack_int8_to_4bit(packed_weights):
    # Unpack two 4-bit values from one int8
    high_bits = (packed_weights >> 4) & 0xF
    low_bits = packed_weights & 0xF
    
    # Interleave the high and low bits
    unpacked_weights = torch.stack((high_bits, low_bits), dim=2).flatten(1)
    
    return unpacked_weights

def dequantize_4bit_to_fp16(quantized, maxima, minima, nbits, blocksize):
    shape = quantized.shape
    
    if quantized.numel() % (blocksize * blocksize) == 0:
        quantized = quantized.reshape(-1, blocksize * blocksize)
        scale = (2 ** nbits - 1) / (maxima - minima)
        
        dequantized = (quantized / scale) + minima
        dequantized = dequantized.reshape(shape)
    else:
        scale = (2 ** nbits - 1) / (maxima - minima)
        dequantized = (quantized / scale) + minima

    return dequantized


def round_to_nearest_pole(x, poles):
    # Compute the absolute differences between each element of x and all poles
    differences = torch.abs(x.unsqueeze(-1) - poles)
    
    # Find the index of the minimum difference along the poles dimension
    nearest_indices = torch.argmin(differences, dim=-1)
    
    # Return the corresponding nearest pole values for each element in x
    nearest_values = poles[nearest_indices]
    
    return nearest_values


def quantize_blockwise(weight, nbits, blocksize, clip_method, clip_threshold, scale_shift=False, use_normal_float=False):
    # weight should be param.data
    shape = weight.shape
    dtype = weight.dtype
    _num_outliers = 0
    if weight.numel() % (blocksize) == 0:
        weight = weight.reshape(-1, blocksize)
        # block wise
        if clip_method != "no":

            if clip_method == "block_percentage":
                num_top_elements = max(int(weight.size(1) * clip_threshold + 1), 1) # per block
                threshold = torch.topk(weight.abs(), num_top_elements, dim=1).values[:, -1].unsqueeze(-1)

            elif clip_method == "tensor_percentage":
                num_top_elements = max(int(weight.numel() * clip_threshold), 1)
                threshold = torch.topk(weight.view(-1).abs(), num_top_elements).values[-1]

            elif clip_method == "zscore":
                means = weight.abs().mean(dim=1, keepdim=True)
                stds = weight.abs().std(dim=1, keepdim=True)
                threshold = clip_threshold * stds + means

            elif clip_method == "iqr":
                q1 = weight.abs().float().quantile(0.25, dim=1, keepdim=True)
                q3 = weight.abs().float().quantile(0.75, dim=1, keepdim=True)
                threshold = q3 + clip_threshold * (q3 - q1) # 1.5 * IQR
                threshold = threshold.to(weight.dtype)
            else:
                raise ValueError(f"Unknown clip method: {clip_method}")

            _num_outliers = int(torch.sum(weight.abs() > threshold ))
            weight = torch.clamp(weight, -threshold, threshold)
        minima, _ = weight.min(dim=1, keepdims=True)
        maxima, _ = weight.max(dim=1, keepdims=True)

    else:
        if clip_method != "no":
            if clip_method == "tensor_percentage" or clip_method == "block_percentage":
                num_top_elements = max(int(weight.numel() * clip_threshold), 1)
                threshold = torch.topk(weight.view(-1).abs(), num_top_elements).values[-1]
            elif clip_method == "zscore":
                mean = weight.abs().mean()
                std = weight.abs().std()
                threshold = clip_threshold * std + mean 
            elif clip_method == "iqr":
                q1 = weight.abs().float().quantile(0.25)
                q3 = weight.abs().float().quantile(0.75)
                threshold = q3 + clip_threshold * (q3 - q1) # 1.5 * IQR
            # count clipped outliers
            _num_outliers = int(torch.sum(weight.abs() > threshold))
            weight = torch.clamp(weight, -threshold, threshold)
        minima, maxima = weight.min(), weight.max()


    if use_normal_float: # NF4 or NF3
        NF4_LEVELS = [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
        
        NF3_LEVELS = [-1, -0.5350227355957031, -
        0.2469314038753510, 0, 0.1833375245332718, 0.3819939494132996, 0.6229856610298157, 1]
        if nbits == 4:
            quantization_levels = NF4_LEVELS
        elif nbits == 3:
            quantization_levels = NF3_LEVELS
        else:
            raise ValueError("Normal Float Quantization only suuports 4 and 3 bits now.")
        quantization_levels = torch.tensor(quantization_levels).to(weight.device)
        scale = 2 / (maxima - minima) # scale to [0, 2]
        weight.sub_(minima).mul_(scale).sub_(1.0) # shift to [-1, 1]
        weight.copy_(round_to_nearest_pole(weight, quantization_levels))
        weight.add_(1).div_(scale).add_(minima)


    else: # INT
        if scale_shift:
            # mapping weights to [-0.4999, 15.4999] and then round to nearest integers
            scale = (2 ** nbits - 0.01) / (maxima - minima)
            weight.sub_(minima).mul_(scale).sub_(0.49)
            weight.round_()
            weight.add_(0.49).div_(scale).add_(minima)
        else:
            # mapping weights to [0, 15] and then round to nearest integers
            scale = (2 ** nbits - 1) / (maxima - minima)
            weight.sub_(minima).mul_(scale)
            weight.round_()
            weight.div_(scale).add_(minima)

    dequantized = weight.reshape(shape).to(dtype)
    return dequantized, _num_outliers
        
```

### outliers/functional/utils.py

```py
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaSdpaAttention, apply_rotary_pos_emb, repeat_kv, LlamaConfig
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("outputs/figures", exist_ok=True)

def add_global_plot_styles(multiplier=1):
    plt.rcParams["font.family"] = "serif"
    plt.grid(color='#CCCCCC', linestyle='--')
    plt.rcParams.update({
        'axes.titlesize': 17 * multiplier,
        'axes.labelsize': 15 * multiplier,
        'xtick.labelsize': 10 * multiplier,
        'ytick.labelsize': 10 * multiplier,
    })
    plt.tight_layout()


def print_box(text):
    lines = text.splitlines()
    length = max(map(len, lines))
    border = '*' * (length + 4)

    print(border)
    for line in lines:
        print(f"* {line} *")
    print(border)



class ScaledLlamaAttention(LlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, scale: Optional[float] = None):
        super().__init__(config, layer_idx)
        self.attn_scale = scale
    

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        scale = 1 / math.sqrt(self.head_dim) if self.attn_scale is None else self.attn_scale # here we customize scale
        # scale = 1 / math.sqrt(self.head_dim) 
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scale

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



class ScaledLlamaSdpaAttention(ScaledLlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.attn_scale, # pass a scale argument to control softmax tempearature
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

```

### outliers/model.py

```py
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import torch
import transformers
from packaging import version
from peft import PeftModel
from peft import __version__ as PEFT_VERSION
from transformers import BitsAndBytesConfig

from lm_eval.models.huggingface import HFLM
from lm_eval import utils
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
import re
from pathlib import Path

from outliers.functional.quantization import quantize_blockwise

eval_logger = utils.eval_logger


SUPER_WEIGHTS_MAP = {
    "Mistral-7B-v0.1": [(1, 2070, 7310)],
    "llama-7B": [(2, 3968, 7003)],
    "llama-13B": [(2, 2231, 2278), (2, 2231, 6939)],
    "llama-30B": [(3, 5633, 12817), (3, 5633, 17439), (10, 5633, 14386)],
    "Meta-Llama-3-8B": [(1, 788, 2427), (1, 1384, 2427), (1, 4062, 2427)],
    "OLMo-1B-0724-hf": [(1, 1764, 1710), (2, 1764, 8041)],
    "OLMo-7B-0724-hf": [(1, 269, 7467), (2, 269, 8275), (7, 269, 453), (24, 269, 2300)],
    "Phi-3-mini-4k-instruct": [(2, 525, 808), (2, 1693, 808), (2, 1113, 808), (4, 525, 2723),  (4, 1113, 2723), (4, 1693, 2723)],
}


def get_layer_number(layer_name):
    # Define the pattern to extract the layer number
    match = re.search(r"layers\.(\d+)\.", layer_name)
    if match:
        return int(match.group(1))
    else:
        return None


def get_weight_type(name):
    for linear in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        if linear in name:
            return linear
    return "other"


def pack_4bit_to_int8(quantized_4bit_weights):
    # Ensure the quantized weights are uint8
    quantized_4bit_weights = quantized_4bit_weights.to(torch.uint8)
    
    # Reshape to ensure even number of elements per row
    if quantized_4bit_weights.size(1) % 2 != 0:
        quantized_4bit_weights = torch.cat((quantized_4bit_weights, torch.zeros((quantized_4bit_weights.size(0), 1), dtype=torch.uint8)), dim=1)
    
    # Packing two 4-bit values into one int8
    packed_weights = (quantized_4bit_weights[:, ::2] << 4) | (quantized_4bit_weights[:, 1::2] & 0xF)
    
    return packed_weights


def _get_accelerate_args(
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
    gpus: Optional[int] = None,
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu for device_idx in range(gpus)
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


@register_model("hf-outlier")
class HFOutlierLM(HFLM):
    def _create_model(
        self,
        pretrained: str,
        revision: Optional[str] = "main",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        trust_remote_code: Optional[bool] = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: Optional[bool] = False,
        gpus: Optional[int] = None,
        device_map_option: Optional[str] = "auto",
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[str] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        outlier_method: Optional[str] = None, # added argument for outlier experiments
        manual_quantize: Optional[str] = None,
        restore_and_scale_GO: Optional[Union[bool, float]] = False,
        bnb_quantize_args: Optional[str] = None,
        clip_and_save_args: Optional[str] = None,
        load_quantized_args: Optional[str] = None,
        adjust_attn_args: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """

        model_kwargs = kwargs if kwargs else {}

        if parallelize:
            model_kwargs.update(
                _get_accelerate_args(
                    device_map_option,  # TODO: phase out device_map_option?
                    max_memory_per_gpu,
                    max_cpu_memory,
                    offload_folder,
                    gpus,
                )
            )
        elif "device_map" not in model_kwargs:
            # set a device_map to initialize model on the right GPU.
            # this is needed because it seems that the default behavior
            # for quantized models now seems to be device_map="auto"
            # which breaks data-parallel mode.
            if hasattr(self, "accelerator"):
                model_kwargs.update({"device_map": {"": f"{self.accelerator.device}"}})
            else:
                model_kwargs.update({"device_map": {"": str(self.device)}})

        if not autogptq:
            if model_kwargs.get("load_in_4bit", None):
                assert (
                    transformers.__version__ >= "4.30.0"
                ), "load_in_4bit requires transformers >= 4.30.0"
            if transformers.__version__ >= "4.30.0":
                if model_kwargs.get("load_in_4bit", None):
                    if model_kwargs.get("bnb_4bit_compute_dtype", None):
                        model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(
                            model_kwargs["bnb_4bit_compute_dtype"]
                        )         


            if bnb_quantize_args:
                load_in_4bit, bnb_4bit_quant_type, blocksize, clip_percentage = bnb_quantize_args.split('_') # True_fp4_512_1e-3
                load_in_4bit = True if load_in_4bit == "True" else False
                blocksize = int(blocksize)
                print("bnb args", load_in_4bit, blocksize, bnb_4bit_quant_type)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_quant_type=bnb_4bit_quant_type,
                    blocksize=blocksize
                )
                model_kwargs["load_in_4bit"] = False
                model_kwargs["quantization_config"] = quantization_config    



            self._model = self.AUTO_MODEL_CLASS.from_pretrained(
                pretrained,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            # Record SW value
            self.record_GO(pretrained)   
            if adjust_attn_args: # Now is working for Llama-7B only
                import math
                from outliers.functional.utils import ScaledLlamaSdpaAttention
                tau = float(adjust_attn_args)
                scale = 1 / (math.sqrt(self._model.model.layers[3].self_attn.head_dim) * tau) 

                def remove_massive_activation_llama(module, input, output):
                    # print(get_module_name(model, module))
                    hidden_states = output[0][0]
                    # Modify the hidden states here

                    hidden_states[0, 3968] *= tau
                    modified_output = hidden_states
                    output[0][0] = modified_output
                    return output


                self._model.model.layers[3].self_attn = ScaledLlamaSdpaAttention(config=self._model.config, layer_idx=3)

                all_hooks = []
                remove_massive_activation_hook = self._model.model.layers[2].register_forward_hook(remove_massive_activation_llama)
                all_hooks.append(remove_massive_activation_hook)


                self._model = self._model.to('cuda').to(torch.bfloat16) # hard code dtype for now. might need to adjust to arguments


            if manual_quantize:
                # Parse the quantization method and parameters from the string
                quantize_method, bits, block_size_arg, clip_method, clip_threshold, scale_shift, use_normal_float = manual_quantize.split('_') # minmax_4_128_bp_1e-4
                bits = int(bits)

                block_size_arg = int(block_size_arg) if block_size_arg not in ["channel", "tensor"] else block_size_arg
                clip_threshold = float(clip_threshold) if clip_threshold != "None" else None 
                clip_method_map = {"bp": "block_percentage", "tp": "tensor_percentage", "z": "zscore", "iqr": "iqr",  "no": "no"}
                clip_method = clip_method_map[clip_method]
                scale_shift = True if scale_shift == "True" else False
                use_normal_float = True if use_normal_float == "True" else False

                if quantize_method == "minmax":
                    print(f"Running {bits} bits manual quantization ...")
                    for name, param in self._model.model.named_parameters():
                        if not name.endswith("weight"):
                            continue
                        if "layernorm" in name or "norm" in name or "embed_tokens" in name or "lm_head" in name:
                            continue
                        weight = param.data
                        shape = weight.shape
                        if block_size_arg == "channel":
                            block_size = shape[-1]
                        elif block_size_arg == "tensor":
                            block_size = weight.numel()
                        elif isinstance(block_size_arg, int):
                            block_size = block_size_arg
                        else:
                            raise ValueError("Block size must be int or 'tensor' or 'channel'")
                        quantized_weight, _num_outliers = quantize_blockwise(weight, bits, block_size, "no", 0, scale_shift, use_normal_float)
                        param.data = quantized_weight    
                
                elif quantize_method == "clip":
                    # clip outliers but restore super outliers
                    # quantize with min max
                    num_outliers = []
                    original_value_ranges = []
                    clipped_value_ranges = []

                    print(f"Running {bits} bits manual quantization ...")
                    print(f"Clipping parameters", clip_method, clip_threshold)
                    print(f"Restore GO?", restore_and_scale_GO)
                    for name, param in self._model.model.named_parameters():
                        if not name.endswith("weight"):
                            continue
                        if "layernorm" in name or "norm" in name or "embed_tokens" in name or "lm_head" in name:
                            continue

                        weight = param.data
                        shape = weight.shape
                        if block_size_arg == "channel":
                            block_size = shape[-1]
                        elif block_size_arg == "tensor":
                            block_size = weight.numel()
                        elif isinstance(block_size_arg, int):
                            block_size = block_size_arg
                        else:
                            raise ValueError("Block size must be int or 'tensor' or 'channel'") 
                        if weight.numel() % (block_size) != 0:
                            print(name,  weight.numel(), block_size)
                        quantized_weight, _num_outliers = quantize_blockwise(weight, bits, block_size, clip_method, clip_threshold, scale_shift, use_normal_float)
                        param.data = quantized_weight    
                        num_outliers.append(_num_outliers)
                    # print statistics
                    print(f"Number of outliers. Max: {max(num_outliers)}, Min: {min(num_outliers)}, Sum: {sum(num_outliers)}, Mean {sum(num_outliers) / len(num_outliers)}")


            if outlier_method:
                def get_weight_type(model_id, name):
                    if model_id.startswith("facebook/opt"):
                        for linear in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "fc1", "fc2"]:
                            if linear in name:
                                return linear   
                    else:

                        for linear in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                            if linear in name:
                                return linear
                    return "other"
                
                def get_layer_number(layer_name):
                    # Define the pattern to extract the layer number
                    match = re.search(r"layers\.(\d+)\.", layer_name)
                    if match:
                        return int(match.group(1))
                    else:
                        return None
                if outlier_method.startswith("manual_scaling_SO"): # for example, manual_scaling_SO_0.1
                    print("manual scaling SO")
                    scaling_factor = outlier_method.split('_')[-1]
                    scaling_factor = float(scaling_factor)
                    print("Original SO:", self.GO_values)
                    self.restore_GO(pretrained, scaling_factor)

                if outlier_method.startswith("removeSW_restoreSA"):
                    if pretrained == "huggyllama/llama-7B":
                        def hook_fn(module, input, output):
                            SW_channel = 3968
                            SW_row = 7003
                            # Record original hidden states
                            X1 = output.detach().clone()
                            # print(f"Shape of input/output: {output.shape}, {input[0].shape}")
                            
                            # Find the largest magnitude in channel SA and its position
                            channel_SA = X1[..., SW_channel]
                            # print(f"Shape of channel_SA: {channel_SA.shape}")
                            max_magnitude_indices = torch.argmax(torch.abs(channel_SA), dim=1)
                            # print(f"Max magnitude indices: {max_magnitude_indices}")
                            max_magnitudes = torch.gather(channel_SA, 1, max_magnitude_indices.unsqueeze(1)).squeeze(1)

                            
                            # Temporarily set SW to 0
                            weight = module.weight.clone()
                            original_weight = weight.data[SW_channel, SW_row].item()
                            weight.data[SW_channel, SW_row] = 0.0
                            
                            # Recompute hidden states
                            with torch.no_grad():
                                X2 = torch.nn.functional.linear(input[0], weight, module.bias)
                                # print(f"Shape of recomputed X2: {X2.shape}")
                            
                                batch_indices = torch.arange(X2.size(0), device=X2.device)
                                # print("New magnitude on SA position:", X2[batch_indices, max_magnitude_indices, 3968])
                                X2[batch_indices, max_magnitude_indices, SW_channel] = max_magnitudes

                            # Restore the original weight
                            module.weight.data[SW_channel, SW_row] = original_weight
                            return X2

                        # Register the hook to the specific layer
                        self._model.model.layers[2].mlp.down_proj.register_forward_hook(hook_fn)
                        print("Hook registered for removing SW and restoring SA")

                elif outlier_method.startswith("search"):
                    _, criterion, search_threhold, layers, outlier_weight_type = outlier_method.split("_") # "search_percentage_1e-4_0-26_downproj"
                    start_layer, end_layer = layers.split('-')
                    layers = range(int(start_layer), int(end_layer))
                    search_threhold = float(search_threhold)
                    # print(layers, search_threhold, criterion)
                    num_selected_elements = []
                    for name, param in self._model.model.named_parameters():
                        if not name.endswith("weight"):
                            continue
                        if "layernorm" in name or "norm" in name or "embed_tokens" in name or "lm_head" in name:
                            continue
                        if outlier_weight_type != "all":
                            if outlier_weight_type == "down":
                                outlier_weight_type = "down_proj"
                            weight_type = get_weight_type(pretrained, name)
                            if weight_type != outlier_weight_type:
                                # print("Skip for unmatched weight type")
                                continue
                        if layers is not None:
                            layer_number = get_layer_number(name)
                            if layer_number not in layers:
                                # print("Skip for unmatched layer")
                                continue
                        weight = param.data
                        if criterion == "percentage":
                            num_top_elements = int(weight.numel() * search_threhold)
                            # print("# Total params:", weight.numel(), "# top params", num_top_elements)

                            if num_top_elements < 1: # too few elements to apply on
                                continue 
                            threshold = torch.topk(weight.view(-1).abs(), num_top_elements).values[-1]
                            mask = weight.abs() >= threshold
                            true_indices = mask.nonzero(as_tuple=False)
                            num_selected_elements.append(len(true_indices))
                            # print(name, len(true_indices))
                            
                            weight[mask] = 0.
                        param = torch.nn.Parameter(weight) 
                    if len(num_selected_elements) > 0:
                        print(f"Number of removed elements. Max: {max(num_selected_elements)}, Min: {min(num_selected_elements)}, Sum: {sum(num_selected_elements)}, Mean {sum(num_selected_elements) / len(num_selected_elements)}")


            if restore_and_scale_GO:
                self.restore_GO(pretrained, restore_and_scale_GO)
            else:
                print("Not restoring or scaling GO...")

        else:
            try:
                from auto_gptq import AutoGPTQForCausalLM
            except ModuleNotFoundError:
                raise Exception(
                    "Tried to load auto_gptq, but auto-gptq is not installed ",
                    "please install auto-gptq via pip install lm-eval[gptq] or pip install -e .[gptq]",
                )

            self._model = AutoGPTQForCausalLM.from_quantized(
                pretrained,
                trust_remote_code=trust_remote_code,
                model_basename=None if autogptq is True else Path(autogptq).stem,
                use_safetensors=True
                if autogptq is True
                else autogptq.endswith(".safetensors"),
                **model_kwargs,
            )

        if peft and delta:
            raise ValueError(
                "Cannot use both 'peft' and 'delta' options at the same time."
            )

        if peft:
            if model_kwargs.get("load_in_4bit", None):
                if version.parse(PEFT_VERSION) < version.parse("0.4.0"):
                    raise AssertionError("load_in_4bit requires peft >= 0.4.0")
            if self._model.config.vocab_size != len(self.tokenizer):
                # resize model for LoRAs with added tokens
                self._model.resize_token_embeddings(len(self.tokenizer))
                eval_logger.info(
                    f"Model config indicates vocab_size='{self._model.config.vocab_size}', but found tokenizer with vocab size '{len(self.tokenizer)}'. Resizing model embedding layer..."
                )
            self._model = PeftModel.from_pretrained(
                self._model, peft, revision=revision
            )
        elif delta:
            if autogptq:
                eval_logger.warning(
                    "Delta weights might trigger unexpected behavior when used with AutoGPTQ."
                )
            _model_delta = self.AUTO_MODEL_CLASS.from_pretrained(
                delta,
                revision=revision,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )
            for name, param in self._model.state_dict().items():
                try:
                    param.data += _model_delta.state_dict()[name]
                except KeyError:
                    raise KeyError(f"Delta model is missing weights for layer: {name}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to add delta weights to layer {name}. Error: {e}"
                    )

            del _model_delta

        return None


    def record_GO(self, pretrained):
        '''Record GO values for original models'''
        def _record_GO(GO_map, layer, row, col):
            if pretrained in [
                "tiiuae/falcon-7b",
            ]:
                GO_map[(layer, row, col)] = self._model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col].item()
            else:
                GO_map[(layer, row, col)] = self._model.model.layers[layer].mlp.down_proj.weight.data[row, col].item()

        GO_values = {}
        for model_name, coordinates in SUPER_WEIGHTS_MAP.items():
            if model_name in pretrained:
                for layer, row, col in coordinates:
                    _record_GO(GO_values, layer, row, col)
                break
        self.GO_values = GO_values


    def restore_GO(self, pretrained, scaling_factor):
        if pretrained in [
            "huggyllama/llama-30B", 
            "huggyllama/llama-13B", 
            "huggyllama/llama-7B", 
            "mistralai/Mistral-7B-v0.1",
            "meta/Meta-Llama-3-8B",
            "allenai/OLMo-1B-0724-hf",
            "allenai/OLMo-7B-0724-hf",
            "google/gemma-7b",
            "microsoft/Phi-3-mini-4k-instruct"
            ]:
            if getattr(self._model, "hf_quantizer", None):
                self._model.dequantize()
            for (layer, row, col), value in self.GO_values.items():  
                old_value = self._model.model.layers[layer].mlp.down_proj.weight.data[row, col].item()
                new_value = value * scaling_factor
                self._model.model.layers[layer].mlp.down_proj.weight.data[row, col] = new_value
                print(f"Layer {layer}, Index [{row}, {col}], Old value: {old_value}, New value: {new_value}")

        elif pretrained in [
            "tiiuae/falcon-7b"
            ]:
            if getattr(self._model, "hf_quantizer", None):
                self._model.dequantize()
            for (layer, row, col), value in self.GO_values.items():  
                old_value = self._model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col].item()
                new_value = value * scaling_factor
                self._model.transformer.h[layer].mlp.dense_4h_to_h.weight.data[row, col] = new_value
                print(f"Layer {layer}, Index [{row}, {col}], Old value: {old_value}, New value: {new_value}")

```

### scripts/figure3_how_to_identify_superweight.sh

```sh
python analyze.py plot-down-proj-input-output
```

### scripts/figure4_how_superweight_affect_superactivation.sh

```sh
python analyze.py plot-max-activation-ablation
```

### scripts/figure5_token_probability.sh

```sh
python analyze.py plot-token-probs
```

### scripts/figure6_super_weight_scaling.sh

```sh
model_name=huggyllama/llama-7B

for scale in 0.8 1.0 1.2 1.5 2.0
do
    outlier_method=manual_scaling_SO_${scale}
    python evaluate.py --model hf-outlier \
        --model_args pretrained=${model_name},outlier_method=${outlier_method},dtype=float16 \
        --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
        --device cuda:0 \
        --batch_size 16 \
        --output_path outputs/${model_name}/sensitivity/${outlier_method} \

done
```

### scripts/run_SW_importance_ablation.sh

```sh
# model_name=huggyllama/llama-13B
# model_name=huggyllama/llama-7B
# model_name=mistralai/Mistral-7B-v0.1

# model_name=microsoft/Phi-3-mini-4k-instruct

outlier_method=search_percentage_1e-6_0-32_all
for restore_and_scale_GO in 1.0 0.0
do
    python evaluate.py --model hf-outlier \
        --model_args pretrained=${model_name},outlier_method=${outlier_method},restore_and_scale_GO=${restore_and_scale_GO},dtype=float16 \
        --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
        --device cuda:0 \
        --batch_size 16 \
        --output_path outputs/${model_name}/search/${outlier_method}_restore_and_scale-${restore_and_scale_GO} \

done


for outlier_method in manual_scaling_SO_0.0 manual_scaling_SO_1.0
do

python evaluate.py --model hf-outlier \
    --model_args pretrained=${model_name},outlier_method=${outlier_method},dtype=float16 \
    --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
    --device cuda:0 \
    --batch_size 16 \
    --output_path outputs/${model_name}/sensitivity/${outlier_method} \

done
```

### scripts/run_test_manual_ablation.sh

```sh
# model_name=huggyllama/llama-13B
model_name=huggyllama/llama-7B
\
outlier_method=search_percentage_5e-7_0-32_all
for restore_and_scale_GO in True False
do
    python evaluate.py --model hf-outlier \
        --model_args pretrained=${model_name},outlier_method=${outlier_method},restore_and_scale_GO=${restore_and_scale_GO} \
        --tasks wikitext \
        --device cuda:0 \
        --batch_size 16 \
        --output_path outputs/${model_name}/search/${outlier_method}_restore_and_scale-${restore_and_scale_GO} \

done
    # --model_args pretrained=huggyllama/llama-7b \
    # --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada \

```

### scripts/run_test_quantize_loop.sh

```sh



# model_name=mistralai/Mistral-7B-v0.1

# # Baseline: NF3, no scale-shift
# for blocksize in 64 128 256 512 1024 2048 channel
# do
#     manual_quantize=minmax_3_${blocksize}_no_0_False_True
#     restore_and_scale_GO=False
#     python evaluate.py --model hf-outlier \
#         --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
#         --tasks wikitext \
#         --device cuda:0 \
#         --batch_size 4 \
#         --output_path outputs/${model_name}/groupwise/nf3/minmax/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
#         --trust_remote_code \

# done



# # Baseline: NF4, no scale-shift
# for blocksize in 64 128 256 512 1024 2048 4096 channel
# do
#     manual_quantize=minmax_4_${blocksize}_no_0_False_True
#     restore_and_scale_GO=False
#     python evaluate.py --model hf-outlier \
#         --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
#         --tasks wikitext \
#         --device cuda:0 \
#         --batch_size 4 \
#         --output_path outputs/${model_name}/groupwise/nf4/minmax/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
#         --trust_remote_code \

# done



# model_name=allenai/OLMo-7B-0724-hf
model_name=mistralai/Mistral-7B-v0.1
# Baseline: INT4, no scale-shift
# for blocksize in tensor 1048576 262144 65536 16384
# do
#     manual_quantize=minmax_4_${blocksize}_no_0_False_False
#     restore_and_scale_GO=False
#     python evaluate.py --model hf-outlier
#         --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
#         --tasks wikitext,winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
#         --device cuda:0 \
#         --batch_size 4 \
#         --output_path outputs/${model_name}/groupwise/int4/minmax/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
#         --trust_remote_code \

# done




for blocksize in 65536
do
    restore_and_scale_GO=1.0
    for manual_quantize in clip_4_${blocksize}_z_9_False_False clip_4_${blocksize}_z_11_False_False clip_4_${blocksize}_tp_1e-6_False_False
    do
        python evaluate.py --model hf-outlier
            --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
            --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
            --device cuda:0 \
            --batch_size 4 \
            --output_path outputs/${model_name}/groupwise/int4/ours/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
            --trust_remote_code \

    done

done




# # # Ours: NF3, clip + restore
# for blocksize in tensor 1048576 262144 65536 16384
# do
#     restore_and_scale_GO in 1.0
#     for manual_quantize in clip_4_${blocksize}_z_12_False_False clip_4_${blocksize}_bp_5e-5_False_True clip_4_${blocksize}_bp_1e-6_False_False clip_4_${blocksize}_z_20_False_False
#     do
#         python evaluate.py --model hf-outlier
#             --model_args pretrained=${model_name},manual_quantize=${manual_quantize},restore_and_scale_GO=${restore_and_scale_GO},trust_remote_code=True,dtype=float16 \
#             --tasks wikitext,winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
#             --device cuda:0 \
#             --batch_size 4 \
#             --output_path outputs/${model_name}/groupwise/int4/ours/manual_${manual_quantize}_restore_scale-${restore_and_scale_GO}_core \
#             --trust_remote_code \

#     done

# done


            # --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
```

### scripts/run_test_sensitivity.sh

```sh
model_name=huggyllama/llama-7B

for scale in 0.0 0.2 0.5 0.8 1.0 1.5 2.0 3.0
do
    outlier_method=manual_scaling_SO_${scale}
    python evaluate.py --model hf-outlier \
        --model_args pretrained=${model_name},outlier_method=${outlier_method},dtype=float16 \
        --tasks winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai \
        --device cuda:0 \
        --batch_size 16 \
        --output_path outputs/${model_name}/sensitivity/${outlier_method} \

done
```

### scripts/table1_superweight_importance.sh

```sh
model_name=huggyllama/llama-7B
tasks=winogrande,arc_challenge,arc_easy,piqa,sciq,hellaswag,lambada_openai

# Run evaluation on the original model
python evaluate.py --model hf \
    --model_args pretrained=${model_name},dtype=float16 \
    --tasks ${tasks} \
    --device cuda:0 \
    --batch_size 16 \
    --output_path outputs/original;

# Run evaluation removing super weight, then removing super weight but keeping
# the induced super activation
for outlier_method in manual_scaling_SO_0.0 removeSW_restoreSA;
do
    python evaluate.py --model hf-outlier \
        --model_args pretrained=${model_name},outlier_method=${outlier_method},dtype=float16 \
        --tasks $tasks \
        --device cuda:0 \
        --batch_size 16 \
        --output_path outputs/${outlier_method};
done
```

### scripts/table2_find_superweight.sh

```sh

```

