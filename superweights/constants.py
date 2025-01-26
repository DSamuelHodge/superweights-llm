"""Constants used across the superweights package."""

# Common linear projection layers in transformer models
LINEAR_PROJECTIONS = {
    "default": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "opt": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "fc1",
        "fc2"
    ]
}

# Default parameters for superweight analysis
DEFAULT_PARAMS = {
    "threshold": 1e-4,
    "quantization_bits": 4,
    "block_size": 128,
    "clip_method": "block_percentage",
}

# Model architecture specific configurations
MODEL_CONFIGS = {
    "llama": {
        "attention_module": "self_attn",
        "mlp_module": "mlp",
        "layer_prefix": "layers"
    },
    "opt": {
        "attention_module": "self_attn",
        "mlp_module": "fc",
        "layer_prefix": "layers"
    },
    "gpt2": {
        "attention_module": "attn",
        "mlp_module": "mlp",
        "layer_prefix": "h"
    }
}
