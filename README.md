# SuperWeights: Discovering Critical Weights in Transformer Models

A Python toolkit for discovering and analyzing critical weights (superweights) in transformer models. This tool helps identify weights that have outsized importance in model behavior, enabling better understanding and optimization of transformer models.

## Supported Models

The library has been tested with several prominent language models:
- Mistral-7B-v0.1
- LLaMA (7B, 13B, 30B)
- Meta-Llama-3-8B
- OLMo (1B, 7B)
- Phi-3-mini-4k
- And other Hugging Face transformer models

## Features

- **Superweight Discovery**: 
  - Identify critical weights using magnitude-based analysis
  - Support for percentage-based or absolute threshold detection
  - Layer-specific and weight-type-specific analysis
- **Impact Analysis**: 
  - Measure the influence of discovered weights on model output
  - Track activation patterns related to critical weights
  - Quantify importance through ablation studies
- **Quantization Tools**:
  - Intelligent quantization that preserves critical weights
  - Block-wise quantization with multiple methods
  - Support for various bit widths (2-8 bits)
- **Visualization**: Comprehensive tools for visualizing weight distributions and impacts
- **Model Manipulation**: Tools for experimenting with weight modifications and measuring their effects

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from superweights.model import TransformerAnalyzer

# Initialize analyzer with any transformer model
analyzer = TransformerAnalyzer("mistralai/Mistral-7B-v0.1")

# Discover critical weights using percentage-based analysis
# This finds the top 0.01% most significant weights by magnitude
superweights = analyzer.find_superweights(
    threshold=0.0001,  # 0.01%
    weight_type="down_proj",  # Focus on down-projection matrices
    criterion="percentage"
)

# Or discover using absolute threshold
superweights_abs = analyzer.find_superweights(
    threshold=0.5,  # Find weights with magnitude > 0.5
    criterion="absolute"
)

# Analyze impact of discovered weights
impact = analyzer.evaluate_impact("Hello, world!", superweights)
print(f"Impact metrics of critical weights: {impact}")

# Analyze activations to understand behavior
activations = analyzer.analyze_activations(
    text="Hello, world!",
    layer_indices=[1, 2, 3]  # Analyze specific layers
)
```

## Project Structure

```
superweights/
├── superweights/
│   ├── __init__.py
│   ├── model.py           # Core model analysis functionality
│   ├── quantization.py    # Weight quantization tools
│   ├── visualization.py   # Plotting and visualization tools
│   └── utils.py          # Utility functions
├── tests/                 # Test files
├── docs/                 # Documentation
└── requirements.txt     # Package dependencies
```

## Documentation

For detailed documentation, visit our [GitHub Pages](https://dsamuelhodge.github.io/superweights-llm/).

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@misc{superweights2025,
  title={SuperWeights: Identifying Critical Weights in Large Language Models},
  year={2025},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/DSamuelHodge/superweights-llm}}
}
