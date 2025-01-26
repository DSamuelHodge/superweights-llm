# SuperWeights: Analyzing Critical Weights in Large Language Models

A Python toolkit for identifying and analyzing superweights in transformer models. Superweights are individual weights that have outsized importance in model behavior, particularly in the down-projection matrices of transformer blocks.

## Supported Models

The library has been tested with several prominent language models:
- Mistral-7B-v0.1
- LLaMA (7B, 13B, 30B)
- Meta-Llama-3-8B
- OLMo (1B, 7B)
- Phi-3-mini-4k
- And other Hugging Face transformer models

## Features

- **Superweight Detection**: Identify critical weights in transformer models using various criteria
- **Weight Analysis**: 
  - Analyze weight distributions and patterns
  - Quantify weight importance through ablation studies
  - Track activation patterns and their relationship to superweights
- **Quantization Tools**:
  - Block-wise quantization with multiple methods
  - Outlier-aware quantization
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

# Initialize analyzer with a model
analyzer = TransformerAnalyzer("mistralai/Mistral-7B-v0.1")

# Find superweights
superweights = analyzer.find_superweights(
    threshold=1e-4,
    weight_type="down_proj",
    criterion="percentage"
)

# Evaluate their impact
impact = analyzer.evaluate_impact("Hello, world!", superweights)
print(f"Impact of superweights: {impact}")
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
