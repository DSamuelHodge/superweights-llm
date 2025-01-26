# SuperWeights

SuperWeights is a Python library for analyzing and manipulating transformer model weights, with a focus on identifying and studying superweights - weights that have outsized importance in model behavior.

## Features

- **Weight Analysis**: Identify and analyze superweights in transformer models
- **Quantization**: Advanced weight quantization with multiple methods
- **Model Manipulation**: Tools for modifying and experimenting with model weights
- **Visualization**: Utilities for visualizing weight distributions and impacts

## Installation

```bash
pip install superweights
```

## Quick Start

```python
from superweights import TransformerAnalyzer

# Initialize analyzer with a model
analyzer = TransformerAnalyzer("gpt2")

# Find superweights
superweights = analyzer.find_superweights(threshold=1e-4)

# Analyze their impact
impact = analyzer.evaluate_impact("Hello, world!", superweights)
print(f"Impact of superweights: {impact}")
```

## Documentation

- [Installation Guide](guide/installation.md)
- [Quick Start Guide](guide/quickstart.md)
- [API Reference](api/model.md)
- [Examples](guide/examples.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details.
