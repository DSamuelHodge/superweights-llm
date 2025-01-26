# SuperWeights Analysis

A toolkit for analyzing and manipulating transformer model weights, with support for any Hugging Face transformer model.

## Features

- Identify and analyze superweights in transformer models
- Support for any Hugging Face transformer model
- Weight quantization and analysis tools
- Activation analysis and visualization
- Model performance evaluation tools

## Installation

```bash
pip install -e .
```

## Usage

```python
from superweights.model import TransformerAnalyzer
from superweights.utils import plot_weight_distribution

# Initialize analyzer with any HF model
analyzer = TransformerAnalyzer("bert-base-uncased")

# Analyze weights
results = analyzer.analyze_weights()

# Visualize results
plot_weight_distribution(results)
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
├── examples/             # Example notebooks and scripts
├── setup.py             # Package setup file
└── requirements.txt     # Package dependencies
```
