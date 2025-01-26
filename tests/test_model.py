"""Tests for the model module."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from superweights.model import TransformerAnalyzer


@pytest.fixture
def sample_model_name():
    """Sample model name for testing."""
    return "gpt2"  # Use small model for testing


@pytest.fixture
def analyzer(sample_model_name):
    """Create a TransformerAnalyzer instance."""
    return TransformerAnalyzer(sample_model_name, device="cpu", dtype=torch.float32)


def test_transformer_analyzer_init(sample_model_name):
    """Test TransformerAnalyzer initialization."""
    analyzer = TransformerAnalyzer(sample_model_name, device="cpu")
    assert isinstance(analyzer.model, AutoModelForCausalLM)
    assert isinstance(analyzer.tokenizer, AutoTokenizer)


def test_transformer_analyzer_init_invalid_model():
    """Test TransformerAnalyzer initialization with invalid model."""
    with pytest.raises(OSError):
        TransformerAnalyzer("nonexistent-model")


def test_find_superweights(analyzer):
    """Test finding superweights in model."""
    threshold = 0.1
    superweights = analyzer.find_superweights(threshold=threshold)
    assert isinstance(superweights, dict)
    
    # Test with specific weight type
    superweights = analyzer.find_superweights(threshold=threshold, weight_type="query")
    assert isinstance(superweights, dict)
    
    # Test with specific layers
    superweights = analyzer.find_superweights(threshold=threshold, layers=[0, 1])
    assert isinstance(superweights, dict)
    
    # Test with absolute criterion
    superweights = analyzer.find_superweights(threshold=threshold, criterion="absolute")
    assert isinstance(superweights, dict)
    
    # Verify position tuples
    for positions in superweights.values():
        assert all(isinstance(pos, tuple) and len(pos) == 2 for pos in positions)


def test_find_superweights_invalid_args(analyzer):
    """Test find_superweights with invalid arguments."""
    # Test invalid threshold
    with pytest.raises(ValueError, match="Threshold must be positive"):
        analyzer.find_superweights(threshold=-1)
    
    # Test invalid weight type
    with pytest.raises(ValueError):
        analyzer.find_superweights(weight_type="invalid_type")
    
    # Test invalid criterion
    with pytest.raises(ValueError, match="Criterion must be 'percentage' or 'absolute'"):
        analyzer.find_superweights(criterion="invalid_criterion")


def test_analyze_activations(analyzer):
    """Test analyzing activations."""
    text = "Hello, world!"
    activations = analyzer.analyze_activations(text)
    assert isinstance(activations, dict)
    assert len(activations) > 0
    
    # Test with specific layers
    layer_indices = [0, 1]
    activations = analyzer.analyze_activations(text, layer_indices=layer_indices)
    assert isinstance(activations, dict)
    assert set(activations.keys()) == set(layer_indices)


def test_analyze_activations_invalid_args(analyzer):
    """Test analyze_activations with invalid arguments."""
    # Test empty text
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        analyzer.analyze_activations("")
    
    # Test empty layer indices
    with pytest.raises(ValueError, match="Layer indices list cannot be empty"):
        analyzer.analyze_activations("Hello", layer_indices=[])
    
    # Test negative layer indices
    with pytest.raises(ValueError, match="Layer indices must be non-negative"):
        analyzer.analyze_activations("Hello", layer_indices=[-1])
    
    # Test out of range layer indices
    with pytest.raises(ValueError, match="Layer index .* is out of range"):
        analyzer.analyze_activations("Hello", layer_indices=[100])


def test_remove_superweights(analyzer):
    """Test removing superweights."""
    text = "Hello, world!"
    threshold = 0.1
    
    # Find superweights first
    superweights = analyzer.find_superweights(threshold=threshold)
    
    # Test zero method
    analyzer.remove_superweights(superweights, method="zero")
    
    # Test mean method
    analyzer.remove_superweights(superweights, method="mean")


def test_remove_superweights_invalid_args(analyzer):
    """Test remove_superweights with invalid arguments."""
    text = "Hello, world!"
    superweights = {"layer.0.weight": [(0, 0)]}
    
    # Test invalid removal method
    with pytest.raises(ValueError, match="Method must be 'zero' or 'mean'"):
        analyzer.remove_superweights(superweights, method="invalid_method")
    
    # Test empty superweights
    with pytest.raises(ValueError, match="Superweights dictionary cannot be empty"):
        analyzer.remove_superweights({})


def test_remove_superweights_and_get_modified_output(analyzer):
    """Test removing superweights and getting modified output."""
    text = "Hello, world!"
    threshold = 0.1
    
    # Find superweights first
    superweights = analyzer.find_superweights(threshold=threshold)
    
    # Test zero method
    modified_output = analyzer.remove_superweights_and_get_modified_output(
        text, superweights, method="zero")
    assert isinstance(modified_output, torch.Tensor)
    assert modified_output.shape[-1] == analyzer.model.config.vocab_size
    
    # Test mean method
    modified_output = analyzer.remove_superweights_and_get_modified_output(
        text, superweights, method="mean")
    assert isinstance(modified_output, torch.Tensor)
    assert modified_output.shape[-1] == analyzer.model.config.vocab_size


def test_evaluate_impact(analyzer):
    """Test evaluating superweight impact."""
    text = "Hello, world!"
    threshold = 0.1
    
    # Find superweights
    superweights = analyzer.find_superweights(threshold=threshold)
    
    # Evaluate impact
    impact_metrics = analyzer.evaluate_impact(text, superweights)
    assert isinstance(impact_metrics, dict)
    assert "output_diff_norm" in impact_metrics
    assert "relative_change" in impact_metrics


def test_evaluate_impact_with_modified_output(analyzer):
    """Test evaluating impact with pre-computed modified output."""
    text = "Hello, world!"
    
    # Get original output
    inputs = analyzer.tokenizer(text, return_tensors="pt").to(analyzer.device)
    with torch.no_grad():
        original_output = analyzer.model(**inputs).logits
        modified_output = original_output.clone()  # Just for testing
    
    # Evaluate impact
    impact_metrics = analyzer.evaluate_impact_with_modified_output(
        text, original_output, modified_output)
    assert isinstance(impact_metrics, dict)
    assert "output_diff_norm" in impact_metrics
    assert "relative_change" in impact_metrics
