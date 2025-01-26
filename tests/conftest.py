"""Pytest configuration file."""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@pytest.fixture
def sample_model_name():
    """Return a small model name for testing."""
    return "sshleifer/tiny-gpt2"

@pytest.fixture
def sample_model(sample_model_name):
    """Return a small model for testing."""
    model = AutoModelForCausalLM.from_pretrained(
        sample_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return model

@pytest.fixture
def sample_tokenizer(sample_model_name):
    """Return a tokenizer for testing."""
    return AutoTokenizer.from_pretrained(sample_model_name)

@pytest.fixture
def sample_text():
    """Return a sample text for testing."""
    return "This is a test input for the model."

@pytest.fixture
def sample_weight_tensor():
    """Return a sample weight tensor for testing."""
    return torch.randn(64, 64)
